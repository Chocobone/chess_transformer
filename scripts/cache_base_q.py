"""동결 베이스의 q(s, a)를 모든 국면·합법수에 대해 미리 계산해 둔다.

    python scripts/cache_base_q.py --player tal --base base_model/270M.pt

베이스는 학습 중 변하지 않으므로 한 번만 계산하면 된다. 기력 항, 모방 항 가중치, 베이스 기준선 평가가
이 캐시를 쓴다. 출력은 `data/processed/{player}/base_q_{베이스 이름}.npz`이고 형식은 `src/data/dataset.py`에 있다.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SPLITS, cache_path, player_dir, read_records  # noqa: E402
from src.model.transformer import ActionValueTransformer, encode_position, load_base  # noqa: E402
from src.players import PLAYERS  # noqa: E402


@torch.no_grad()
def compute_base_q(
    model: ActionValueTransformer,
    fens: list[str],
    batch_moves: int = 256,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """국면 목록 → (offsets[N + 1], actions[M], q[M]). 수의 순서는 `encode_position`의 순서다."""
    device = model.head.weight.device
    counts, actions, q = [], [], []
    pending, pending_moves = [], 0

    def flush() -> None:
        nonlocal pending, pending_moves
        if not pending:
            return
        tokens = torch.from_numpy(np.concatenate(pending)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            q.append(model.win_prob(tokens).float().cpu().numpy())
        pending, pending_moves = [], 0

    for fen in fens:
        tokens, _ = encode_position(fen)
        if len(tokens) == 0:
            raise ValueError(f"합법수가 없는 국면이다: {fen}")
        if pending_moves + len(tokens) > batch_moves:
            flush()
        pending.append(tokens)
        pending_moves += len(tokens)
        counts.append(len(tokens))
        actions.append(tokens[:, -2])
    flush()

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    empty = np.zeros(0)
    return (
        offsets,
        np.concatenate(actions or [empty]).astype(np.int16),
        np.concatenate(q or [empty]).astype(np.float32),
    )


def build_cache(
    model: ActionValueTransformer,
    data_dir: Path,
    base: str | Path,
    batch_moves: int = 256,
    amp: bool = False,
) -> Path:
    arrays = {}
    for split in SPLITS:
        fens = [record["fen"] for record in read_records(data_dir, split)]
        start = time.time()
        offsets, actions, q = compute_base_q(model, fens, batch_moves, amp)
        arrays[f"{split}_offsets"], arrays[f"{split}_actions"], arrays[f"{split}_q"] = offsets, actions, q
        print(f"{split}: 국면 {len(fens):,}개, 합법수 {len(q):,}개, {time.time() - start:.1f}초")
    path = cache_path(data_dir, base)
    np.savez(path, **arrays)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--player", required=True, choices=sorted(PLAYERS))
    parser.add_argument("--base", required=True, help="convert.py가 만든 .pt")
    parser.add_argument("--data-dir", type=Path, default=None, help="기본값: data/processed/{player}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-moves", type=int, default=256,
                        help="한 번에 순전파하는 합법수 수. 키워도 빨라지지 않고 GPU 메모리가 넘치면 급격히 느려진다")
    parser.add_argument("--amp", action="store_true",
                        help="bfloat16 혼합 정밀도로 계산한다 (빠르지만 q가 조금 달라진다)")
    args = parser.parse_args()

    model = load_base(args.base, args.device).eval()
    path = build_cache(model, args.data_dir or player_dir(args.player), args.base, args.batch_moves, args.amp)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
