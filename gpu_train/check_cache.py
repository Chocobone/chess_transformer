"""베이스 q 캐시(`base_q_{베이스}.npz`) 검증 (Task #14 해석 7).

① 구조: split마다 국면 수가 데이터와 같고, offsets가 0에서 시작해 늘어나며, 국면별 합법수 수와
   행동 번호가 `encode_position`(학습·캐시와 같은 순서)과 같다. q는 유한하고 [0, 1] 안이다.
② 다시 계산 (선택): split마다 `default_rng(0)`으로 고른 국면의 q를 이 장비에서 다시 계산해 캐시와 비교한다.
   최대 |Δq|가 기준(기본 1e-3) 이하여야 한다.

    python -m gpu_train.check_cache --player tal --base base_model/270M.pt
    python -m gpu_train.check_cache --player tal --base base_model/270M.pt --recompute 100 --device cpu

종료 코드: 0 통과, 1 실패. 마지막 줄에 결과를 JSON 한 줄로 출력한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.cache_base_q import compute_base_q
from src.data.dataset import SPLITS, cache_path, player_dir, read_records
from src.model.transformer import encode_position
from src.players import PLAYERS

TOLERANCE = 1e-3


def check_structure(data_dir: Path, base: str | Path) -> list[str]:
    """① 구조 검사. 문제 목록을 돌려준다 (비면 통과)."""
    path = cache_path(data_dir, base)
    if not path.exists():
        return [f"캐시가 없다: {path}"]
    problems = []
    try:
        cache = dict(np.load(path))
    except Exception as error:  # 저장 중에 끊긴 파일 등
        return [f"캐시를 읽을 수 없다: {path} ({error})"]
    for split in SPLITS:
        keys = [f"{split}_offsets", f"{split}_actions", f"{split}_q"]
        missing = [key for key in keys if key not in cache]
        if missing:
            problems.append(f"{split}: 키 없음 {missing}")
            continue
        offsets, actions, q = (cache[key] for key in keys)
        fens = [record["fen"] for record in read_records(data_dir, split)]
        if len(offsets) != len(fens) + 1:
            problems.append(f"{split}: 국면 수 {len(offsets) - 1} != 데이터 {len(fens)}")
            continue
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0) or offsets[-1] != len(actions) or len(q) != len(actions):
            problems.append(f"{split}: offsets·길이가 맞지 않는다")
            continue
        if not np.all(np.isfinite(q)) or q.min() < 0 or q.max() > 1:
            problems.append(f"{split}: q가 유한하지 않거나 [0, 1] 밖이다")
        bad = 0
        for i, fen in enumerate(fens):
            tokens, _ = encode_position(fen)
            span = actions[offsets[i]:offsets[i + 1]]
            if len(span) != len(tokens) or not np.array_equal(span.astype(np.int64), tokens[:, -2]):
                bad += 1
        if bad:
            problems.append(f"{split}: 합법수 수·순서가 다른 국면 {bad}개")
    return problems


def recompute(data_dir: Path, base: str | Path, count: int, device: str, seed: int = 0) -> dict:
    """② split마다 국면 count개를 다시 계산한다. {split: 최대 |Δq|}."""
    from src.model.transformer import load_base

    model = load_base(base, device).eval()
    with np.load(cache_path(data_dir, base)) as cache:
        stored = {key: cache[key] for key in cache.files}
    result = {}
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        fens = [record["fen"] for record in read_records(data_dir, split)]
        picks = np.sort(rng.choice(len(fens), size=min(count, len(fens)), replace=False))
        offsets, _, q = compute_base_q(model, [fens[i] for i in picks])
        cached_offsets, cached_q = stored[f"{split}_offsets"], stored[f"{split}_q"]
        worst = 0.0
        for j, i in enumerate(picks):
            mine = q[offsets[j]:offsets[j + 1]]
            theirs = cached_q[cached_offsets[i]:cached_offsets[i + 1]]
            worst = max(worst, float(np.abs(mine.astype(np.float64) - theirs).max()))
        result[split] = {"positions": int(len(picks)), "max_abs_diff": worst}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="베이스 q 캐시 검증")
    parser.add_argument("--player", required=True, choices=sorted(PLAYERS))
    parser.add_argument("--base", required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="기본값: data/processed/{player}")
    parser.add_argument("--recompute", type=int, default=0, help="split마다 다시 계산할 국면 수 (0이면 안 한다)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or player_dir(args.player)
    problems = check_structure(data_dir, args.base)
    for problem in problems:
        print(f"구조 문제: {problem}")
    summary = {"player": args.player, "cache": str(cache_path(data_dir, args.base)), "structure_ok": not problems}
    if not problems and args.recompute:
        summary["recompute"] = recompute(data_dir, args.base, args.recompute, args.device)
        summary["recompute_ok"] = all(r["max_abs_diff"] <= args.tolerance for r in summary["recompute"].values())
    ok = summary["structure_ok"] and summary.get("recompute_ok", True)
    print(("통과" if ok else "실패") + ": " + json.dumps(summary, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
