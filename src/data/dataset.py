"""학습용 데이터셋: 전처리 JSONL과 베이스 q 캐시를 읽어 배치로 묶는다.

q 캐시(`base_q_{베이스 이름}.npz`)는 `scripts/cache_base_q.py`가 만든다. split마다 세 배열이 들어 있다.

    {split}_offsets[N + 1]  국면 i의 합법수는 [offsets[i], offsets[i + 1]) 구간
    {split}_actions[M]      합법수의 행동 번호 (국면 안에서 행동 번호 순)
    {split}_q[M]            베이스의 q(s, a)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.model.actions import MOVE_TO_ACTION
from src.train.policy import legal_action_tokens, target_index

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val")


def player_dir(player: str) -> Path:
    return ROOT / "data" / "processed" / player


def cache_path(data_dir: Path, base: str | Path) -> Path:
    """베이스 체크포인트 파일명(확장자 제외)으로 캐시를 구분한다. 예: base_model/270M.pt → base_q_270M.npz."""
    return Path(data_dir) / f"base_q_{Path(base).stem}.npz"


def read_records(data_dir: Path, split: str) -> list[dict]:
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class Batch:
    """국면 B개의 합법수 M개를 한 줄로 편 배치. owner는 0부터 오름차순으로 이어진다."""

    tokens: torch.Tensor  # [M, 79] 모델 입력
    owner: torch.Tensor  # [M] 각 줄이 속한 국면 번호
    q_base: torch.Tensor  # [M] 베이스의 q(s, a)
    target_idx: torch.Tensor  # [B] 기사가 둔 수가 그 국면의 합법수 중 몇 번째인지

    @property
    def num_positions(self) -> int:
        return len(self.target_idx)

    @property
    def num_moves(self) -> int:
        return len(self.owner)

    def to(self, device: str | torch.device) -> "Batch":
        return Batch(*(getattr(self, name).to(device, non_blocking=True) for name in self.__dataclass_fields__))


class PositionDataset(Dataset):
    """한 항목 = 한 국면: (FEN, 기사가 둔 수의 행동 번호, 캐시의 합법수 행동 번호, 캐시의 베이스 q)."""

    def __init__(self, data_dir: Path, split: str, base: str | Path):
        records = read_records(data_dir, split)
        self.fens = [record["fen"] for record in records]
        self.targets = np.array([MOVE_TO_ACTION[record["move"]] for record in records], dtype=np.int64)

        path = cache_path(data_dir, base)
        if not path.exists():
            raise FileNotFoundError(f"베이스 q 캐시가 없다: {path} (scripts/cache_base_q.py를 먼저 실행한다)")
        with np.load(path) as cache:
            self.offsets = cache[f"{split}_offsets"]
            self.actions = cache[f"{split}_actions"].astype(np.int64)
            self.q = cache[f"{split}_q"].astype(np.float32)
        if len(self.offsets) != len(self.fens) + 1:
            raise ValueError(
                f"캐시의 국면 수({len(self.offsets) - 1})가 {split}.jsonl({len(self.fens)})과 다르다. "
                "전처리를 다시 했다면 캐시도 다시 만든다"
            )

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, index: int) -> tuple[str, int, np.ndarray, np.ndarray]:
        span = slice(self.offsets[index], self.offsets[index + 1])
        return self.fens[index], int(self.targets[index]), self.actions[span], self.q[span]


def collate_fn(items: list[tuple[str, int, np.ndarray, np.ndarray]]) -> Batch:
    fens, targets, actions, q = zip(*items)
    tokens, owner, action_ids = legal_action_tokens(list(fens))
    cached_actions = torch.from_numpy(np.concatenate(actions))
    # 캐시의 q는 수 순서가 같아야만 의미가 있다
    if not torch.equal(action_ids, cached_actions):
        raise ValueError("캐시의 합법수 순서가 국면과 맞지 않는다. 캐시를 다시 만든다")
    return Batch(
        tokens=tokens,
        owner=owner,
        q_base=torch.from_numpy(np.concatenate(q)),
        target_idx=target_index(action_ids, owner, torch.tensor(targets)),
    )


def split_by_moves(batch: Batch, max_moves: int) -> list[Batch]:
    """배치를 합법수 합계가 max_moves 이하인 묶음으로 나눈다 (기울기 누적용).

    한 국면의 합법수는 쪼개지 않는다. 정책이 그 국면의 모든 합법수에 대한 softmax이기 때문이다.
    합법수가 한도보다 많은 국면은 혼자 한 묶음이 된다.
    """
    counts = torch.bincount(batch.owner).tolist()
    starts = np.concatenate([[0], np.cumsum(counts)])

    boundaries = [0]
    moves = 0
    for position, count in enumerate(counts):
        if moves and moves + count > max_moves:
            boundaries.append(position)
            moves = 0
        moves += count
    boundaries.append(len(counts))

    chunks = []
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        span = slice(int(starts[first]), int(starts[last]))
        chunks.append(Batch(
            tokens=batch.tokens[span],
            owner=batch.owner[span] - first,
            q_base=batch.q_base[span],
            target_idx=batch.target_idx[first:last],
        ))
    return chunks
