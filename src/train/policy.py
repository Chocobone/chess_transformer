"""합법수별 q(s, a)로부터 정책 π(a|s) = softmax(q / T)를 만든다.

한 배치의 모든 합법수를 한 줄로 펴서(M개) 순전파 1회로 처리하고, owner로 국면별로 다시 묶는다.
국면마다 합법수 개수가 다르므로 묶은 결과는 [B, A_max]로 패딩하고 mask로 실제 수를 표시한다.
수의 순서는 국면 안에서 행동 번호 순이다 (`ordered_legal_moves`).
"""

import numpy as np
import torch

from src.model.transformer import encode_position


def legal_action_tokens(fens: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """국면 B개 → 모델 입력 tokens[M, 79], 각 줄이 속한 국면 번호 owner[M], 행동 번호 action_ids[M]."""
    tokens, owner = [], []
    for index, fen in enumerate(fens):
        position_tokens, _ = encode_position(fen)
        if len(position_tokens) == 0:
            raise ValueError(f"합법수가 없는 국면이다: {fen}")
        tokens.append(position_tokens)
        owner.append(np.full(len(position_tokens), index, dtype=np.int64))
    tokens = torch.from_numpy(np.concatenate(tokens))
    return tokens, torch.from_numpy(np.concatenate(owner)), tokens[:, -2].clone()


def pad_by_owner(values: torch.Tensor, owner: torch.Tensor, fill: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """한 줄로 편 values[M]을 국면별로 묶어 [B, A_max]로 만든다. mask[B, A_max]는 실제 수의 위치다.

    owner는 0부터 시작해 오름차순으로 이어져 있어야 한다 (`legal_action_tokens`의 출력 형태).
    """
    counts = torch.bincount(owner)
    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(len(owner), device=owner.device) - starts[owner]
    padded = values.new_full((len(counts), int(counts.max())), fill)
    padded[owner, slot] = values
    mask = torch.zeros_like(padded, dtype=torch.bool)
    mask[owner, slot] = True
    return padded, mask


def policy_from_q(q: torch.Tensor, owner: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """q[M] → log_pi[B, A_max], mask[B, A_max]. 패딩 위치의 log_pi는 -inf(확률 0)다."""
    padded, mask = pad_by_owner(q.float() / temperature, owner, fill=float("-inf"))
    return torch.log_softmax(padded, dim=-1), mask


def target_index(action_ids: torch.Tensor, owner: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    """국면별 목표 수(행동 번호 target_actions[B])가 그 국면의 합법수 중 몇 번째인지 [B]."""
    padded, _ = pad_by_owner(action_ids, owner, fill=-1)
    hit = padded == target_actions[:, None]
    if not bool(hit.any(dim=1).all()):
        raise ValueError("목표 수가 합법수 목록에 없는 국면이 있다")
    return hit.int().argmax(dim=1)
