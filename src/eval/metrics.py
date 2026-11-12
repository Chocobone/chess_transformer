"""덤프에서 국면별 값과 지표를 계산한다. numpy만 쓴다 (torch를 불러오지 않는다).

덤프의 합법수 배열은 국면마다 [offsets[i], offsets[i + 1]) 구간이다 (`src/eval/dump.py`).
국면별 값의 정의는 학습 평가 `src.train.train.evaluate`와 같다.
"""

from __future__ import annotations

import numpy as np

METRICS = ("imitation", "top1", "top3", "regret", "regret_top1", "band_violation", "strength")
EXACT = ("top1", "top3")
TOLERANCE = 5e-5


def segment_argmax(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """국면마다 값이 가장 큰 합법수의 국면 안 번호. 동률이면 앞 번호(행동 번호가 작은 수)다."""
    counts = np.diff(offsets)
    owner = np.repeat(np.arange(len(counts)), counts)
    best = np.maximum.reduceat(values, offsets[:-1])
    is_best = values == best[owner]
    slot = np.arange(len(values)) - offsets[:-1][owner]
    first = np.where(is_best, slot, np.iinfo(np.int64).max)
    return np.minimum.reduceat(first, offsets[:-1])


def count_ties(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """국면마다 최댓값을 가진 합법수의 개수 (1이면 동률 없음)."""
    counts = np.diff(offsets)
    owner = np.repeat(np.arange(len(counts)), counts)
    best = np.maximum.reduceat(values, offsets[:-1])
    return np.add.reduceat((values == best[owner]).astype(np.int64), offsets[:-1])


def per_position(log_pi: np.ndarray, offsets: np.ndarray, target: np.ndarray, regret: np.ndarray,
                 tau: float = 0.02) -> dict[str, np.ndarray]:
    """국면별 값. 학습 평가(`evaluate`)의 각 항을 국면 단위로 남긴 것이다.

    imitation       −log π(a_p|s)
    top1, top3      기사 수가 정책의 상위 1/3개 안에 드는지 (동률은 앞 번호 우선)
    regret          Σ π·regret, regret = max q_base − q_base
    regret_top1     1순위 수의 regret
    band_violation  regret > τ인 수에 실린 확률의 합
    strength        Σ π·max(regret − τ, 0)
    choice          1순위 수의 국면 안 번호
    """
    log_pi = np.asarray(log_pi, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    regret = np.asarray(regret, dtype=np.float32)
    starts = offsets[:-1]
    counts = np.diff(offsets)
    owner = np.repeat(np.arange(len(counts)), counts)

    pi = np.exp(log_pi.astype(np.float64))
    target_log_pi = log_pi[starts + target].astype(np.float64)
    choice = segment_argmax(log_pi, offsets)
    # 기사 수보다 확률이 큰 수의 개수(동률은 앞 번호만 앞선다)로 순위를 낸다
    slot = np.arange(len(log_pi)) - starts[owner]
    ahead = (log_pi > log_pi[starts + target][owner]) | (
        (log_pi == log_pi[starts + target][owner]) & (slot < target[owner]))
    rank = np.add.reduceat(ahead.astype(np.int64), starts)
    reg = regret.astype(np.float64)
    return {
        "imitation": -target_log_pi,
        "top1": (rank < 1).astype(np.float64),
        "top3": (rank < 3).astype(np.float64),
        "regret": np.add.reduceat(pi * reg, starts),
        "regret_top1": reg[starts + choice],
        "band_violation": np.add.reduceat(pi * (reg > tau), starts),
        "strength": np.add.reduceat(pi * np.clip(reg - tau, 0, None), starts),
        "choice": choice,
    }


def summarize(values: dict[str, np.ndarray], select: np.ndarray | None = None) -> dict[str, float]:
    """국면 평균 (학습 평가의 7개 지표)."""
    return {name: float(np.mean(values[name] if select is None else values[name][select])) for name in METRICS}


def compare(metrics: dict[str, float], reference: dict, tolerance: float = TOLERANCE) -> dict:
    """지표별 차이(다시 잰 값 − 기록값)와 판정. top1·top3는 정확히 같아야 하고, 나머지는 |차이| < tolerance.

    규칙은 `gpu_train/eval_adapter.py`와 같다.
    """
    items = {}
    for name in METRICS:
        diff = metrics[name] - reference[name]
        passed = diff == 0 if name in EXACT else abs(diff) < tolerance
        items[name] = {"value": metrics[name], "reference": reference[name], "diff": diff, "pass": bool(passed)}
    return {"pass": all(item["pass"] for item in items.values()), "items": items}
