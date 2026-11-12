"""대국 단위 부트스트랩 (기준 문서 2.2절). numpy만 쓴다.

- 10,000회, seed 0, 백분위 95% 구간(2.5·97.5 백분위, `np.percentile` 기본 선형 보간)
- 국면 평균 지표는 "뽑힌 대국들의 국면 합 / 국면 수"로 계산한다 (비율 추정). 원표본 값은 전체 국면 평균과 같다
- 쌍대 비교: 같은 모집단의 값들은 같은 뽑기 행렬을 쓰고, 차이는 반복마다 계산한 뒤 백분위를 낸다
- 뽑기 행렬은 모집단마다 `default_rng(seed)`로 새로 만든다. 계산 순서와 관계없이 같은 결과가 나온다
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

N_BOOT = 10_000
SEED = 0


@dataclass(frozen=True)
class Interval:
    value: float  # 원표본 값
    low: float  # 2.5 백분위
    high: float  # 97.5 백분위

    def as_dict(self) -> dict:
        return asdict(self)


def group_index(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """국면별 대국 키 → (국면별 대국 번호 0..G-1, G). 대국 번호는 키의 정렬 순서다."""
    unique, inverse = np.unique(keys, return_inverse=True, axis=0)
    return inverse.reshape(-1), len(unique)


def resample_weights(n_groups: int, n_boot: int = N_BOOT, rng: np.random.Generator | None = None) -> np.ndarray:
    """[n_boot, n_groups]: 반복마다 대국을 복원 추출했을 때 대국별로 뽑힌 횟수."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    weights = np.zeros((n_boot, n_groups))
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1.0)
    return weights


def stratified_weights(strata: list[int], n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """층마다 따로 복원 추출해 이어 붙인 뽑기 행렬. strata는 층별 대국 수(층 순서대로 대국 번호가 이어진다).

    한 생성기(`default_rng(seed)`)로 층 순서대로 뽑는다. 반복마다 층의 대국 수가 그대로 유지된다.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([resample_weights(n, n_boot, rng) for n in strata], axis=1)


def group_sums(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """국면별 값 → 대국별 합."""
    return np.bincount(groups, weights=np.asarray(values, dtype=np.float64), minlength=n_groups)


def boot_ratio(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """대국별 합 num[G], den[G] → (원표본 Σnum/Σden, 반복별 값[n_boot])."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(num.sum() / den.sum()), (weights @ num) / (weights @ den)


def interval(value: float, boot: np.ndarray) -> Interval:
    """백분위 95% 구간. 분모가 0이 된 반복(예: 새 국면이 있는 대국을 하나도 뽑지 않음)은 값이 없어 뺀다."""
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return Interval(float(value), float(low), float(high))


def ratio_ci(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> Interval:
    return interval(*boot_ratio(num, den, weights))


def binom_two_sided(b: int, c: int) -> float:
    """McNemar 정확 검정: b + c번 중 작은 쪽이 min(b, c) 이하일 확률 × 2 (최대 1). 참고값이다."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    top = max(log_terms)
    tail = math.exp(top) * sum(math.exp(t - top) for t in log_terms)
    return min(1.0, 2 * tail)


def verdict(iv: Interval, passes: str, threshold: float = 0.0) -> str:
    """구간과 조건으로 통과·실패·판정 보류 (2.5절).

    passes="above": 하한 > threshold이면 통과, 상한 < threshold이면 실패
    passes="below": 상한 < threshold이면 통과, 하한 > threshold이면 실패
    """
    if passes == "above":
        return "통과" if iv.low > threshold else "실패" if iv.high < threshold else "판정 보류"
    if passes == "below":
        return "통과" if iv.high < threshold else "실패" if iv.low > threshold else "판정 보류"
    raise ValueError(passes)
