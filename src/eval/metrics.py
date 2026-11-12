"""덤프에서 국면별 값과 지표를 계산한다. numpy만 쓴다 (torch를 불러오지 않는다).

덤프의 합법수 배열은 국면마다 [offsets[i], offsets[i + 1]) 구간이다 (`src/eval/dump.py`).
국면별 값의 정의는 학습 평가 `src.train.train.evaluate`와 같다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval import stats

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


# ── 기준 문서의 지표 ───────────────────────────────────────
# 입력은 덤프 한 벌을 읽은 `Data`다. 기사 순서는 PLAYERS(탈, 카르포프, 스미슬로프)이고,
# 모델별 국면 값은 `per_position`의 결과를 쓴다. 구간은 src/eval/stats.py로 낸다.

PLY_BINS = ((0, 10), (10, 20), (20, 40), (40, None))
TAU = 0.02


@dataclass
class Data:
    """덤프 한 벌 (src/eval/dump.py의 positions.npz, base.npz, {모델 ID}.npz)."""

    player: np.ndarray  # [N] 기사 번호
    game: np.ndarray  # [N] 기사 안 game_id
    ply: np.ndarray  # [N]
    offsets: np.ndarray  # [N + 1]
    target: np.ndarray  # [N]
    regret: np.ndarray  # [M]
    new: np.ndarray  # [N]
    log_pi: dict[str, np.ndarray]  # 모델 ID → [M]
    tau: float = TAU
    _values: dict = field(default_factory=dict)

    def values(self, model: str) -> dict[str, np.ndarray]:
        if model not in self._values:
            self._values[model] = per_position(self.log_pi[model], self.offsets, self.target, self.regret, self.tau)
        return self._values[model]

    def target_log_pi(self, model: str) -> np.ndarray:
        return self.log_pi[model][self.offsets[:-1] + self.target].astype(np.float64)

    def games(self, select: np.ndarray) -> tuple[np.ndarray, int]:
        """선택한 국면의 대국 번호(0..G-1, (기사, game_id) 정렬 순)와 G."""
        return stats.group_index(np.stack([self.player[select], self.game[select]], axis=1))


def ply_distribution(ply: np.ndarray) -> dict[str, float]:
    total = max(len(ply), 1)
    out = {}
    for low, high in PLY_BINS:
        hit = (ply >= low) if high is None else (ply >= low) & (ply < high)
        out[f"{low}~" if high is None else f"{low}~{high - 1}"] = float(hit.sum() / total)
    return out


def js_per_position(log_p: np.ndarray, log_q: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """국면별 Jensen-Shannon 발산, 밑 2 (0~1). B2 (4.3절). 0·log 0 = 0, float64로 계산한다."""
    lp, lq = np.asarray(log_p, dtype=np.float64), np.asarray(log_q, dtype=np.float64)
    lm = np.logaddexp(lp, lq) - np.log(2.0)
    p, q = np.exp(lp), np.exp(lq)
    with np.errstate(invalid="ignore"):  # p = 0인 항(0·(−∞))은 아래에서 0으로 둔다
        terms = 0.5 * (np.where(p > 0, p * (lp - lm), 0.0) + np.where(q > 0, q * (lq - lm), 0.0))
    return np.add.reduceat(terms, offsets[:-1]) / np.log(2.0)


def a_metrics(data: Data, player: int, adapter: str, n_boot: int = stats.N_BOOT) -> dict:
    """A1·A2·A3 (3.3절). 기사 p의 val에서 어댑터와 베이스를 쌍대로 비교한다 (같은 뽑기 행렬)."""
    select = data.player == player
    groups, n = data.games(select)
    weights = stats.resample_weights(n, n_boot, np.random.default_rng(stats.SEED))
    a, b = data.values(adapter), data.values("base")
    ply, new = data.ply[select], data.new[select]
    ones = np.ones(int(select.sum()))

    def diff(name: str, subset: np.ndarray) -> stats.Interval:
        delta = (a[name] - b[name])[select] * subset
        return stats.ratio_ci(stats.group_sums(delta, groups, n), stats.group_sums(subset, groups, n), weights)

    late = (ply >= 20).astype(float)
    out = {
        "positions": int(select.sum()), "games": n,
        "A1": {part: {name: diff(name, mask).as_dict() for name in ("top1", "imitation", "top3")}
               for part, mask in (("all", ones), ("ply20", late))},
        "A2": {"top1": diff("top1", new.astype(float)).as_dict(), "positions": int(new.sum()),
               "games": int(len(np.unique(groups[new])))},
    }
    hit_a, hit_b = a["top1"][select], b["top1"][select]
    disagree = (a["choice"] != b["choice"])[select]
    num = stats.group_sums((hit_a - hit_b) * disagree, groups, n)
    iv = stats.ratio_ci(num, stats.group_sums(disagree.astype(float), groups, n), weights)
    only_a = int(np.sum(disagree & (hit_a > hit_b)))
    only_b = int(np.sum(disagree & (hit_b > hit_a)))
    out["A3"] = {"r": iv.as_dict(), "disagree": int(disagree.sum()), "share": float(disagree.mean()),
                 "adapter_only": only_a, "base_only": only_b, "mcnemar_p": stats.binom_two_sided(only_a, only_b),
                 "ply_disagree": ply_distribution(ply[disagree]), "ply_all": ply_distribution(ply)}
    return out


def c3_metrics(data: Data, player: int, model: str, n_boot: int = stats.N_BOOT) -> dict:
    """C3 (5.4절): 기사 p val에서 regret_top1 평균과 구간. regret·band_violation·strength는 보고."""
    select = data.player == player
    groups, n = data.games(select)
    weights = stats.resample_weights(n, n_boot, np.random.default_rng(stats.SEED))
    v = data.values(model)
    den = stats.group_sums(np.ones(int(select.sum())), groups, n)
    iv = stats.ratio_ci(stats.group_sums(v["regret_top1"][select], groups, n), den, weights)
    out = {"regret_top1": iv.as_dict()}
    for name in ("regret", "band_violation", "strength"):
        out[name] = float(v[name][select].mean())
    return out


def b2_b3(data: Data, main: list[str], seed0: list[str], n_boot: int = stats.N_BOOT) -> dict:
    """B2 정책 거리와 B3 잡음 바닥 (4.3절). S_val 전체, 기사별 층화 대국 부트스트랩 (해석 1).

    main[i], seed0[i]는 기사 i의 seed 1 어댑터와 seed 0 어댑터다.
    n_p = d(seed0_p, main_p), M(p, q) = d(p, q) − max(n_p, n_q)를 반복마다 계산한다.
    """
    everything = np.ones(len(data.target), dtype=bool)
    groups, n = data.games(everything)
    strata = [len(np.unique(data.game[data.player == i])) for i in range(len(main))]
    weights = stats.stratified_weights(strata, n_boot)
    models = list(main) + list(seed0)
    late = (data.ply >= 20).astype(float)
    den_all = stats.group_sums(np.ones(len(data.target)), groups, n)
    den_late = stats.group_sums(late, groups, n)

    pairs, boots = {}, {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            js = js_per_position(data.log_pi[m1], data.log_pi[m2], data.offsets)
            delta = (data.values(m1)["choice"] != data.values(m2)["choice"]).astype(float)
            value, boot = stats.boot_ratio(stats.group_sums(js, groups, n), den_all, weights)
            value_late, boot_late = stats.boot_ratio(stats.group_sums(js * late, groups, n), den_late, weights)
            boots[(m1, m2)] = boots[(m2, m1)] = (value, boot, value_late, boot_late)
            pairs[f"{m1}-{m2}"] = {
                "d": stats.interval(value, boot).as_dict(),
                "d_ply20": stats.interval(value_late, boot_late).as_dict(),
                "delta": float(delta.mean()), "delta_ply20": float(delta[late > 0].mean()),
            }

    noise = {main[i]: boots[(seed0[i], main[i])] for i in range(len(main))}
    margins = {}
    for i, p in enumerate(main):
        for q in main[i + 1:]:
            d, n_p, n_q = boots[(p, q)], noise[p], noise[q]
            margin = {}
            for part, k in (("all", 0), ("ply20", 2)):
                value = d[k] - max(n_p[k], n_q[k])
                boot = d[k + 1] - np.maximum(n_p[k + 1], n_q[k + 1])
                margin[part] = stats.interval(value, boot).as_dict()
            margin["verdict"] = stats.verdict(stats.Interval(**margin["all"]), "above")
            margins[f"{p}-{q}"] = margin
    verdicts = [m["verdict"] for m in margins.values()]
    b3 = "통과" if all(v == "통과" for v in verdicts) else "실패" if "실패" in verdicts else "판정 보류"
    noise_out = {p: {"pair": f"{seed0[i]}-{p}", **stats.interval(noise[p][0], noise[p][1]).as_dict()}
                 for i, p in enumerate(main)}
    return {"pairs": pairs, "noise": noise_out, "margins": margins, "B3": b3}


def b10_metrics(data: Data, adapters: list[str], n_boot: int = stats.N_BOOT, min_ply: int = 20,
                bundles: tuple[int, ...] = (2, 5, 10), bundle_repeats: int = 1000) -> dict:
    """B10 우도 혼동 행렬 (4.3절). adapters[i]는 기사 i의 어댑터다.

    - 대국 점수 L_m(g) = Σ_{s∈g, ply ≥ min_ply} log π_m(a_p(s)|s). 기사 p의 val 대국을 argmax_m L_m(g)로 분류한다
    - 동률(최댓값이 둘 이상)은 오답으로 센다 (해석 2)
    - ply ≥ min_ply 국면이 없는 대국은 점수가 없어 뺀다. 뺀 수를 보고한다
    - 묶음 k > 1: 대국을 k개씩 무작위로 나누고(남는 대국은 버림) 점수를 더해 분류한다. 1,000번 평균 (해석 3)
    - 3지선다 통과: 세 기사 재현율 구간 하한 > 1/3, 실패: 어느 기사든 상한 < 1/3
    - 쌍별: 두 어댑터만으로 2지선다, 두 기사 모두 하한 > 1/2이면 그 쌍 통과. 기준 문서에 쌍별 실패 조건은 없다
    """
    n_players = len(adapters)
    late = data.ply >= min_ply
    per_player = []
    for p in range(n_players):
        select = (data.player == p) & late
        groups, n = data.games(select)
        scores = np.stack([stats.group_sums(data.target_log_pi(m)[select], groups, n) for m in adapters], axis=1)
        all_games = len(np.unique(data.game[data.player == p]))
        per_player.append((scores, n, all_games - n))

    out = {"recall": {}, "matrix": [], "ties": [], "excluded_games": [], "pairwise": {}, "bundles": {}}
    for p, (scores, n, excluded) in enumerate(per_player):
        winners = scores == scores.max(axis=1, keepdims=True)
        tied = winners.sum(axis=1) > 1
        correct = (winners[:, p] & ~tied).astype(float)
        weights = stats.resample_weights(n, n_boot, np.random.default_rng(stats.SEED))
        iv = stats.ratio_ci(correct, np.ones(n), weights)
        out["recall"][adapters[p]] = {**iv.as_dict(), "verdict": stats.verdict(iv, "above", 1 / 3), "games": n}
        out["matrix"].append([float(np.sum(winners[:, q] & ~tied) / n) for q in range(n_players)])
        out["ties"].append(int(tied.sum()))
        out["excluded_games"].append(int(excluded))
        for k in bundles:
            rng = np.random.default_rng(stats.SEED)
            usable = (n // k) * k
            hits = []
            for _ in range(bundle_repeats):
                order = rng.permutation(n)[:usable]
                summed = scores[order].reshape(-1, k, n_players).sum(axis=1)
                win = summed == summed.max(axis=1, keepdims=True)
                hits.append(np.mean(win[:, p] & (win.sum(axis=1) == 1)) if usable else np.nan)
            out["bundles"].setdefault(str(k), {})[adapters[p]] = float(np.mean(hits))
    out["column_totals"] = [int(round(sum(per_player[p][1] * out["matrix"][p][q] for p in range(n_players))))
                            for q in range(n_players)]

    for p in range(n_players):
        for q in range(p + 1, n_players):
            pair = {}
            for me, other in ((p, q), (q, p)):
                scores, n, _ = per_player[me]
                correct = (scores[:, me] > scores[:, other]).astype(float)
                weights = stats.resample_weights(n, n_boot, np.random.default_rng(stats.SEED))
                iv = stats.ratio_ci(correct, np.ones(n), weights)
                pair[adapters[me]] = {**iv.as_dict(), "verdict": stats.verdict(iv, "above", 1 / 2)}
            both = all(pair[adapters[i]]["verdict"] == "통과" for i in (p, q))
            pair["verdict"] = "통과" if both else "판정 보류"
            out["pairwise"][f"{adapters[p]}-{adapters[q]}"] = pair

    three = [r["verdict"] for r in out["recall"].values()]
    out["three_way"] = "통과" if all(v == "통과" for v in three) else "실패" if "실패" in three else "판정 보류"
    pairwise_pass = all(pair["verdict"] == "통과" for pair in out["pairwise"].values())
    if out["three_way"] == "실패":
        out["B10"] = "실패"
    else:
        out["B10"] = "통과" if out["three_way"] == "통과" and pairwise_pass else "판정 보류"
    return out


def b1_metrics(data: Data, adapters: list[str]) -> dict:
    """B1 교차 평가 (4.3절, 보고). 3×3 imitation·top1과 기사 특이 몫 (Task #6 방식, 해석 7).

    행은 val의 기사, 열은 어댑터다. 기사 특이 몫 = 1 − (다른 두 어댑터의 imitation 개선 평균) / (자기 어댑터의 개선).
    개선 = 베이스 imitation − 어댑터 imitation.
    """
    base = data.values("base")
    out = {}
    for part, min_ply in (("all", 0), ("ply20", 20)):
        table = {"imitation": [], "top1": [], "base_imitation": [], "base_top1": [], "specific_share": []}
        for p in range(len(adapters)):
            select = (data.player == p) & (data.ply >= min_ply)
            table["base_imitation"].append(float(base["imitation"][select].mean()))
            table["base_top1"].append(float(base["top1"][select].mean()))
            imit = [float(data.values(m)["imitation"][select].mean()) for m in adapters]
            table["imitation"].append(imit)
            table["top1"].append([float(data.values(m)["top1"][select].mean()) for m in adapters])
            gains = [table["base_imitation"][-1] - x for x in imit]
            others = [g for q, g in enumerate(gains) if q != p]
            table["specific_share"].append(float(1 - np.mean(others) / gains[p]))
        out[part] = table
    return out


def b11_metrics(data: Data, adapters: list[str]) -> dict:
    """B11 (4.6절, 보조): 세 어댑터의 1순위가 모두 같지는 않은 국면에서 기사 p val의 어댑터 m 적중률 3×3."""
    choices = np.stack([data.values(m)["choice"] for m in adapters], axis=1)
    split = ~(choices == choices[:, :1]).all(axis=1)
    rates, counts = [], []
    for p in range(len(adapters)):
        select = split & (data.player == p)
        counts.append(int(select.sum()))
        rates.append([float(data.values(m)["top1"][select].mean()) if select.any() else float("nan")
                      for m in adapters])
    return {"rates": rates, "positions": counts}
