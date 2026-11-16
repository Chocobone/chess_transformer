"""C1·C2·C4 집계와 판정, C7 반복 비율, C2 노드 수 보정 선택 (평가 기준 문서 5.4절, Task #13).

numpy만 쓴다 (torch, python-chess 없이 돈다). 입력은 대국 실행기(`src/match/play.py`)의 `index.jsonl`·`config.json`과
C4 탐색(`src/match/quality.py`)의 `{모델}.npz`, Task #12 덤프의 `positions.npz`다.

- C1·C2 (해석 4·5): 짝(오프닝 하나, 2국)의 점수 합 0~2를 단위로 복원 추출한다(10,000회, seed 0). 점수율 s를 Elo로
  바꾸고 s는 [0.001, 0.999]로 자른다. C2는 같은 오프닝을 뽑아 어댑터 짝과 베이스 짝에 같이 쓴다
- C4 (해석 9): S_val 전체, 대국 단위 쌍대 부트스트랩을 기사별 층화로 한다 (Task #12 B2와 같은 뽑기)
- 판정 한계: C1·C2는 −50 Elo(구간 하한 > −50 통과, 상한 < −50 실패), C4는 손실 차이 0.02(상한 < 0.02 통과)

    python -m src.eval.games judge --match runs/gpu/match --quality runs/gpu/quality --dump runs/gpu/dump --out runs/gpu/games
    python -m src.eval.games calibrate runs/gpu/match/cal_base_n100 runs/gpu/match/cal_base_n300 ...
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.eval import stats

ELO_LIMIT = -50.0
LOSS_LIMIT = 0.02
BLUNDER = 0.20
CLIP = 0.001
CAL_RANGE = (0.35, 0.65)
MAIN = ("R1", "R8", "R2")  # 탈, 카르포프, 스미슬로프 (기사 순서)
NAMES = {"R1": "탈", "R8": "카르포프", "R2": "스미슬로프"}


class GamesError(RuntimeError):
    pass


# ---- 기보 읽기 ----

def read_dir(path: Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    with open(path / "index.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return config, rows


def pair_scores(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """→ (오프닝 번호[P] 오름차순, 짝 점수[P] 0~2). 짝이 다 차지 않았거나 같은 대국이 두 번 있으면 거부한다."""
    games: dict[tuple[int, str], float] = {}
    for row in rows:
        key = (row["opening"], row["model_color"])
        if key in games:
            raise GamesError(f"같은 대국이 두 번 있다: {key}")
        games[key] = row["score"]
    openings = sorted({key[0] for key in games})
    missing = [(o, c) for o in openings for c in ("white", "black") if (o, c) not in games]
    if missing:
        raise GamesError(f"짝이 다 차지 않았다: {missing[:5]}")
    return np.array(openings), np.array([games[(o, "white")] + games[(o, "black")] for o in openings])


# ---- 점수율과 Elo ----

def elo(s):
    """점수율 → Elo. s는 [0.001, 0.999]로 자른다 (해석 4)."""
    s = np.clip(np.asarray(s, dtype=np.float64), CLIP, 1 - CLIP)
    return -400.0 * np.log10(1.0 / s - 1.0)


def boot_scores(pairs: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    return stats.boot_ratio(pairs, np.full(len(pairs), 2.0), weights)


def pair_weights(n: int, n_boot: int) -> np.ndarray:
    return stats.resample_weights(n, n_boot, np.random.default_rng(stats.SEED))


def score_summary(rows: list[dict], pairs: np.ndarray) -> dict:
    wins = sum(r["score"] == 1.0 for r in rows)
    draws = sum(r["score"] == 0.5 for r in rows)
    terminations: dict[str, int] = {}
    for r in rows:
        terminations[r["termination"]] = terminations.get(r["termination"], 0) + 1
    return {"games": len(rows), "pairs": int(len(pairs)), "wins": int(wins), "draws": int(draws),
            "losses": int(len(rows) - wins - draws), "terminations": terminations,
            "mean_ply": float(np.mean([r["ply"] for r in rows])) if rows else 0.0}


def c1(rows: list[dict], n_boot: int = stats.N_BOOT) -> dict:
    """C1: 어댑터 대 베이스. Elo 구간 하한 > −50 통과, 상한 < −50 실패."""
    _, pairs = pair_scores(rows)
    s, boot = boot_scores(pairs, pair_weights(len(pairs), n_boot))
    iv = stats.interval(float(elo(s)), elo(boot))
    return {"score": stats.interval(s, boot).as_dict(), "elo": iv.as_dict(), "clipped": bool(s <= CLIP or s >= 1 - CLIP),
            "verdict": stats.verdict(iv, "above", ELO_LIMIT), **score_summary(rows, pairs)}


def c2(model_rows: list[dict], base_rows: list[dict], n_boot: int = stats.N_BOOT) -> dict:
    """C2: (어댑터 대 SF Elo) − (베이스 대 SF Elo). 같은 오프닝 짝끼리 쌍대 부트스트랩 (해석 5)."""
    o_m, p_m = pair_scores(model_rows)
    o_b, p_b = pair_scores(base_rows)
    if not np.array_equal(o_m, o_b):
        raise GamesError("C2 어댑터와 베이스의 오프닝이 다르다")
    weights = pair_weights(len(p_m), n_boot)
    s_m, boot_m = boot_scores(p_m, weights)
    s_b, boot_b = boot_scores(p_b, weights)
    iv = stats.interval(float(elo(s_m) - elo(s_b)), elo(boot_m) - elo(boot_b))
    return {"diff": iv.as_dict(), "verdict": stats.verdict(iv, "above", ELO_LIMIT),
            "model": {"score": stats.interval(s_m, boot_m).as_dict(), "elo": stats.interval(float(elo(s_m)), elo(boot_m)).as_dict(),
                      **score_summary(model_rows, p_m)},
            "base": {"score": stats.interval(s_b, boot_b).as_dict(), "elo": stats.interval(float(elo(s_b)), elo(boot_b)).as_dict(),
                     **score_summary(base_rows, p_b)},
            "clipped": bool(min(s_m, s_b) <= CLIP or max(s_m, s_b) >= 1 - CLIP)}


def c7_repetition(rows: list[dict]) -> float:
    """C7 (2): 3회(또는 5회) 반복으로 끝난 대국의 비율."""
    if not rows:
        return 0.0
    return sum(r["termination"] in ("threefold_repetition", "fivefold_repetition") for r in rows) / len(rows)


# ---- C4 ----

def c4(loss_model: np.ndarray, loss_base: np.ndarray, player: np.ndarray, game: np.ndarray, ply: np.ndarray,
       n_boot: int = stats.N_BOOT) -> dict:
    """C4: 1순위 수 평균 손실 차이 (어댑터 − 베이스). 상한 < 0.02 통과, 하한 > 0.02 실패 (해석 9)."""
    groups, n = stats.group_index(np.stack([player, game], axis=1))
    strata = [len(np.unique(game[player == p])) for p in np.unique(player)]
    weights = stats.stratified_weights(strata, n_boot)
    ones = np.ones(len(player))
    late = (ply >= 20).astype(float)
    delta = loss_model - loss_base
    blunder = (loss_model >= BLUNDER).astype(float) - (loss_base >= BLUNDER).astype(float)

    def ci(values, mask):
        return stats.ratio_ci(stats.group_sums(values * mask, groups, n), stats.group_sums(mask, groups, n), weights)

    iv = ci(delta, ones)
    per_player = {}
    for p in np.unique(player):
        mask = (player == p).astype(float)
        per_player[int(p)] = {"diff": ci(delta, mask).as_dict(), "model": float(loss_model[player == p].mean()),
                              "base": float(loss_base[player == p].mean())}
    return {"diff": iv.as_dict(), "verdict": stats.verdict(iv, "below", LOSS_LIMIT),
            "diff_ply20": ci(delta, late).as_dict(), "model": float(loss_model.mean()), "base": float(loss_base.mean()),
            "blunder_diff": ci(blunder, ones).as_dict(), "blunder_model": float((loss_model >= BLUNDER).mean()),
            "blunder_base": float((loss_base >= BLUNDER).mean()), "per_player": per_player}


# ---- C2 노드 수 보정 (해석 6) ----

def calibrate(dirs: list[Path]) -> dict:
    """베이스 대 SF 보정 대국 폴더들 → 노드 수별 점수율과 선택.

    - 선택: 점수율이 0.35~0.65에 드는 노드 수 중 0.5에 가장 가까운 것 (같으면 작은 노드 수)
    - 예비 단계 제안(next): 노드 수 순으로 점수율이 0.5를 지나는 이웃 두 후보와, 그 바깥 이웃 중 0.5에 더 가까운
      하나 → 본 보정 후보 3개. 0.5를 지나지 않으면 0.5 쪽 끝의 3개
    - 0.5까지의 거리는 소수 9자리로 반올림해 비교한다 (0.55와 0.45를 같은 거리로 본다)
    - 범위 안이 없으면 extra: 0.5를 지나는 이웃 두 노드 수의 기하 평균 (반올림)
    """
    rows = []
    for d in dirs:
        config, games = read_dir(d)
        if config["model"] != "base" or config["opponent"] != "sf":
            raise GamesError(f"{d}는 베이스 대 SF 대국이 아니다")
        _, pairs = pair_scores(games)
        rows.append({"dir": str(d), "nodes": int(config["nodes"]), "games": len(games),
                     "score": float(pairs.sum() / (2 * len(pairs)))})
    rows.sort(key=lambda r: r["nodes"])
    if len({r["nodes"] for r in rows}) != len(rows):
        raise GamesError("같은 노드 수가 두 번 있다")

    in_range = [r for r in rows if CAL_RANGE[0] <= r["score"] <= CAL_RANGE[1]]
    chosen = min(in_range, key=lambda r: (round(abs(r["score"] - 0.5), 9), r["nodes"]))["nodes"] if in_range else None

    cross = next((i for i in range(len(rows) - 1) if rows[i]["score"] >= 0.5 >= rows[i + 1]["score"]), None)
    if cross is None:
        nxt = [r["nodes"] for r in (rows[:3] if all(r["score"] < 0.5 for r in rows) else rows[-3:])]
        extra = None
    else:
        picks = [cross, cross + 1]
        outer = [i for i in (cross - 1, cross + 2) if 0 <= i < len(rows)]
        if outer:
            picks.append(min(outer, key=lambda i: (round(abs(rows[i]["score"] - 0.5), 9), rows[i]["nodes"])))
        nxt = sorted(rows[i]["nodes"] for i in picks)
        extra = int(round(math.sqrt(rows[cross]["nodes"] * rows[cross + 1]["nodes"])))
    return {"rows": rows, "chosen": chosen, "next": nxt, "extra": None if chosen is not None else extra}


# ---- 판정 묶음 ----

def judge(match: Path, quality: Path | None, dump: Path | None, main: list[str] = list(MAIN), base_dir: str = "C2_base",
          n_boot: int = stats.N_BOOT) -> dict:
    """match 아래 C1_{모델}, C2_{모델}, C2_base 폴더와 quality의 C4 결과로 판정한다."""
    match = Path(match)
    out: dict = {"models": main, "C1": {}, "C2": {}, "C4": {}, "C7": {}}
    _, base_rows = read_dir(match / base_dir)
    out["C7"]["base_vs_sf"] = c7_repetition(base_rows)
    for model in main:
        _, rows = read_dir(match / f"C1_{model}")
        out["C1"][model] = c1(rows, n_boot)
        out["C7"][f"{model}_vs_base"] = c7_repetition(rows)
        _, rows = read_dir(match / f"C2_{model}")
        out["C2"][model] = c2(rows, base_rows, n_boot)
        out["C7"][f"{model}_vs_sf"] = c7_repetition(rows)
    if quality is not None and dump is not None:
        with np.load(Path(dump) / "positions.npz") as data:
            player, game, ply = data["player"], data["game"], data["ply"]
        with np.load(Path(quality) / "base.npz") as data:
            loss_base = data["loss"]
        for model in main:
            with np.load(Path(quality) / f"{model}.npz") as data:
                out["C4"][model] = c4(data["loss"], loss_base, player, game, ply, n_boot)
    return out


def fmt(iv: dict, digits: int = 1) -> str:
    return f"{iv['value']:+.{digits}f} [{iv['low']:+.{digits}f}, {iv['high']:+.{digits}f}]"


def report(result: dict) -> str:
    lines = ["# C1·C2·C4 판정", "", "| 모델 | C1 Elo (대 베이스) | C1 | C2 Elo 차이 | C2 | C4 손실 차이 | C4 |", "|---|---|---|---|---|---|---|"]
    for model in result["models"]:
        c1r, c2r, c4r = result["C1"][model], result["C2"][model], result["C4"].get(model)
        c4s = (fmt(c4r["diff"], 4), c4r["verdict"]) if c4r else ("—", "—")
        lines.append(f"| {model} ({NAMES.get(model, model)}) | {fmt(c1r['elo'])} | {c1r['verdict']} | "
                     f"{fmt(c2r['diff'])} | {c2r['verdict']} | {c4s[0]} | {c4s[1]} |")
    lines += ["", "C7 반복 종료 비율: " + ", ".join(f"{k} {v:.3f}" for k, v in result["C7"].items()), ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C1·C2·C4 판정과 C2 노드 수 보정")
    sub = parser.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--match", type=Path, required=True)
    j.add_argument("--quality", type=Path, default=None)
    j.add_argument("--dump", type=Path, default=None)
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--models", nargs="+", default=list(MAIN))
    c = sub.add_parser("calibrate")
    c.add_argument("dirs", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        print(json.dumps(calibrate(args.dirs), ensure_ascii=False, indent=1))
        return 0
    result = judge(args.match, args.quality, args.dump, args.models)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (args.out / "report.md").write_text(report(result), encoding="utf-8")
    print(report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
