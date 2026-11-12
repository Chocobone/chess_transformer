"""덤프로 특성 없는 지표를 계산하고 판정한다 (기준 문서 3.5·4.7·5.4절). numpy만 쓴다.

    python -m src.eval.judge runs/gpu/dump --out runs/gpu/judge

판정 대상은 기사별 seed 1 어댑터(--main)이고, seed 0 어댑터(--seed0)는 B3의 잡음 바닥에 쓴다.
seed 0 어댑터의 A·B10·C3 값은 참고로 함께 내고 판정하지 않는다.
덤프의 학습 기록 대조({모델 ID}.json)가 통과하지 않은 모델이 있으면 판정하지 않는다 (종료 코드 2).

결과: {out}/results.json (모든 값·구간·판정), {out}/report.md (표)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval import metrics, stats
from src.players import PLAYERS

PLAYER_ORDER = tuple(PLAYERS)  # 탈, 카르포프, 스미슬로프
NAMES = {"tal": "탈", "karpov": "카르포프", "smyslov": "스미슬로프"}
MAIN = ("R1", "R8", "R2")
SEED0 = ("R1s0", "R8s0", "R2s0")


class DumpError(RuntimeError):
    pass


def load_data(dump: Path, models: list[str]) -> tuple[metrics.Data, dict, dict]:
    """덤프를 읽는다. 모델마다 대조 기록을 확인하고 기사가 PLAYERS 순서와 맞는지 본다."""
    dump = Path(dump)
    with np.load(dump / "positions.npz") as data:
        arrays = {name: data[name] for name in data.files if name != "settings"}
        settings = json.loads(str(data["settings"]))
    records, log_pi = {}, {}
    with np.load(dump / "base.npz") as data:
        log_pi["base"] = data["log_pi"]
    for model in models:
        path = dump / f"{model}.json"
        if not path.exists():
            raise DumpError(f"{path}가 없다")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("pass"):
            raise DumpError(f"{model}의 학습 기록 대조가 통과하지 않았다")
        records[model] = record
        with np.load(dump / f"{model}.npz") as data:
            log_pi[model] = data["log_pi"]
        if len(log_pi[model]) != arrays["offsets"][-1]:
            raise DumpError(f"{model}.npz의 길이가 positions.npz와 다르다")
    data = metrics.Data(player=arrays["player"], game=arrays["game"], ply=arrays["ply"], offsets=arrays["offsets"],
                        target=arrays["target"], regret=arrays["regret"], new=arrays["new"], log_pi=log_pi,
                        tau=settings["tau"])
    return data, settings, records


def check_roles(records: dict, main: list[str], seed0: list[str]) -> None:
    for i, player in enumerate(PLAYER_ORDER):
        for model in (main[i], seed0[i]):
            if records[model]["player"] != player:
                raise DumpError(f"{model}의 기사는 {records[model]['player']}인데 {player} 자리에 있다")


def a_verdicts(a: dict) -> dict:
    """A1(top1·imitation 둘 다), A2, A3의 판정 (3.3절)."""
    top1 = stats.verdict(stats.Interval(**a["A1"]["all"]["top1"]), "above")
    imitation = stats.verdict(stats.Interval(**a["A1"]["all"]["imitation"]), "below")
    a1 = "통과" if top1 == imitation == "통과" else "실패" if "실패" in (top1, imitation) else "판정 보류"
    return {"A1": a1, "A1_top1": top1, "A1_imitation": imitation,
            "A2": stats.verdict(stats.Interval(**a["A2"]["top1"]), "above"),
            "A3": stats.verdict(stats.Interval(**a["A3"]["r"]), "above")}


def combine_a(per_player: dict) -> str:
    """특성 없는 A (3.5절, 2.5절). A5·A6는 특성 이슈 전이라 판정 보류다."""
    verdicts = [v for item in per_player.values() for k, v in item["verdict"].items() if k in ("A1", "A2", "A3")]
    if "실패" in verdicts:
        return "실패"
    if "통과" in verdicts:
        return "부분 통과 (A5·A6 판정 보류)" if all(v == "통과" for v in verdicts) else \
            "부분 통과 (A5·A6와 일부 지표 판정 보류)"
    return "판정 보류"


def combine_b(b3: str, b10: str) -> str:
    """특성 없는 층 (4.7절): B3 통과 + B10 통과면 통과, 어느 쪽이든 실패면 실패, 그 밖 판정 보류."""
    if "실패" in (b3, b10):
        return "실패"
    return "통과" if b3 == b10 == "통과" else "판정 보류"


def judge(dump: Path, main: list[str] = MAIN, seed0: list[str] = SEED0, n_boot: int = stats.N_BOOT,
          bundle_repeats: int = 1000) -> dict:
    main, seed0 = list(main), list(seed0)
    data, settings, records = load_data(dump, main + seed0)
    check_roles(records, main, seed0)

    result = {"dump": str(dump), "settings": settings, "n_boot": n_boot, "main": main, "seed0": seed0,
              "positions": int(len(data.target)), "games": int(data.games(np.ones(len(data.target), bool))[1]),
              "dump_records": {m: {k: records[m][k] for k in ("adapter_sha256", "environment", "pass")}
                               for m in main + seed0},
              "A": {}, "C3": {}}
    for i, player in enumerate(PLAYER_ORDER):
        entry = {}
        for role, model in (("main", main[i]), ("seed0", seed0[i])):
            entry[role] = metrics.a_metrics(data, i, model, n_boot)
        entry["verdict"] = a_verdicts(entry["main"])
        result["A"][player] = entry
        c3 = {role: metrics.c3_metrics(data, i, model, n_boot) for role, model in (("main", main[i]),
                                                                                   ("seed0", seed0[i]))}
        c3["base"] = metrics.c3_metrics(data, i, "base", n_boot)
        c3["verdict"] = stats.verdict(stats.Interval(**c3["main"]["regret_top1"]), "below", metrics.TAU)
        result["C3"][player] = c3
    result["A_verdict"] = combine_a(result["A"])

    result["B2_B3"] = metrics.b2_b3(data, main, seed0, n_boot)
    result["B10"] = metrics.b10_metrics(data, main, n_boot, bundle_repeats=bundle_repeats)
    result["B10_all_ply"] = metrics.b10_metrics(data, main, n_boot, min_ply=0, bundles=(),
                                                bundle_repeats=bundle_repeats)
    result["B10_seed0"] = metrics.b10_metrics(data, seed0, n_boot, bundles=(), bundle_repeats=bundle_repeats)
    result["B_verdict"] = combine_b(result["B2_B3"]["B3"], result["B10"]["B10"])
    if result["B2_B3"]["B3"] == "판정 보류":
        result["B_note"] = "B3 판정 보류: B 전체를 판정 보류로 두고 나머지 지표는 보고만 한다 (4.3절)"
    result["B1"] = metrics.b1_metrics(data, main)
    result["B11"] = metrics.b11_metrics(data, main)
    result["C3_verdict"] = {p: result["C3"][p]["verdict"] for p in PLAYER_ORDER}
    return result


# ── 보고 ──────────────────────────────────────────────


def fmt(iv: dict, digits: int = 4) -> str:
    return f"{iv['value']:+.{digits}f} [{iv['low']:+.{digits}f}, {iv['high']:+.{digits}f}]"


def report(result: dict) -> str:
    main, seed0 = result["main"], result["seed0"]
    names = [NAMES[p] for p in PLAYER_ORDER]
    lines = [f"# 특성 없는 평가 결과", "",
             f"- 덤프 `{result['dump']}`, S_val {result['positions']}국면 {result['games']}국, "
             f"부트스트랩 {result['n_boot']}회",
             f"- 판정 대상 {', '.join(main)}, 잡음 바닥 {', '.join(seed0)}", "",
             "## 판정 요약", "",
             "| 항목 | 판정 |", "|---|---|",
             f"| 특성 없는 A (A1~A3) | {result['A_verdict']} |",
             f"| B3 잡음 바닥 | {result['B2_B3']['B3']} |",
             f"| B10 우도 혼동 행렬 | {result['B10']['B10']} (3지선다 {result['B10']['three_way']}) |",
             f"| 특성 없는 층 B | {result['B_verdict']} |",
             "| C3 (단독 판정 불가) | " + ", ".join(f"{n} {result['C3_verdict'][p]}"
                                                for n, p in zip(names, PLAYER_ORDER)) + " |", "",
             "## A 재현 (어댑터 − 베이스, 기사 자기 val)", "",
             "| 기사 | A1 Δtop1 | A1 Δimitation | A2 새 국면 Δtop1 | A3 r | A1 | A2 | A3 |",
             "|---|---|---|---|---|---|---|---|"]
    for p, n in zip(PLAYER_ORDER, names):
        a, v = result["A"][p]["main"], result["A"][p]["verdict"]
        lines.append(f"| {n} | {fmt(a['A1']['all']['top1'])} | {fmt(a['A1']['all']['imitation'])} | "
                     f"{fmt(a['A2']['top1'])} | {fmt(a['A3']['r'])} | {v['A1']} | {v['A2']} | {v['A3']} |")
    lines += ["", "| 기사 | ply ≥ 20 Δtop1 | ply ≥ 20 Δimitation | Δtop3 | 새 국면 수 (대국) | 불일치 비율 | McNemar p |",
              "|---|---|---|---|---|---|---|"]
    for p, n in zip(PLAYER_ORDER, names):
        a = result["A"][p]["main"]
        lines.append(f"| {n} | {fmt(a['A1']['ply20']['top1'])} | {fmt(a['A1']['ply20']['imitation'])} | "
                     f"{fmt(a['A1']['all']['top3'])} | {a['A2']['positions']} ({a['A2']['games']}) | "
                     f"{a['A3']['share']:.4f} | {a['A3']['mcnemar_p']:.2e} |")

    b = result["B2_B3"]
    lines += ["", "## B3 잡음 바닥 (S_val, 기사별 층화)", "", "| 잡음 쌍 | n_p |", "|---|---|"]
    for p in main:
        lines.append(f"| {b['noise'][p]['pair']} | {fmt(b['noise'][p], 5)} |")
    lines += ["", "| 기사 쌍 | d(p, q) | M (전체) | M (ply ≥ 20, 보고) | 판정 |", "|---|---|---|---|---|"]
    for key, margin in b["margins"].items():
        p, q = key.split("-")
        pair = b["pairs"].get(f"{p}-{q}") or b["pairs"][f"{q}-{p}"]
        lines.append(f"| {key} | {fmt(pair['d'], 5)} | {fmt(margin['all'], 5)} | {fmt(margin['ply20'], 5)} | "
                     f"{margin['verdict']} |")
    lines += ["", "| 모델 쌍 | d | d (ply ≥ 20) | δ | δ (ply ≥ 20) |", "|---|---|---|---|---|"]
    for key, pair in b["pairs"].items():
        lines.append(f"| {key} | {pair['d']['value']:.5f} | {pair['d_ply20']['value']:.5f} | "
                     f"{pair['delta']:.4f} | {pair['delta_ply20']:.4f} |")

    b10 = result["B10"]
    lines += ["", "## B10 우도 혼동 행렬 (ply ≥ 20, k = 1)", "",
              "| 실제 \\ 분류 | " + " | ".join(names) + " | 재현율 [구간] | 판정 | 대국 (제외, 동률) |",
              "|---|" + "---|" * (len(names) + 3)]
    for i, (m, n) in enumerate(zip(main, names)):
        r = b10["recall"][m]
        lines.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in b10["matrix"][i]) +
                     f" | {r['value']:.3f} [{r['low']:.3f}, {r['high']:.3f}] | {r['verdict']} | "
                     f"{r['games']} ({b10['excluded_games'][i]}, {b10['ties'][i]}) |")
    lines.append("| 열 합계 | " + " | ".join(str(x) for x in b10["column_totals"]) + " | | | |")
    lines += ["", "| 쌍별 2지선다 | 재현율 [구간] | 판정 |", "|---|---|---|"]
    for key, pair in b10["pairwise"].items():
        parts = [f"{m} {pair[m]['value']:.3f} [{pair[m]['low']:.3f}, {pair[m]['high']:.3f}]" for m in key.split("-")]
        lines.append(f"| {key} | {'; '.join(parts)} | {pair['verdict']} |")
    lines += ["", "| 묶음 k (보고) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, row in b10["bundles"].items():
        lines.append(f"| {k} | " + " | ".join(f"{row[m]:.3f}" for m in main) + " |")
    lines += ["", "| 보고 | " + " | ".join(names) + " | 3지선다 |", "|---|" + "---|" * (len(names) + 1)]
    for title, key, models in (("전체 ply 재현율", "B10_all_ply", main), ("seed 0 어댑터 재현율", "B10_seed0", seed0)):
        rec = result[key]["recall"]
        lines.append(f"| {title} | " + " | ".join(f"{rec[m]['value']:.3f}" for m in models) +
                     f" | {result[key]['three_way']} |")

    b1 = result["B1"]
    lines += ["", "## B1 교차 평가 (보고)", ""]
    for part, title in (("all", "전체"), ("ply20", "ply ≥ 20")):
        lines += [f"| val \\ 어댑터 ({title}) | 베이스 | " + " | ".join(main) + " | 기사 특이 몫 |",
                  "|---|" + "---|" * (len(main) + 2)]
        for i, n in enumerate(names):
            lines.append(f"| {n} imitation | {b1[part]['base_imitation'][i]:.4f} | " +
                         " | ".join(f"{x:.4f}" for x in b1[part]["imitation"][i]) +
                         f" | {b1[part]['specific_share'][i]:.1%} |")
            lines.append(f"| {n} top1 | {b1[part]['base_top1'][i]:.4f} | " +
                         " | ".join(f"{x:.4f}" for x in b1[part]["top1"][i]) + " | |")
        lines.append("")
    b11 = result["B11"]
    lines += ["## B11 불일치 국면 적중률 (보조)", "", "| val \\ 어댑터 | " + " | ".join(main) + " | 국면 수 |",
              "|---|" + "---|" * (len(main) + 1)]
    for i, n in enumerate(names):
        lines.append(f"| {n} | " + " | ".join(f"{x:.4f}" for x in b11["rates"][i]) + f" | {b11['positions'][i]} |")

    lines += ["", "## C3 베이스 기준 regret (단독 판정 불가)", "",
              "| 기사 | regret_top1 [구간] | 판정 | regret | band_violation | strength | 베이스 regret | seed 0 regret_top1 |",
              "|---|---|---|---|---|---|---|---|"]
    for p, n in zip(PLAYER_ORDER, names):
        c = result["C3"][p]
        lines.append(f"| {n} | {fmt(c['main']['regret_top1'], 5)} | {c['verdict']} | {c['main']['regret']:.5f} | "
                     f"{c['main']['band_violation']:.4f} | {c['main']['strength']:.5f} | {c['base']['regret']:.5f} | "
                     f"{c['seed0']['regret_top1']['value']:.5f} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="덤프로 특성 없는 지표를 계산하고 판정한다")
    parser.add_argument("dump", type=Path, help="src.eval.dump의 출력 폴더")
    parser.add_argument("--out", type=Path, default=Path("runs/gpu/judge"))
    parser.add_argument("--main", default=",".join(MAIN), help="기사별 판정 대상 (탈,카르포프,스미슬로프 순)")
    parser.add_argument("--seed0", default=",".join(SEED0), help="기사별 seed 0 어댑터 (같은 순서)")
    parser.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    parser.add_argument("--bundle-repeats", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = judge(args.dump, args.main.split(","), args.seed0.split(","), args.n_boot, args.bundle_repeats)
    except DumpError as error:
        print(f"판정하지 않는다: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    text = report(result)
    (args.out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
