import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import judge as judge_module
from src.eval import metrics, stats
from src.eval.judge import MAIN, SEED0, judge, main

ROOT = Path(__file__).resolve().parent.parent
PLIES = (0, 10, 20, 22, 30, 40)  # 대국마다 국면 6개, ply ≥ 20이 4개
GAMES = 4
MOVES = 3
N_BOOT = 300


# ── 부트스트랩 ───────────────────────────────────────────


def test_group_index_orders_by_key():
    keys = np.array([[1, 5], [0, 9], [1, 5], [0, 2]])
    groups, n = stats.group_index(keys)
    assert n == 3 and groups.tolist() == [2, 1, 2, 0]


def test_resample_weights_shape_and_seed():
    a = stats.resample_weights(7, 50, np.random.default_rng(0))
    b = stats.resample_weights(7, 50, np.random.default_rng(0))
    assert a.shape == (50, 7) and np.array_equal(a, b)
    assert np.all(a.sum(axis=1) == 7)


def test_stratified_weights_keep_strata_sizes():
    weights = stats.stratified_weights([3, 5, 2], n_boot=40)
    assert weights.shape == (40, 10)
    assert np.all(weights[:, :3].sum(axis=1) == 3)
    assert np.all(weights[:, 3:8].sum(axis=1) == 5)
    assert np.all(weights[:, 8:].sum(axis=1) == 2)


def test_ratio_value_is_position_mean():
    # 대국 3개, 국면 수 1·2·3. 원표본 값은 대국 평균이 아니라 국면 평균이다
    num, den = np.array([1.0, 0.0, 3.0]), np.array([1.0, 2.0, 3.0])
    iv = stats.ratio_ci(num, den, stats.resample_weights(3, 200))
    assert iv.value == pytest.approx(4 / 6)
    assert iv.low <= iv.value <= iv.high


def test_single_group_has_zero_width():
    iv = stats.ratio_ci(np.array([2.0]), np.array([4.0]), stats.resample_weights(1, 100))
    assert iv.value == iv.low == iv.high == 0.5


def test_interval_skips_undefined_replicates():
    # 대국 2개 중 하나만 국면이 있다. 그 대국을 뽑지 않은 반복은 0/0이라 뺀다
    weights = stats.resample_weights(2, 500)
    value, boot = stats.boot_ratio(np.array([3.0, 0.0]), np.array([3.0, 0.0]), weights)
    assert np.isnan(boot).any() and not np.isnan(boot).all()
    iv = stats.interval(value, boot)
    assert (iv.value, iv.low, iv.high) == (1.0, 1.0, 1.0)


def test_binom_two_sided():
    assert stats.binom_two_sided(0, 0) == 1.0
    assert stats.binom_two_sided(0, 5) == pytest.approx(2 / 32)
    assert stats.binom_two_sided(3, 3) == 1.0
    assert stats.binom_two_sided(2, 8) == pytest.approx(2 * sum(math.comb(10, k) for k in range(3)) / 2 ** 10)


@pytest.mark.parametrize("low, high, passes, expected", [
    (0.1, 0.2, "above", "통과"), (-0.2, -0.1, "above", "실패"), (-0.1, 0.1, "above", "판정 보류"),
    (0.0, 0.1, "above", "판정 보류"),  # 하한이 정확히 0이면 통과가 아니다
    (-0.2, -0.1, "below", "통과"), (0.1, 0.2, "below", "실패"), (-0.1, 0.1, "below", "판정 보류"),
])
def test_verdict_boundaries(low, high, passes, expected):
    assert stats.verdict(stats.Interval(0.0, low, high), passes) == expected


def test_verdict_threshold():
    assert stats.verdict(stats.Interval(0.5, 0.34, 0.6), "above", 1 / 3) == "통과"
    assert stats.verdict(stats.Interval(0.2, 0.1, 0.3), "above", 1 / 3) == "실패"
    assert stats.verdict(stats.Interval(0.01, 0.005, 0.019), "below", 0.02) == "통과"


# ── JS ───────────────────────────────────────────────────


def test_js_identical_disjoint_symmetric():
    offsets = np.array([0, 2, 5])
    p = np.log(np.array([0.5, 0.5, 0.2, 0.3, 0.5]))
    q = np.log(np.array([0.9, 0.1, 0.5, 0.25, 0.25]))
    assert metrics.js_per_position(p, p, offsets) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert metrics.js_per_position(p, q, offsets) == pytest.approx(metrics.js_per_position(q, p, offsets))
    disjoint_p = np.log(np.array([1.0, 1e-300]))
    disjoint_q = np.log(np.array([1e-300, 1.0]))
    assert metrics.js_per_position(disjoint_p, disjoint_q, np.array([0, 2])) == pytest.approx([1.0], abs=1e-9)
    # 손 계산: P=(0.5, 0.5), Q=(0.9, 0.1), M=(0.7, 0.3)
    expected = 0.5 * (0.5 * math.log2(0.5 / 0.7) + 0.5 * math.log2(0.5 / 0.3)) + \
        0.5 * (0.9 * math.log2(0.9 / 0.7) + 0.1 * math.log2(0.1 / 0.3))
    assert metrics.js_per_position(p, q, offsets)[0] == pytest.approx(expected)


# ── 합성 덤프 ────────────────────────────────────────────
# 국면마다 합법수 3개, 기사 수는 0번. 자기 기사 val에서 자기 어댑터는 기사 수에 0.8을 주고,
# 다른 기사 val에서는 0.4(1순위는 1번 수)를 준다. 베이스는 어디서나 (0.3, 0.6, 0.1)이다.


def layout():
    player = np.repeat(np.arange(3), GAMES * len(PLIES))
    game = np.tile(np.repeat(np.arange(GAMES), len(PLIES)), 3)
    ply = np.tile(PLIES, 3 * GAMES)
    n = len(player)
    return player, game, ply, np.arange(n + 1) * MOVES


def policies(player, own, other, base=(0.3, 0.6, 0.1)):
    """모델별 log π. own[i]·other[i]: 어댑터 i가 자기·다른 기사 val에서 주는 분포."""
    out = {"base": np.log(np.tile(base, len(player))).astype(np.float32)}
    for i in range(3):
        rows = np.where((player == i)[:, None], own[i], other[i])
        out[i] = np.log(rows.reshape(-1)).astype(np.float32)
    return out


def make_data(identical=False):
    player, game, ply, offsets = layout()
    n = len(player)
    own = [(0.8, 0.1, 0.1)] * 3
    other = [(0.4, 0.5, 0.1)] * 3
    seed_own = [(0.79, 0.11, 0.1)] * 3
    if identical:  # 세 어댑터가 모두 같다
        own = other = seed_own = [(0.5, 0.4, 0.1)] * 3
    main_pi, seed_pi = policies(player, own, other), policies(player, seed_own, other)
    log_pi = {"base": main_pi["base"]}
    for i in range(3):
        log_pi[MAIN[i]], log_pi[SEED0[i]] = main_pi[i], seed_pi[i]
    data = metrics.Data(player=player.astype(np.int8), game=game.astype(np.int32), ply=ply.astype(np.int16),
                        offsets=offsets, target=np.zeros(n, dtype=np.int16),
                        regret=np.tile(np.array([0.0, 0.05, 0.1], dtype=np.float32), n),
                        new=(ply >= 20) & (game % 2 == 0), log_pi=log_pi)
    return data


def test_a_metrics_by_hand():
    data = make_data()
    a = metrics.a_metrics(data, 0, MAIN[0], n_boot=N_BOOT)
    assert a["positions"] == GAMES * len(PLIES) and a["games"] == GAMES
    assert a["A1"]["all"]["top1"] == {"value": 1.0, "low": 1.0, "high": 1.0}
    imit = a["A1"]["all"]["imitation"]
    assert imit["value"] == pytest.approx(-math.log(0.8) + math.log(0.3), rel=1e-6)
    assert a["A2"]["positions"] == 8  # 짝수 번호 대국 2개 × ply ≥ 20 국면 4개
    assert a["A2"]["games"] == 2
    assert a["A3"]["r"]["value"] == 1.0 and a["A3"]["share"] == 1.0
    assert (a["A3"]["adapter_only"], a["A3"]["base_only"]) == (24, 0)
    assert a["A3"]["mcnemar_p"] == pytest.approx(2 / 2 ** 24)
    assert a["A3"]["ply_all"] == {"0~9": 1 / 6, "10~19": 1 / 6, "20~39": 3 / 6, "40~": 1 / 6}


def test_c3_by_hand():
    data = make_data()
    c3 = metrics.c3_metrics(data, 1, MAIN[1], n_boot=N_BOOT)
    assert c3["regret_top1"]["value"] == 0.0  # 자기 val의 1순위는 기사 수(regret 0)
    base = metrics.c3_metrics(data, 1, "base", n_boot=N_BOOT)
    assert base["regret_top1"]["value"] == pytest.approx(0.05)
    assert base["regret"] == pytest.approx(0.6 * 0.05 + 0.1 * 0.1, rel=1e-5)
    assert base["band_violation"] == pytest.approx(0.7, rel=1e-5)


def test_b3_passes_when_seeds_agree_and_players_differ():
    b = metrics.b2_b3(make_data(), list(MAIN), list(SEED0), n_boot=N_BOOT)
    assert len(b["pairs"]) == 15
    for p in MAIN:
        assert 0 < b["noise"][p]["value"] < 0.01
    assert b["B3"] == "통과"
    assert all(m["verdict"] == "통과" for m in b["margins"].values())
    assert b["pairs"][f"{MAIN[0]}-{MAIN[1]}"]["delta"] == pytest.approx(2 / 3)  # 두 기사 val에서 1순위가 갈린다


def test_b3_holds_when_everything_is_identical():
    b = metrics.b2_b3(make_data(identical=True), list(MAIN), list(SEED0), n_boot=N_BOOT)
    assert all(m["all"]["value"] == pytest.approx(0.0, abs=1e-12) for m in b["margins"].values())
    assert b["B3"] == "판정 보류"


def test_b10_diagonal():
    b10 = metrics.b10_metrics(make_data(), list(MAIN), n_boot=N_BOOT, bundle_repeats=20)
    assert b10["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert all(r["value"] == 1.0 and r["verdict"] == "통과" for r in b10["recall"].values())
    assert all(pair["verdict"] == "통과" for pair in b10["pairwise"].values())
    assert b10["three_way"] == "통과" and b10["B10"] == "통과"
    assert b10["column_totals"] == [GAMES] * 3 and b10["ties"] == [0, 0, 0]
    assert b10["bundles"]["2"] == {m: 1.0 for m in MAIN}
    assert b10["excluded_games"] == [0, 0, 0]


def test_b10_ties_count_as_wrong():
    b10 = metrics.b10_metrics(make_data(identical=True), list(MAIN), n_boot=N_BOOT, bundles=())
    assert b10["ties"] == [GAMES] * 3
    assert all(r["value"] == 0.0 and r["verdict"] == "실패" for r in b10["recall"].values())
    assert b10["B10"] == "실패"


def test_b10_excludes_games_without_late_positions():
    data = make_data()
    data.ply[(data.player == 0) & (data.game == 0)] = 5  # 탈의 대국 하나는 ply 20 이상 국면이 없다
    b10 = metrics.b10_metrics(data, list(MAIN), n_boot=N_BOOT, bundles=())
    assert b10["excluded_games"] == [1, 0, 0]
    assert b10["recall"][MAIN[0]]["games"] == GAMES - 1


def test_b1_specific_share_by_hand():
    b1 = metrics.b1_metrics(make_data(), list(MAIN))
    expected = 1 - math.log(0.4 / 0.3) / math.log(0.8 / 0.3)
    assert b1["all"]["specific_share"] == pytest.approx([expected] * 3, rel=1e-5)
    assert b1["all"]["top1"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_b11_rates():
    b11 = metrics.b11_metrics(make_data(), list(MAIN))
    assert b11["positions"] == [GAMES * len(PLIES)] * 3
    assert b11["rates"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# ── 판정 묶음 ────────────────────────────────────────────


def write_dump(folder: Path, data: metrics.Data, players=("tal", "karpov", "smyslov"), failing=()):
    folder.mkdir(parents=True)
    settings = {"base": "136M.pt", "temperature": 0.02, "tau": 0.02, "batch_positions": 32,
                "eval_max_moves": 256, "eval_amp": False}
    np.savez(folder / "positions.npz", settings=json.dumps(settings), player=data.player, game=data.game,
             ply=data.ply, offsets=data.offsets, target=data.target, regret=data.regret, new=data.new)
    np.savez(folder / "base.npz", log_pi=data.log_pi["base"])
    for i, (m, s) in enumerate(zip(MAIN, SEED0)):
        for model in (m, s):
            np.savez(folder / f"{model}.npz", log_pi=data.log_pi[model])
            record = {"model": model, "player": players[i], "adapter_sha256": "0" * 64,
                      "environment": {"device": "cpu"}, "pass": model not in failing}
            (folder / f"{model}.json").write_text(json.dumps(record), encoding="utf-8")


def test_judge_end_to_end(tmp_path, capsys):
    write_dump(tmp_path / "dump", make_data())
    code = main([str(tmp_path / "dump"), "--out", str(tmp_path / "judge"), "--n-boot", str(N_BOOT),
                 "--bundle-repeats", "10"])
    assert code == 0
    result = json.loads((tmp_path / "judge" / "results.json").read_text(encoding="utf-8"))
    assert result["A_verdict"] == "부분 통과 (A5·A6 판정 보류)"
    assert result["B2_B3"]["B3"] == "통과" and result["B10"]["B10"] == "통과"
    assert result["B_verdict"] == "통과"
    assert result["C3_verdict"] == {"tal": "통과", "karpov": "통과", "smyslov": "통과"}
    assert result["positions"] == 3 * GAMES * len(PLIES) and result["games"] == 3 * GAMES
    text = (tmp_path / "judge" / "report.md").read_text(encoding="utf-8")
    assert "| 특성 없는 층 B | 통과 |" in text and "## B10" in text
    assert text in capsys.readouterr().out


def test_judge_combines_failures(tmp_path):
    write_dump(tmp_path / "dump", make_data(identical=True))
    result = judge(tmp_path / "dump", n_boot=N_BOOT, bundle_repeats=5)
    assert result["B2_B3"]["B3"] == "판정 보류" and result["B10"]["B10"] == "실패"
    assert result["B_verdict"] == "실패"


def test_judge_refuses_unverified_or_misplaced_dump(tmp_path, capsys):
    write_dump(tmp_path / "bad", make_data(), failing=("R8s0",))
    assert main([str(tmp_path / "bad"), "--out", str(tmp_path / "j1"), "--n-boot", "10"]) == 2
    assert "R8s0" in capsys.readouterr().err
    write_dump(tmp_path / "swap", make_data(), players=("karpov", "tal", "smyslov"))
    assert main([str(tmp_path / "swap"), "--out", str(tmp_path / "j2"), "--n-boot", "10"]) == 2
    assert not (tmp_path / "j1").exists() and not (tmp_path / "j2").exists()


def test_combine_rules():
    assert judge_module.combine_b("통과", "통과") == "통과"
    assert judge_module.combine_b("실패", "통과") == "실패"
    assert judge_module.combine_b("통과", "판정 보류") == "판정 보류"
    item = lambda *v: {"verdict": dict(zip(("A1", "A2", "A3"), v))}  # noqa: E731
    assert judge_module.combine_a({"a": item("통과", "통과", "통과")}) == "부분 통과 (A5·A6 판정 보류)"
    assert judge_module.combine_a({"a": item("통과", "판정 보류", "통과")}).startswith("부분 통과 (A5·A6와 일부")
    assert judge_module.combine_a({"a": item("통과", "실패", "통과")}) == "실패"
    assert judge_module.combine_a({"a": item("판정 보류", "판정 보류", "판정 보류")}) == "판정 보류"


def test_analysis_modules_do_not_import_torch():
    code = "import sys, src.eval.judge, src.eval.metrics, src.eval.stats; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
