"""Task #13 3단계: C1·C2·C4 집계·판정, C7, 노드 수 보정 선택."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.eval import games, stats


def rows_from_pairs(pair_points: list[float], termination="checkmate") -> list[dict]:
    """짝 점수(0~2) 목록 → index.jsonl 줄. 백 대국에 몫을 먼저 준다."""
    rows = []
    for o, total in enumerate(pair_points):
        white = min(total, 1.0)
        for color, score in (("white", white), ("black", total - white)):
            rows.append({"opening": o, "model_color": color, "score": score, "termination": termination,
                         "ply": 60, "result": "*"})
    return rows


def write_dir(path: Path, config: dict, rows: list[dict]) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_elo_values():
    assert games.elo(0.5) == pytest.approx(0.0)
    assert games.elo(0.4285) == pytest.approx(-50.0, abs=0.2)  # 기준 문서: −50 Elo ≈ 점수율 0.429
    assert games.elo(0.75) == pytest.approx(400 * math.log10(3))
    assert games.elo(0.0) == pytest.approx(-400 * math.log10(999))  # 0.001로 자른다
    assert games.elo(1.0) == pytest.approx(400 * math.log10(999))


def test_pair_scores_and_errors():
    openings, pairs = games.pair_scores(rows_from_pairs([2, 1, 0.5]))
    assert openings.tolist() == [0, 1, 2] and pairs.tolist() == [2, 1, 0.5]
    with pytest.raises(games.GamesError):
        games.pair_scores(rows_from_pairs([1])[:1])  # 짝 미완성
    with pytest.raises(games.GamesError):
        games.pair_scores(rows_from_pairs([1]) + rows_from_pairs([1])[:1])  # 중복


def test_c1_verdicts():
    even = games.c1(rows_from_pairs([1.0] * 50), n_boot=2000)
    assert even["score"]["value"] == 0.5 and even["elo"] == {"value": 0.0, "low": 0.0, "high": 0.0}
    assert even["verdict"] == "통과" and even["draws"] == 0 and even["wins"] == 50
    weak = games.c1(rows_from_pairs([0.0] * 45 + [2.0] * 5), n_boot=2000)
    assert weak["score"]["value"] == pytest.approx(0.1) and weak["verdict"] == "실패"
    mixed = games.c1(rows_from_pairs([0.0, 2.0] * 10 + [1.0] * 2), n_boot=2000)
    assert mixed["verdict"] == "판정 보류"  # 구간이 −50을 걸친다


def test_c2_paired_identical_is_zero():
    rows = rows_from_pairs([0.0, 1.0, 2.0, 0.5, 1.5] * 8)
    result = games.c2(rows, rows, n_boot=2000)
    assert result["diff"] == {"value": 0.0, "low": 0.0, "high": 0.0}
    assert result["verdict"] == "통과"


def test_c2_paired_shift_and_mismatch():
    base = rows_from_pairs([0.5, 1.0, 0.0, 1.5] * 25)
    better = rows_from_pairs([1.0, 1.5, 0.5, 2.0] * 25)  # 모든 짝이 0.5점 높다
    result = games.c2(better, base, n_boot=2000)
    assert result["diff"]["value"] > 0 and result["diff"]["low"] > 0 and result["verdict"] == "통과"
    worse = games.c2(base, better, n_boot=2000)
    assert worse["diff"]["high"] < -50 and worse["verdict"] == "실패"
    with pytest.raises(games.GamesError):
        games.c2(rows_from_pairs([1.0] * 3), rows_from_pairs([1.0] * 4))


def test_c4_diff_stratified_and_verdict():
    player = np.repeat([0, 1, 2], 40)
    game = np.tile(np.repeat(np.arange(10), 4), 3)
    ply = np.tile([10, 25, 30, 40], 30)
    base = np.full(120, 0.01)
    model = base + 0.005
    out = games.c4(model, base, player, game, ply, n_boot=2000)
    assert out["diff"]["value"] == pytest.approx(0.005)
    assert out["diff"]["low"] == pytest.approx(0.005) and out["diff"]["high"] == pytest.approx(0.005)
    assert out["verdict"] == "통과" and out["diff_ply20"]["value"] == pytest.approx(0.005)
    bad = games.c4(base + 0.05, base, player, game, ply, n_boot=2000)
    assert bad["verdict"] == "실패"
    blunders = model.copy()
    blunders[:12] = 0.5
    out = games.c4(blunders, base, player, game, ply, n_boot=2000)
    assert out["blunder_model"] == pytest.approx(0.1) and out["blunder_base"] == 0.0
    assert set(out["per_player"]) == {0, 1, 2}


def test_c7_repetition():
    rows = rows_from_pairs([1, 1], "threefold_repetition") + rows_from_pairs([1, 1, 1], "checkmate")
    assert games.c7_repetition(rows) == pytest.approx(4 / 10)


def cal_dirs(tmp_path, scores: dict[int, float], pairs: int = 50) -> list[Path]:
    dirs = []
    for nodes, s in scores.items():
        total = round(s * 2 * pairs)
        points = [2.0] * (total // 2) + [1.0] * (total % 2)
        points += [0.0] * (pairs - len(points))
        dirs.append(write_dir(tmp_path / f"n{nodes}", {"model": "base", "opponent": "sf", "nodes": nodes},
                              rows_from_pairs(points)))
    return dirs


def test_calibrate_choose_and_next(tmp_path):
    result = games.calibrate(cal_dirs(tmp_path / "a", {100: 0.9, 300: 0.7, 1000: 0.45, 3000: 0.2, 10000: 0.0}))
    assert [r["nodes"] for r in result["rows"]] == [100, 300, 1000, 3000, 10000]
    assert result["chosen"] == 1000  # 0.35~0.65 안에서 0.5에 가장 가깝다
    assert result["next"] == [300, 1000, 3000]  # 0.5를 지나는 300·1000과 바깥 이웃 중 0.5에 가까운 3000
    tie = games.calibrate(cal_dirs(tmp_path / "b", {300: 0.55, 1000: 0.45}))
    assert tie["chosen"] == 300  # 같은 거리면 작은 노드 수


def test_calibrate_none_in_range(tmp_path):
    result = games.calibrate(cal_dirs(tmp_path / "c", {100: 0.8, 300: 0.7, 1000: 0.2}))
    assert result["chosen"] is None and result["extra"] == round(math.sqrt(300 * 1000))
    low = games.calibrate(cal_dirs(tmp_path / "d", {100: 0.3, 300: 0.2, 1000: 0.1, 3000: 0.0}))
    assert low["next"] == [100, 300, 1000] and low["extra"] is None


def test_calibrate_rejects_non_base(tmp_path):
    d = write_dir(tmp_path / "x", {"model": "R1", "opponent": "sf", "nodes": 100}, rows_from_pairs([1.0]))
    with pytest.raises(games.GamesError):
        games.calibrate([d])


def test_judge_end_to_end(tmp_path):
    match = tmp_path / "match"
    write_dir(match / "C2_base", {"model": "base", "opponent": "sf", "nodes": 300}, rows_from_pairs([1.0, 0.5] * 10))
    for model in ("R1", "R8", "R2"):
        write_dir(match / f"C1_{model}", {"model": model, "opponent": "base"}, rows_from_pairs([1.0] * 20))
        write_dir(match / f"C2_{model}", {"model": model, "opponent": "sf", "nodes": 300}, rows_from_pairs([1.0, 0.5] * 10))
    dump, q = tmp_path / "dump", tmp_path / "q"
    dump.mkdir(), q.mkdir()
    np.savez(dump / "positions.npz", player=np.repeat([0, 1, 2], 4), game=np.tile([0, 0, 1, 1], 3),
             ply=np.tile([5, 25, 5, 25], 3))
    for model in ("base", "R1", "R8", "R2"):
        np.savez(q / f"{model}.npz", loss=np.full(12, 0.01))
    result = games.judge(match, q, dump, n_boot=500)
    for model in ("R1", "R8", "R2"):
        assert result["C1"][model]["verdict"] == "통과"
        assert result["C2"][model]["diff"]["value"] == 0.0 and result["C2"][model]["verdict"] == "통과"
        assert result["C4"][model]["verdict"] == "통과"
    assert "R8 (카르포프)" in games.report(result)
