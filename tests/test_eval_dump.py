import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.cache_base_q import build_cache
from src.data.dataset import PositionDataset, collate_fn
from src.eval import dump as dump_module
from src.eval.dump import PLAYER_ORDER, dump_model, fen_key, load_positions, main
from src.eval.metrics import METRICS, count_ties, per_position, segment_argmax, summarize
from src.model.lora import load_adapter
from src.model.transformer import ActionValueTransformer, load_base, save_base
from src.train.train import Objective, build_parser, evaluate, train
from test_train import TINY

ROOT = Path(__file__).resolve().parent.parent
GAME_PLIES = 26


def random_games(count: int, seed: int, first_id: int) -> list[dict]:
    """무작위 대국의 모든 국면 (전처리 기록처럼 fen, move, ply, game_id를 채운다)."""
    rng = random.Random(seed)
    records = []
    for game_id in range(first_id, first_id + count):
        board = chess.Board()
        for ply in range(GAME_PLIES):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            records.append({"fen": board.fen(), "move": move.uci(), "ply": ply, "game_id": game_id})
            board.push(move)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """세 기사의 소형 데이터·캐시, 소형 베이스. 스미슬로프 val의 ply ≥ 20 국면 하나를 카르포프 train에 넣는다."""
    root = tmp_path_factory.mktemp("dump")
    data_root = root / "data"
    val = {player: random_games(2, 10 + i, first_id=100 * i) for i, player in enumerate(PLAYER_ORDER)}
    train_records = {player: random_games(3, 20 + i, first_id=100 * i + 10) for i, player in enumerate(PLAYER_ORDER)}
    shared = next(r for r in val["smyslov"] if r["ply"] >= 20)
    train_records["karpov"].append({**shared, "game_id": 999})

    torch.manual_seed(0)
    base = root / "tiny.pt"
    save_base(ActionValueTransformer(TINY), base)
    model = load_base(base).eval()
    for player in PLAYER_ORDER:
        data_dir = data_root / player
        data_dir.mkdir(parents=True)
        write_jsonl(data_dir / "train.jsonl", train_records[player])
        write_jsonl(data_dir / "val.jsonl", val[player])
        build_cache(model, data_dir, base, batch_moves=200)
    return root, data_root, base, val, train_records, shared


def run_args(base, data_dir, out, player, seed):
    return build_parser().parse_args([
        "--player", player, "--base", str(base), "--out", str(out), "--data-dir", str(data_dir),
        "--rank", "4", "--lora-alpha", "8", "--lr", "1e-2", "--epochs", "2", "--batch-positions", "16",
        "--max-moves-per-forward", "64", "--max-steps", "6", "--log-every", "2", "--eval-every", "3",
        "--seed", str(seed), "--device", "cpu", "--num-workers", "0",
    ])


@pytest.fixture(scope="module")
def runs(workspace):
    """탈 학습 하나(Ta)와 카르포프 학습 하나(Ka)."""
    root, data_root, base, *_ = workspace
    outs = {}
    for name, player in (("Ta", "tal"), ("Ka", "karpov")):
        outs[name] = root / "runs" / name
        train(run_args(base, data_root / player, outs[name], player, seed=0))
    return outs


@pytest.fixture(scope="module")
def dumped(workspace, runs):
    root, data_root, *_ = workspace
    out = root / "dump"
    results = {name: dump_model(run, out, "cpu", num_workers=0, data_root=data_root) for name, run in runs.items()}
    return out, results


def slices(positions):
    """기사별 (국면 선택, 합법수 구간, 기사 안 offsets)."""
    for index, player in enumerate(PLAYER_ORDER):
        (rows,) = np.nonzero(positions["player"] == index)
        offsets = positions["offsets"][rows[0]:rows[-1] + 2]
        yield player, rows, slice(int(offsets[0]), int(offsets[-1])), offsets - offsets[0]


# ── 국면 정보 ─────────────────────────────────────────────


def test_positions_follow_records_and_cache(workspace, dumped):
    _, data_root, base, val, *_ = workspace
    out, _ = dumped
    positions, settings = load_positions(out)
    assert settings["temperature"] == 0.02 and settings["eval_amp"] is False
    for player, rows, span, offsets in slices(positions):
        dataset = PositionDataset(data_root / player, "val", base)
        assert len(rows) == len(val[player]) == len(dataset)
        assert positions["game"][rows].tolist() == [r["game_id"] for r in val[player]]
        assert positions["ply"][rows].tolist() == [r["ply"] for r in val[player]]
        assert np.array_equal(offsets, dataset.offsets)
        # target이 가리키는 합법수가 기사가 둔 수다
        assert np.array_equal(dataset.actions[offsets[:-1] + positions["target"][rows]], dataset.targets)
        q = dataset.q
        expected = np.concatenate([q[a:b].max() - q[a:b] for a, b in zip(offsets[:-1], offsets[1:])])
        assert np.array_equal(positions["regret"][span], expected)


def test_new_positions(workspace, dumped):
    *_, val, train_records, shared = workspace
    out, _ = dumped
    positions, _ = load_positions(out)
    train_keys = {fen_key(r["fen"]) for records in train_records.values() for r in records}
    expected = [r["ply"] >= 20 and fen_key(r["fen"]) not in train_keys for p in PLAYER_ORDER for r in val[p]]
    assert positions["new"].tolist() == expected
    # 다른 기사(카르포프) train에 있는 국면은 새 국면이 아니다
    index = sum(len(val[p]) for p in PLAYER_ORDER[:2]) + val["smyslov"].index(shared)
    assert positions["ply"][index] >= 20 and not positions["new"][index]
    assert positions["new"].any()


# ── 덤프와 학습 평가의 일치 ─────────────────────────────────


def reference_metrics(data_dir, base, run_dir, settings):
    loader = DataLoader(PositionDataset(data_dir, "val", base), batch_size=settings["batch_positions"],
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    objective = Objective(temperature=settings["temperature"], tau=settings["tau"])
    if run_dir is None:
        return evaluate(None, loader, objective, settings["eval_max_moves"])
    model = load_base(base)
    load_adapter(model, run_dir / "adapter.pt")
    return evaluate(model.eval(), loader, objective, settings["eval_max_moves"])


@pytest.mark.parametrize("name", ["base", "Ta", "Ka"])
def test_dump_reproduces_evaluate_on_every_val(workspace, runs, dumped, name):
    """세 기사 val 모두에서, 덤프로 다시 계산한 7개 지표가 학습 평가(evaluate)와 같다."""
    _, data_root, base, *_ = workspace
    out, _ = dumped
    positions, settings = load_positions(out)
    with np.load(out / f"{name}.npz") as data:
        log_pi = data["log_pi"]
    assert len(log_pi) == positions["offsets"][-1]
    for player, rows, span, offsets in slices(positions):
        values = per_position(log_pi[span], offsets, positions["target"][rows], positions["regret"][span],
                              settings["tau"])
        got = summarize(values)
        expected = reference_metrics(data_root / player, base, runs.get(name), settings)
        for metric in METRICS:
            if metric in ("top1", "top3"):
                assert got[metric] == expected[metric], (player, metric)
            else:
                assert got[metric] == pytest.approx(expected[metric], abs=1e-6), (player, metric)


def test_record_passes_and_matches_log(runs, dumped):
    out, results = dumped
    for name, result in results.items():
        assert result["pass"] and result["eval"]["pass"] and result["baseline"]["pass"]
        assert result["eval"]["step"] == 6
        saved = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        assert saved["adapter_sha256"] == dump_module.sha256(runs[name] / "adapter.pt")
        assert saved["player"] == {"Ta": "tal", "Ka": "karpov"}[name]


def test_rerun_skips_passed_dump(workspace, runs, dumped):
    _, data_root, *_ = workspace
    out, _ = dumped
    before = (out / "Ta.npz").stat().st_mtime_ns
    result = dump_model(runs["Ta"], out, "cpu", num_workers=0, data_root=data_root)
    assert result.get("skipped") and (out / "Ta.npz").stat().st_mtime_ns == before


def test_swapped_adapter_fails(workspace, runs, tmp_path, capsys):
    _, data_root, *_ = workspace
    swapped = tmp_path / "swapped"
    shutil.copytree(runs["Ta"], swapped)
    shutil.copy(runs["Ka"] / "adapter.pt", swapped / "adapter.pt")
    code = main([str(swapped), "--out", str(tmp_path / "out"), "--device", "cpu", "--num-workers", "0",
                 "--data-root", str(data_root)])
    assert code == 1
    assert "판정: 실패" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / "swapped.json").read_text(encoding="utf-8"))["pass"] is False


def test_rejects_different_settings(workspace, runs, dumped, tmp_path):
    _, data_root, *_ = workspace
    out, _ = dumped
    other = tmp_path / "Tb"
    shutil.copytree(runs["Ta"], other)
    config = json.loads((other / "config.json").read_text(encoding="utf-8"))
    config["temperature"] = 0.05
    (other / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="덤프 설정"):
        dump_model(other, out, "cpu", num_workers=0, data_root=data_root)
    code = main([str(other), "--out", str(out), "--device", "cpu", "--num-workers", "0",
                 "--data-root", str(data_root)])
    assert code == 3


# ── 국면별 값 (손 계산) ─────────────────────────────────────


def test_segment_argmax_and_ties():
    values = np.array([0.1, 0.5, 0.5, -1.0, 0.0, 0.3], dtype=np.float32)
    offsets = np.array([0, 3, 4, 6])
    assert segment_argmax(values, offsets).tolist() == [1, 0, 1]
    assert count_ties(values, offsets).tolist() == [2, 1, 1]


def test_per_position_by_hand():
    pi = np.array([0.7, 0.2, 0.1, 0.5, 0.5])
    log_pi = np.log(pi).astype(np.float32)
    offsets = np.array([0, 3, 5])
    target = np.array([1, 1])
    regret = np.array([0.0, 0.01, 0.05, 0.03, 0.0], dtype=np.float32)
    values = per_position(log_pi, offsets, target, regret, tau=0.02)
    assert values["top1"].tolist() == [0.0, 0.0]  # 두 번째 국면은 동률이라 앞 번호(0)가 1순위
    assert values["top3"].tolist() == [1.0, 1.0]
    assert values["choice"].tolist() == [0, 0]
    assert values["imitation"] == pytest.approx([-np.log(0.2), -np.log(0.5)], rel=1e-6)
    assert values["regret"] == pytest.approx([0.2 * 0.01 + 0.1 * 0.05, 0.5 * 0.03], rel=1e-5)
    assert values["regret_top1"] == pytest.approx([0.0, 0.03])
    assert values["band_violation"] == pytest.approx([0.1, 0.5], rel=1e-5)
    assert values["strength"] == pytest.approx([0.1 * 0.03, 0.5 * 0.01], rel=1e-4)


# ── 서버 작업 스크립트 ──────────────────────────────────────


def run_dump_job(tmp_path, data_root, *folders):
    out = tmp_path / "dumpjob"
    env = {**os.environ, "SITE_ENV": str(tmp_path / "없음.env"), "DUMP_OUT": str(out), "DUMP_DEVICE": "cpu",
           "DUMP_NUM_WORKERS": "0", "DUMP_DATA_ROOT": str(data_root), "DUMP_SKIP_INPUT_CHECK": "1"}
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", "gpu_train/dump.sbatch", *map(str, folders)], cwd=ROOT,
                            capture_output=True, text=True, env=env)
    return result, out


def test_dump_job_reports_pass_failure_and_error(workspace, runs, tmp_path):
    _, data_root, *_ = workspace
    swapped = tmp_path / "swapped"
    shutil.copytree(runs["Ta"], swapped)
    shutil.copy(runs["Ka"] / "adapter.pt", swapped / "adapter.pt")
    missing = tmp_path / "missing"
    missing.mkdir()
    result, out = run_dump_job(tmp_path, data_root, swapped, missing, runs["Ka"])
    assert result.returncode == 1, result.stdout + result.stderr
    (meta,) = out.glob("meta_local_*.txt")
    text = meta.read_text(encoding="utf-8")
    assert "git HEAD:" in text and "torch:" in text
    assert "  swapped 실패" in text
    assert "  missing 오류(종료 코드 3)" in text
    assert "  Ka 통과" in text  # 앞의 실패·오류 뒤에도 덤프한다
    assert {"positions.npz", "base.npz", "Ka.npz", "Ka.json"} <= {p.name for p in out.iterdir()}


def run_script(*args, env=None):
    return subprocess.run(["bash", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


@pytest.mark.parametrize("script", ["gpu_train/dump.sbatch", "gpu_train/submit_dump.sh"])
def test_dump_scripts_parse(script):
    assert run_script("-n", script).returncode == 0


def fake_run(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for name in ("config.json", "log.jsonl", "adapter.pt"):
        (folder / name).write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def site_env(tmp_path):
    path = tmp_path / "site.env"
    path.write_text("REPO_DIR=/srv/repo\nSBATCH_PARTITION=gpu\nSBATCH_GRES=gpu:1\nSBATCH_TIME=06:00:00\n"
                    "SBATCH_CPUS=4\nSBATCH_MEM=32G\nSBATCH_ACCOUNT=\nSBATCH_EXTRA=--nodelist=moana-u1\n",
                    encoding="utf-8")
    return path


def test_submit_dump_dry_run_pins_node(site_env, tmp_path):
    first, second = fake_run(tmp_path / "a" / "R1"), fake_run(tmp_path / "a" / "R1s0")
    result = run_script("gpu_train/submit_dump.sh", str(first), str(second),
                        env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    assert command[:2] == ["sbatch", "--parsable"]
    for option in ("--job-name=dump", "--output=runs/gpu/logs/%x_%j.out", "--chdir=/srv/repo", "--partition=gpu",
                   "--gres=gpu:1", "--time=06:00:00", "--nodelist=moana-u1"):
        assert option in command
    assert not any(option.startswith(("--account", "--dependency")) for option in command)
    assert command[-3:] == ["gpu_train/dump.sbatch", str(first), str(second)]


def test_submit_dump_rejects_incomplete_or_clashing_folders(site_env, tmp_path):
    env = {"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"}
    broken = fake_run(tmp_path / "broken")
    (broken / "log.jsonl").unlink()
    result = run_script("gpu_train/submit_dump.sh", str(broken), env=env)
    assert result.returncode == 1 and "log.jsonl가 없다" in result.stderr
    first, second = fake_run(tmp_path / "x" / "R1"), fake_run(tmp_path / "y" / "R1")
    result = run_script("gpu_train/submit_dump.sh", str(first), str(second), env=env)
    assert result.returncode == 1 and "겹친다" in result.stderr


# ── 덤프 → 판정 (소형 실제 파이프라인) ─────────────────────────


def test_judge_runs_on_real_dump(workspace, runs, tmp_path):
    """소형 학습 폴더로 6개 모델을 덤프하고 판정까지 돈다. seed 0 자리에는 같은 어댑터의 복사본을 둔다."""
    from src.eval.judge import main as judge_main

    root, data_root, base, *_ = workspace
    smyslov = root / "runs" / "Sm"
    if not smyslov.exists():
        train(run_args(base, data_root / "smyslov", smyslov, "smyslov", seed=0))
    folders = tmp_path / "runs"
    for source, names in ((runs["Ta"], ("R1", "R1s0")), (runs["Ka"], ("R8", "R8s0")), (smyslov, ("R2", "R2s0"))):
        for name in names:
            shutil.copytree(source, folders / name)
    out = tmp_path / "dump"
    for name in ("R1", "R8", "R2", "R1s0", "R8s0", "R2s0"):
        assert dump_model(folders / name, out, "cpu", num_workers=0, data_root=data_root)["pass"]
    code = judge_main([str(out), "--out", str(tmp_path / "judge"), "--n-boot", "200", "--bundle-repeats", "5"])
    assert code == 0
    result = json.loads((tmp_path / "judge" / "results.json").read_text(encoding="utf-8"))
    assert result["positions"] == int(load_positions(out)[0]["offsets"].size - 1)
    for p in ("R1", "R8", "R2"):  # seed 0 자리가 복사본이라 잡음 바닥은 0이다
        assert result["B2_B3"]["noise"][p]["value"] == 0.0
    assert (tmp_path / "judge" / "report.md").read_text(encoding="utf-8").startswith("# 특성 없는 평가 결과")
