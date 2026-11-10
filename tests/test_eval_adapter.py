import json
import shutil

import pytest

from gpu_train.eval_adapter import METRICS, TOLERANCE, compare, main, reevaluate
from src.train.train import build_parser, train
from test_train import workspace  # noqa: F401  (소형 베이스·데이터·q 캐시 fixture)


def run_args(base, data_dir, out, seed: int):
    return build_parser().parse_args([
        "--player", "tal", "--base", str(base), "--out", str(out), "--data-dir", str(data_dir),
        "--rank", "4", "--lora-alpha", "8", "--lr", "1e-2", "--epochs", "2", "--batch-positions", "16",
        "--max-moves-per-forward", "64", "--max-steps", "6", "--log-every", "2", "--eval-every", "3",
        "--seed", str(seed), "--device", "cpu", "--num-workers", "0",
    ])


@pytest.fixture(scope="module")
def runs(workspace):  # noqa: F811
    """seed만 다른 학습 두 개."""
    root, data_dir, base = workspace
    outs = []
    for seed in (0, 1):
        out = root / f"eval_run{seed}"
        train(run_args(base, data_dir, out, seed))
        outs.append(out)
    return outs


def last_line(out, kind):
    records = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    return [r for r in records if r["type"] == kind][-1]


def test_reproduces_last_eval_and_baseline(runs):
    result = reevaluate(runs[0], "cpu", num_workers=0, baseline=True)
    assert result["pass"] and result["eval"]["pass"] and result["baseline"]["pass"]
    assert result["eval"]["step"] == 6
    # 같은 장치·같은 조건이면 기록과 정확히 같다
    for part in ("eval", "baseline"):
        assert all(item["diff"] == 0 for item in result[part]["items"].values()), part
    assert result["adapter_settings"] == {"rank": 4, "alpha": 8.0, "dropout": 0.0, "targets": ["q", "k", "v", "o"]}
    assert result["settings"]["val_positions"] == 40


def test_main_exit_code_and_json(runs, tmp_path, capsys):
    path = tmp_path / "out" / "result.json"
    assert main([str(runs[0]), "--device", "cpu", "--num-workers", "0", "--json", str(path)]) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["pass"] and "baseline" not in saved
    assert saved["eval"]["items"]["top1"]["reference"] == last_line(runs[0], "eval")["top1"]
    assert "판정: 통과" in capsys.readouterr().out


def test_swapped_adapter_fails(runs, tmp_path):
    """다른 seed의 어댑터로 바꿔치기하면 기록과 맞지 않는다."""
    swapped = tmp_path / "swapped"
    shutil.copytree(runs[0], swapped)
    shutil.copy(runs[1] / "adapter.pt", swapped / "adapter.pt")
    assert main([str(swapped), "--device", "cpu", "--num-workers", "0"]) == 1
    # baseline은 어댑터와 관계없으므로 그대로 통과한다
    result = reevaluate(swapped, "cpu", num_workers=0, baseline=True)
    assert not result["eval"]["pass"] and result["baseline"]["pass"]


def test_rejects_different_val_size(runs, tmp_path):
    changed = tmp_path / "changed"
    shutil.copytree(runs[0], changed)
    config = json.loads((changed / "config.json").read_text(encoding="utf-8"))
    config["val_positions"] += 1
    (changed / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="val 국면 수"):
        reevaluate(changed, "cpu", num_workers=0)


REFERENCE = {"imitation": 1.0, "top1": 0.5, "top3": 0.8, "regret": 0.01, "regret_top1": 0.002,
             "band_violation": 0.2, "strength": 0.004}


def test_compare_identical():
    result = compare(dict(REFERENCE), {"type": "eval", **REFERENCE})
    assert result["pass"] and set(result["items"]) == set(METRICS)


@pytest.mark.parametrize("name", ["top1", "top3"])
def test_compare_top_must_be_exact(name):
    metrics = dict(REFERENCE, **{name: REFERENCE[name] + 1e-12})
    result = compare(metrics, REFERENCE)
    assert not result["pass"] and not result["items"][name]["pass"]


@pytest.mark.parametrize("name", ["imitation", "regret", "regret_top1", "band_violation", "strength"])
@pytest.mark.parametrize("diff, passed", [(4.9e-5, True), (-4.9e-5, True), (5.1e-5, False), (-5.1e-5, False)])
def test_compare_tolerance(name, diff, passed):
    assert TOLERANCE == 5e-5
    metrics = dict(REFERENCE, **{name: REFERENCE[name] + diff})
    assert compare(metrics, REFERENCE)["pass"] is passed


def run_eval_job(tmp_path, *folders):
    import os
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "eval"
    env = {**os.environ, "SITE_ENV": str(tmp_path / "없음.env"), "EVAL_OUT": str(out), "EVAL_DEVICE": "cpu",
           "EVAL_NUM_WORKERS": "0", "EVAL_SKIP_INPUT_CHECK": "1"}
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", "gpu_train/eval.sbatch", *map(str, folders)], cwd=root,
                            capture_output=True, text=True, env=env)
    return result, out


def test_eval_job_passes_and_records(runs, tmp_path):
    result, out = run_eval_job(tmp_path, runs[0], runs[1])
    assert result.returncode == 0, result.stdout + result.stderr
    for run in runs:
        saved = json.loads((out / f"{run.name}.json").read_text(encoding="utf-8"))
        assert saved["pass"] and saved["baseline"]["pass"]
        assert "판정: 통과" in (out / f"{run.name}.txt").read_text(encoding="utf-8")
    (meta,) = out.glob("meta_local_*.txt")
    text = meta.read_text(encoding="utf-8")
    assert "git HEAD:" in text and "torch:" in text
    assert f"  {runs[0].name} 통과" in text and f"  {runs[1].name} 통과" in text


def test_eval_job_reports_failure_and_error_but_continues(runs, tmp_path):
    swapped = tmp_path / "swapped"
    shutil.copytree(runs[0], swapped)
    shutil.copy(runs[1] / "adapter.pt", swapped / "adapter.pt")
    missing = tmp_path / "missing"
    missing.mkdir()
    result, out = run_eval_job(tmp_path, swapped, missing, runs[0])
    assert result.returncode == 1
    (meta,) = out.glob("meta_local_*.txt")
    text = meta.read_text(encoding="utf-8")
    assert "  swapped 실패" in text
    assert "  missing 오류(종료 코드 1)" in text
    assert f"  {runs[0].name} 통과" in text  # 앞의 실패·오류 뒤에도 평가한다
