import json
from pathlib import Path

import pytest

from gpu_train import runs
from src.train.train import build_parser

READY = {"R1": "tal", "R2": "smyslov", "R8": "karpov"}
PENDING = ("R3", "R4", "R5", "R6", "R7", "R9", "R10")
# 학습 결과가 아니라 데이터·실행에 따라 정해지는 config.json 키
RUN_SPECIFIC = {"out", "seed", "base_config", "amp", "total_steps", "train_positions", "val_positions",
                "trainable_parameters", "device"}


def parse(run_id: str):
    command = runs.train_args(run_id)
    assert command[:4] == ["python", "-u", "-m", "src.train.train"]
    return build_parser().parse_args(command[4:])


@pytest.mark.parametrize("run_id", READY)
def test_ready_runs_follow_criteria(run_id):
    """기준 문서 8.1절의 공통 설정, seed 1."""
    args = parse(run_id)
    assert args.player == READY[run_id]
    assert args.base == "base_model/136M.pt"
    assert (args.rank, args.lora_alpha, args.lr, args.epochs, args.batch_positions) == (16, 32, 1e-4, 3, 32)
    assert (args.temperature, args.tau, args.lam, args.alpha, args.gamma) == (0.02, 0.02, 1.0, 0.0, 0.0)
    assert (args.seed, args.resume, args.max_steps, args.eval_positions) == (1, True, 0, 0)
    assert args.out == f"runs/gpu/{run_id}"


@pytest.mark.parametrize("run_id", PENDING)
def test_pending_runs_are_rejected(run_id):
    with pytest.raises(ValueError, match="α가 아직 없다"):
        runs.train_args(run_id)


def test_ablation_run_has_no_strength_term():
    assert runs.RUNS["R7"].lam == 0.0
    assert all(run.lam == 1.0 for run_id, run in runs.RUNS.items() if run_id != "R7")


def test_output_folders_are_distinct():
    folders = [runs.out_dir(run_id) for run_id in runs.RUNS]
    assert len(set(folders)) == len(folders)


def test_smoke_is_short_and_otherwise_like_r1():
    smoke, r1 = vars(parse("smoke")), vars(parse("R1"))
    assert (smoke["max_steps"], smoke["eval_every"], smoke["eval_positions"]) == (50, 25, 500)
    changed = {key for key in r1 if smoke[key] != r1[key]}
    assert changed == {"out", "max_steps", "eval_every", "eval_positions"}


def test_unknown_run_id():
    with pytest.raises(KeyError):
        runs.get_run("R11")
    assert runs.main(["args", "R11"]) == 1
    assert runs.main(["args", "R3"]) == 1


@pytest.mark.parametrize("run_id", READY)
def test_matches_existing_adapter_config(run_id):
    """기존 seed 0 어댑터와 seed·출력 폴더 말고는 학습 설정이 같다 (runs/는 git 제외라 없으면 건너뛴다)."""
    path = Path(f"runs/{READY[run_id]}_136M/config.json")
    if not path.exists():
        pytest.skip(f"{path} 없음")
    old = json.loads(path.read_text())
    new = vars(parse(run_id))
    differing = {key for key in old if key not in RUN_SPECIFIC and old[key] != new.get(key)}
    assert differing == set()
    assert (old["seed"], new["seed"]) == (0, 1)


SEED0 = {"R1s0": "R1", "R2s0": "R2", "R8s0": "R8"}


@pytest.mark.parametrize("run_id", SEED0)
def test_seed0_runs_differ_from_pair_only_in_seed(run_id):
    """잡음 바닥 쌍: 대응하는 R와 seed·출력 폴더만 다르다 (Task #11)."""
    seed0, pair = vars(parse(run_id)), vars(parse(SEED0[run_id]))
    changed = {key for key in pair if seed0[key] != pair[key]}
    assert changed == {"seed", "out"}
    assert (seed0["seed"], pair["seed"]) == (0, 1)
    assert seed0["out"] == f"runs/gpu/{run_id}"


def test_seed0_runs_are_listed_as_ready(capsys):
    assert runs.main(["list"]) == 0
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    for run_id in SEED0:
        assert lines[run_id].split()[5] == "가능"


ROOT = Path(__file__).resolve().parent.parent


def run_script(*args, env=None):
    import os
    import subprocess
    return subprocess.run(["bash", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


@pytest.mark.parametrize("script", ["gpu_train/eval.sbatch", "gpu_train/submit_eval.sh"])
def test_eval_scripts_parse(script):
    assert run_script("-n", script).returncode == 0


@pytest.fixture
def site_env(tmp_path):
    path = tmp_path / "site.env"
    path.write_text("REPO_DIR=/srv/repo\nSBATCH_PARTITION=gpu\nSBATCH_GRES=gpu:1\nSBATCH_TIME=02:00:00\n"
                    "SBATCH_CPUS=4\nSBATCH_MEM=32G\nSBATCH_ACCOUNT=\nSBATCH_EXTRA=\n", encoding="utf-8")
    return path


def fake_run(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for name in ("config.json", "log.jsonl", "adapter.pt"):
        (folder / name).write_text("{}", encoding="utf-8")
    return folder


def test_submit_eval_dry_run(site_env, tmp_path):
    first, second = fake_run(tmp_path / "a" / "R1"), fake_run(tmp_path / "a" / "R2")
    result = run_script("gpu_train/submit_eval.sh", str(first), str(second),
                        env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    assert command[:2] == ["sbatch", "--parsable"]
    for option in ("--job-name=eval", "--output=runs/gpu/logs/%x_%j.out", "--chdir=/srv/repo", "--partition=gpu",
                   "--gres=gpu:1", "--time=02:00:00", "--cpus-per-task=4", "--mem=32G"):
        assert option in command
    assert not any(option.startswith(("--account", "--dependency")) for option in command)
    assert command[-3:] == ["gpu_train/eval.sbatch", str(first), str(second)]


def test_submit_eval_rejects_incomplete_or_clashing_folders(site_env, tmp_path):
    env = {"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"}
    broken = fake_run(tmp_path / "broken")
    (broken / "adapter.pt").unlink()
    result = run_script("gpu_train/submit_eval.sh", str(broken), env=env)
    assert result.returncode == 1 and "adapter.pt가 없다" in result.stderr
    first, second = fake_run(tmp_path / "x" / "R1"), fake_run(tmp_path / "y" / "R1")
    result = run_script("gpu_train/submit_eval.sh", str(first), str(second), env=env)
    assert result.returncode == 1 and "겹친다" in result.stderr
