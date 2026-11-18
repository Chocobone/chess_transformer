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


# ── Task #14: 270M 정의와 입력 목록 ──────────────────────────
# 270M 정의를 더하기 전(2026-11-18, f715d0f)의 출력. 136M 학습 명령은 한 글자도 바뀌면 안 된다
GOLDEN_136M = {
    "R1": "python -u -m src.train.train --player tal --out runs/gpu/R1 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --resume",
    "R2": "python -u -m src.train.train --player smyslov --out runs/gpu/R2 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --resume",
    "R8": "python -u -m src.train.train --player karpov --out runs/gpu/R8 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --resume",
    "R1s0": "python -u -m src.train.train --player tal --out runs/gpu/R1s0 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 0 --resume",
    "R2s0": "python -u -m src.train.train --player smyslov --out runs/gpu/R2s0 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 0 --resume",
    "R8s0": "python -u -m src.train.train --player karpov --out runs/gpu/R8s0 --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 0 --resume",
    "smoke": "python -u -m src.train.train --player tal --out runs/gpu/smoke --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --max-steps 50 --eval-every 25 --eval-positions 500 --resume",
}
PAIRS_270M = {"R1_270M": "R1", "R2_270M": "R2", "R8_270M": "R8", "smoke_270M": "smoke"}


def test_136m_commands_are_unchanged():
    ready = [run_id for run_id, run in runs.RUNS.items() if run.alpha is not None and run.base == runs.BASE_136M]
    assert ready == list(GOLDEN_136M)
    for run_id, command in GOLDEN_136M.items():
        assert " ".join(runs.train_args(run_id)) == command


@pytest.mark.parametrize("run_id", PAIRS_270M)
def test_270m_runs_differ_from_pair_only_in_base_and_move_limit(run_id):
    big, pair = vars(parse(run_id)), vars(parse(PAIRS_270M[run_id]))
    assert big["base"] == "base_model/270M.pt"
    assert big["max_moves_per_forward"] == runs.MAX_MOVES_270M == 48
    assert big["out"] == f"runs/gpu/{run_id}"
    changed = {key for key in pair if big[key] != pair[key]}
    assert changed == {"base", "max_moves_per_forward", "out"}


def test_270m_runs_are_listed_with_base(capsys):
    assert runs.main(["list"]) == 0
    lines = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()[1:]}
    for run_id in PAIRS_270M:
        assert lines[run_id][5:7] == ["가능", "270M"]
    assert lines["R1"][5:7] == ["가능", "136M"]


@pytest.fixture
def manifest_repo(tmp_path):
    """make_manifest.sh가 읽을 가짜 입력 17개가 있는 임시 저장소."""
    import shutil
    import subprocess
    (tmp_path / "gpu_train").mkdir()
    shutil.copy(ROOT / "gpu_train/make_manifest.sh", tmp_path / "gpu_train")
    for name in ("136M.pt", "270M.pt"):
        (tmp_path / "base_model").mkdir(exist_ok=True)
        (tmp_path / "base_model" / name).write_text(name, encoding="utf-8")
    for player in ("tal", "karpov", "smyslov"):
        folder = tmp_path / "data/processed" / player
        folder.mkdir(parents=True)
        for name in ("train.jsonl", "val.jsonl", "stats.json", "base_q_136M.npz", "base_q_270M.npz"):
            (folder / name).write_text(player + name, encoding="utf-8")
    git = ["git", "-c", "user.email=test@example.com", "-c", "user.name=test"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    return tmp_path


def manifest_paths(path: Path) -> list[str]:
    return [line.split(maxsplit=1)[1] for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def test_make_manifest_keeps_136m_order_and_appends_270m(manifest_repo):
    import subprocess
    current = manifest_paths(ROOT / "gpu_train/inputs.sha256")[:13]
    script = manifest_repo / "gpu_train/make_manifest.sh"
    out = manifest_repo / "gpu_train/inputs.sha256"

    assert subprocess.run(["bash", str(script), "--base-only"], capture_output=True).returncode == 0
    base_only = manifest_paths(out)
    assert base_only[:13] == current
    assert base_only[13:] == ["base_model/270M.pt"]

    assert subprocess.run(["bash", str(script)], capture_output=True).returncode == 0
    full = manifest_paths(out)
    assert full[:14] == base_only
    assert full[14:] == [f"data/processed/{p}/base_q_270M.npz" for p in ("tal", "karpov", "smyslov")]
    assert out.read_text(encoding="utf-8").startswith("# 입력 파일 17개")

    assert subprocess.run(["bash", str(script), "--all"], capture_output=True).returncode == 1
