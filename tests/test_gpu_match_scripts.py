"""Task #13 4단계: 대국·C4 서버 실행 틀 (문법, 제출 명령, 입력 거부)."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["gpu_train/match.sbatch", "gpu_train/submit_match.sh", "gpu_train/quality.sbatch",
           "gpu_train/submit_quality.sh", "gpu_train/install_stockfish.sh", "gpu_train/build_stockfish.sh",
           "gpu_train/setup_env.sh"]


def run_script(*args, env=None):
    return subprocess.run(["bash", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})


@pytest.fixture
def site_env(tmp_path):
    path = tmp_path / "site.env"
    path.write_text("REPO_DIR=/srv/repo\nSBATCH_PARTITION=gpu\nSBATCH_GRES=gpu:1\nSBATCH_TIME=10:00:00\n"
                    "SBATCH_CPUS=4\nSBATCH_MEM=32G\nSBATCH_ACCOUNT=\nSBATCH_EXTRA=--nodelist=moana-u1\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse(script):
    assert run_script("-n", script).returncode == 0


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_use_lf(script):
    assert b"\r\n" not in (ROOT / script).read_bytes()


def test_submit_match_dry_run(site_env):
    specs = ["cal1_n100:base:sf:100:0-9", "C1_R1:R1:base:-:0-199"]
    result = run_script("gpu_train/submit_match.sh", *specs, env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    assert command[:2] == ["sbatch", "--parsable"]
    for option in ("--job-name=match", "--output=runs/gpu/logs/%x_%j.out", "--chdir=/srv/repo", "--partition=gpu",
                   "--gres=gpu:1", "--time=10:00:00", "--cpus-per-task=4", "--mem=32G", "--nodelist=moana-u1"):
        assert option in command
    assert command[-3:] == ["gpu_train/match.sbatch", *specs]
    more = run_script("gpu_train/submit_match.sh", specs[0],
                      env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1", "MATCH_CPUS": "12"})
    assert "--cpus-per-task=12" in more.stdout.split()


@pytest.mark.parametrize("spec, message", [
    ("cal:base:sf:100", "표기가 틀렸다"),
    ("cal:base:sf:100:0-9:x", "표기가 틀렸다"),
    ("cal:base:sf:-:0-9", "노드 수가 필요하다"),
    ("C1:R1:base:300:0-9", "노드 수는 -다"),
    ("C1:R1:R2:-:0-9", "base 또는 sf"),
    ("C1:base:base:-:0-9", "같다"),
    ("C1:R1:base:-:0~9", "오프닝 범위"),
    ("C1/x:R1:base:-:0-9", "쓸 수 없는 문자"),
])
def test_submit_match_rejects_bad_specs(site_env, spec, message):
    result = run_script("gpu_train/submit_match.sh", spec, env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 1 and message in result.stderr


def test_submit_match_rejects_duplicate_names(site_env):
    result = run_script("gpu_train/submit_match.sh", "a:base:sf:100:0-9", "a:base:sf:300:0-9",
                        env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 1 and "겹친다" in result.stderr


def test_submit_quality_dry_run(site_env):
    env = {"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"}
    result = run_script("gpu_train/submit_quality.sh", "base", "R1", "R8", "R2", env=env)
    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    for option in ("--job-name=quality", "--chdir=/srv/repo", "--cpus-per-task=16", "--mem=16G", "--time=10:00:00"):
        assert option in command
    assert not any(option.startswith("--gres") for option in command)  # GPU를 쓰지 않는다
    assert command[-5:] == ["gpu_train/quality.sbatch", "base", "R1", "R8", "R2"]
    gpu = run_script("gpu_train/submit_quality.sh", "base", env={**env, "QUALITY_GRES": "gpu:1", "QUALITY_CPUS": "8"})
    assert "--gres=gpu:1" in gpu.stdout.split() and "--cpus-per-task=8" in gpu.stdout.split()
    bad = run_script("gpu_train/submit_quality.sh", "base", "base", env=env)
    assert bad.returncode == 1 and "겹친다" in bad.stderr


def test_quality_sbatch_rejects_missing_dump(site_env, tmp_path):
    if not (ROOT / "runs/tools/stockfish/stockfish").exists():
        pytest.skip("Stockfish 미설치 (설치 확인이 먼저 실패한다)")
    result = run_script("gpu_train/quality.sbatch", "base",
                        env={"SITE_ENV": str(tmp_path / "none.env"), "QUALITY_DUMP": str(tmp_path / "nodump")})
    assert result.returncode == 2 and "positions.npz가 없다" in result.stderr


def test_install_stockfish_rejects_wrong_file(tmp_path):
    """빈 저장소 사본에서: 원본 파일이 없으면 거부, SHA-256이 다르면 거부 (CPU가 AVX2를 지원할 때)."""
    import shutil
    (tmp_path / "gpu_train").mkdir()
    shutil.copy(ROOT / "gpu_train/install_stockfish.sh", tmp_path / "gpu_train")
    flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    if " avx2" not in flags:
        pytest.skip("AVX2가 없는 CPU")
    script = str(tmp_path / "gpu_train/install_stockfish.sh")
    missing = subprocess.run(["bash", script], capture_output=True, text=True)
    assert missing.returncode != 0 and "가 없다" in missing.stderr
    fake = tmp_path / "fake"
    fake.write_bytes(b"not stockfish")
    wrong = subprocess.run(["bash", script, str(fake)], capture_output=True, text=True)
    assert wrong.returncode != 0 and "FAILED" in wrong.stdout + wrong.stderr
    assert not (tmp_path / "runs/tools/stockfish/stockfish").exists()
