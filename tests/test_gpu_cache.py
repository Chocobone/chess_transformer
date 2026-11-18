"""Task #14 2단계: 베이스 q 캐시 작업(cache.sbatch, submit_cache.sh)과 검증 도구(gpu_train/check_cache.py)."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from gpu_train import check_cache
from scripts.cache_base_q import build_cache
from src.data.dataset import cache_path
from src.model.transformer import ActionValueTransformer, TransformerConfig, load_base, save_base
from tests.test_train import random_records

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["gpu_train/cache.sbatch", "gpu_train/submit_cache.sh"]
TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)


def run_script(*args, env=None):
    return subprocess.run(["bash", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})


@pytest.fixture
def workspace(tmp_path):
    """소형 무작위 베이스와 기사 tal의 데이터·캐시."""
    data_dir = tmp_path / "data" / "tal"
    data_dir.mkdir(parents=True)
    for split, count, seed in (("train", 60, 1), ("val", 20, 2)):
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for record in random_records(count, seed):
                f.write(json.dumps(record) + "\n")
    torch.manual_seed(0)
    base = tmp_path / "tiny.pt"
    save_base(ActionValueTransformer(TINY), base)
    build_cache(load_base(base).eval(), data_dir, base, batch_moves=200)
    return tmp_path, data_dir, base


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_and_use_lf(script):
    assert run_script("-n", script).returncode == 0
    assert b"\r\n" not in (ROOT / script).read_bytes()


@pytest.fixture
def site_env(tmp_path):
    path = tmp_path / "site.env"
    path.write_text("REPO_DIR=/srv/repo\nSBATCH_PARTITION=gpu\nSBATCH_GRES=gpu:1\nSBATCH_TIME=08:00:00\n"
                    "SBATCH_CPUS=4\nSBATCH_MEM=32G\nSBATCH_ACCOUNT=\nSBATCH_EXTRA=--nodelist=moana-u1\n", encoding="utf-8")
    return path


def test_submit_cache_dry_run_submits_one_job_per_player(site_env):
    result = run_script("gpu_train/submit_cache.sh", "tal", "karpov", "smyslov",
                        env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 3
    for line, player in zip(lines, ("tal", "karpov", "smyslov")):
        for option in (f"--job-name=cache_{player}", "--chdir=/srv/repo", "--partition=gpu", "--gres=gpu:1",
                       "--time=08:00:00", "--cpus-per-task=4", "--mem=32G", "--nodelist=moana-u1"):
            assert option in line
        assert line.rstrip().endswith(f"gpu_train/cache.sbatch {player}")
        assert "--account" not in line


@pytest.mark.parametrize("players", [["morphy"], ["tal", "tal"], []])
def test_submit_cache_rejects_bad_players(site_env, players):
    result = run_script("gpu_train/submit_cache.sh", *players, env={"SITE_ENV": str(site_env), "SBATCH_DRY_RUN": "1"})
    assert result.returncode == 1
    assert result.stdout == ""


def test_check_cache_passes_and_recomputes_exactly(workspace):
    _, data_dir, base = workspace
    assert check_cache.check_structure(data_dir, base) == []
    result = check_cache.recompute(data_dir, base, count=10, device="cpu")
    assert set(result) == {"train", "val"}
    # 같은 장비라도 묶음 크기(캐시 200, 다시 계산 256)가 다르면 float32 끝자리가 다를 수 있다
    assert all(r["positions"] == 10 and r["max_abs_diff"] < 1e-6 for r in result.values())
    args = ["--player", "tal", "--base", str(base), "--data-dir", str(data_dir), "--recompute", "5"]
    assert check_cache.main(args) == 0


def rewrite_cache(data_dir, base, **changes):
    path = cache_path(data_dir, base)
    arrays = dict(np.load(path))
    for key, change in changes.items():
        arrays[key] = change(arrays[key])
    np.savez(path, **arrays)


def test_check_cache_detects_wrong_q(workspace):
    _, data_dir, base = workspace

    def shift(q):
        return np.clip(q + 0.01, 0, 1).astype(q.dtype)

    rewrite_cache(data_dir, base, train_q=shift, val_q=shift)
    assert check_cache.check_structure(data_dir, base) == []  # 구조는 맞다
    args = ["--player", "tal", "--base", str(base), "--data-dir", str(data_dir), "--recompute", "20"]
    assert check_cache.main(args) == 1
    assert check_cache.main([*args, "--tolerance", "0.05"]) == 0


def test_check_cache_detects_structure_problems(workspace):
    _, data_dir, base = workspace
    rewrite_cache(data_dir, base, val_offsets=lambda o: o[:-1])
    assert any("val: 국면 수" in p for p in check_cache.check_structure(data_dir, base))

    def swap(actions):
        actions = actions.copy()
        actions[[0, 1]] = actions[[1, 0]]
        return actions

    rewrite_cache(data_dir, base, train_actions=swap)
    problems = check_cache.check_structure(data_dir, base)
    assert any(p.startswith("train: 합법수 수·순서") for p in problems)

    path = cache_path(data_dir, base)
    path.write_bytes(path.read_bytes()[:100])  # 저장 중에 끊긴 파일
    assert check_cache.check_structure(data_dir, base)[0].startswith("캐시를 읽을 수 없다")
    path.unlink()
    assert check_cache.check_structure(data_dir, base)[0].startswith("캐시가 없다")


def assert_same_cache(cache: dict, reference: dict) -> None:
    """국면·합법수는 같고, q는 묶음 크기에 따른 float32 차이 안에서 같다 (작업은 묶음 256, 기준은 200)."""
    assert cache.keys() == reference.keys()
    for key in reference:
        if key.endswith("_q"):
            assert np.abs(cache[key] - reference[key]).max() < 1e-6
        else:
            assert np.array_equal(cache[key], reference[key])


def test_cache_job_end_to_end(workspace):
    """Slurm 밖에서 작업 본문을 끝까지 돈다: 만들기 → 건너뛰기 → 깨진 캐시 다시 만들기."""
    root, data_dir, base = workspace
    cache = cache_path(data_dir, base)
    reference = dict(np.load(cache))
    cache.unlink()
    env = {"SITE_ENV": str(root / "none.env"), "CACHE_BASE": str(base), "CACHE_DATA_ROOT": str(root / "data"),
           "CACHE_LOG_ROOT": str(root / "logs"), "CACHE_SKIP_INPUT_CHECK": "1", "CACHE_DEVICE": "cpu"}

    first = run_script("gpu_train/cache.sbatch", "tal", env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "tal 완료" in first.stdout
    rebuilt = dict(np.load(cache))
    assert_same_cache(rebuilt, reference)
    meta = list((root / "logs" / "meta").glob("cache_tal_local_*.txt"))
    assert meta and "입력 SHA-256" in meta[0].read_text(encoding="utf-8")

    second = run_script("gpu_train/cache.sbatch", "tal", env=env)
    assert second.returncode == 0 and "건너뛴다" in second.stdout

    cache.write_bytes(cache.read_bytes()[:100])
    third = run_script("gpu_train/cache.sbatch", "tal", env=env)
    assert third.returncode == 0, third.stdout + third.stderr
    assert "검증 실패. 지우고 다시 만든다" in third.stdout and "tal 완료" in third.stdout
    assert_same_cache(dict(np.load(cache)), reference)


def test_cache_job_checks_manifest(tmp_path):
    """매니페스트에 베이스와 기사 입력이 없으면 종료 코드 2 (지금 매니페스트에는 270M.pt가 없다)."""
    result = run_script("gpu_train/cache.sbatch", "tal",
                        env={"SITE_ENV": str(tmp_path / "none.env"), "CACHE_LOG_ROOT": str(tmp_path)})
    assert result.returncode == 2
    assert "inputs.sha256에" in result.stderr


def test_cache_job_rejects_unknown_player():
    assert run_script("gpu_train/cache.sbatch", "morphy").returncode == 1
