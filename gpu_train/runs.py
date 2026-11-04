"""학습 정의 R1~R10과 smoke. 학습 ID를 `src.train.train` 명령으로 바꾼다.

설정은 기준 문서 `mydocs/tech/evaluation_criteria.md` 8.1절을 따른다.
작업 스크립트(`train.sbatch`)는 ID만 받고 인자는 모두 여기서 만든다.

    python -m gpu_train.runs list
    python -m gpu_train.runs args R1
    python -m gpu_train.runs out R1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

OUT_ROOT = "runs/gpu"

# 8.1절의 공통 설정. train.py 기본값이 바뀌어도 학습이 달라지지 않도록 모두 적는다.
COMMON: tuple[tuple[str, str], ...] = (
    ("--base", "base_model/136M.pt"),
    ("--rank", "16"),
    ("--lora-alpha", "32"),
    ("--lr", "1e-4"),
    ("--epochs", "3"),
    ("--batch-positions", "32"),
    ("--temperature", "0.02"),
    ("--tau", "0.02"),
    ("--gamma", "0"),
    ("--eval-every", "1000"),
    ("--checkpoint-every", "200"),
    ("--num-workers", "2"),
)


@dataclass(frozen=True)
class Run:
    run_id: str
    player: str
    alpha: float | None  # None: 특성 구현 전이라 α가 정해지지 않았다
    lam: float = 1.0
    seed: int = 1
    role: str = ""
    extra: tuple[str, ...] = ()  # COMMON 뒤에 붙는다. 같은 옵션이면 뒤쪽 값이 쓰인다


RUNS: dict[str, Run] = {run.run_id: run for run in (
    Run("R1", "tal", 0.0, role="α = 0. 잡음 바닥, D 기준선"),
    Run("R2", "smyslov", 0.0, role="α = 0. 잡음 바닥, D 기준선"),
    Run("R8", "karpov", 0.0, role="α = 0. 잡음 바닥, D 기준선"),
    Run("R3", "tal", None, role="α = a₁"),
    Run("R4", "tal", None, role="α = a₂ (주 결과)"),
    Run("R5", "smyslov", None, role="α = a₁"),
    Run("R6", "smyslov", None, role="α = a₂ (주 결과)"),
    Run("R9", "karpov", None, role="α = a₁"),
    Run("R10", "karpov", None, role="α = a₂ (주 결과)"),
    Run("R7", "tal", None, lam=0.0, role="α = a₂, λ = 0. 기력 항 ablation"),
    Run("smoke", "tal", 0.0, role="서버 확인용 50 갱신",
        extra=("--max-steps", "50", "--eval-every", "25", "--eval-positions", "500")),
)}


def get_run(run_id: str) -> Run:
    if run_id not in RUNS:
        raise KeyError(f"알 수 없는 학습 ID {run_id!r}. 가능한 값: {', '.join(RUNS)}")
    return RUNS[run_id]


def out_dir(run_id: str) -> str:
    return f"{OUT_ROOT}/{get_run(run_id).run_id}"


def train_args(run_id: str) -> list[str]:
    """`python -u -m src.train.train ...` 명령. 항상 --resume을 붙인다."""
    run = get_run(run_id)
    if run.alpha is None:
        raise ValueError(f"{run_id}의 α가 아직 없다. 특성 구현 이슈에서 α를 정한 뒤 gpu_train/runs.py에 적는다")
    args = ["python", "-u", "-m", "src.train.train", "--player", run.player, "--out", out_dir(run_id)]
    for option, value in COMMON:
        args += [option, value]
    args += ["--alpha", repr(run.alpha), "--lam", repr(run.lam), "--seed", str(run.seed), *run.extra, "--resume"]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU 서버 학습 정의")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="학습 ID 목록")
    for name in ("args", "out"):
        sub.add_parser(name).add_argument("run_id")
    ns = parser.parse_args(argv)

    try:
        if ns.command == "list":
            print(f"{'ID':<6} {'기사':<8} {'α':<6} {'λ':<4} {'seed':<4} {'실행':<4} 역할")
            for run in RUNS.values():
                alpha = "미정" if run.alpha is None else f"{run.alpha:g}"
                ready = "가능" if run.alpha is not None else "불가"
                print(f"{run.run_id:<6} {run.player:<8} {alpha:<6} {run.lam:<4g} {run.seed:<4} {ready:<4} {run.role}")
        elif ns.command == "args":
            print(" ".join(train_args(ns.run_id)))
        else:
            print(out_dir(ns.run_id))
    except (KeyError, ValueError) as error:
        print(f"오류: {error.args[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
