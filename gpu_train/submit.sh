#!/bin/bash
# 학습 하나를 sbatch로 제출한다. site.env의 Slurm 값으로 옵션을 붙이고, 빈 값은 옵션을 붙이지 않는다.
# 사용: bash gpu_train/submit.sh R1        (학습 ID는 python -m gpu_train.runs list)
#       SBATCH_DRY_RUN=1 bash gpu_train/submit.sh R1   (제출하지 않고 sbatch 명령만 출력)
# train.sbatch가 다음 작업을 이어서 제출할 때도 이 스크립트를 쓴다. 그때는 아래 환경변수가 붙는다.
#   SUBMIT_AFTER  앞 작업 ID (--dependency=afterany). 있으면 같은 이름 작업 확인을 건너뛴다
#   CHAIN_INDEX   새 작업의 사슬 번호 (첫 작업은 1)
#   PREV_START    앞 작업이 시작할 때의 체크포인트 step (전진 확인용)
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -eq 1 ] || { echo "사용: bash gpu_train/submit.sh 학습ID" >&2; exit 1; }
run=$1
[ -f gpu_train/site.env ] || { echo "gpu_train/site.env가 없다. site.env.example을 복사해 채운다" >&2; exit 1; }
source gpu_train/site.env
[ -n "${REPO_DIR:-}" ] || { echo "site.env의 REPO_DIR가 비어 있다" >&2; exit 1; }

# 실행할 수 없는 ID(없는 ID, α 미정)는 제출 전에 거부한다
python -m gpu_train.runs args "$run" > /dev/null

opts=(--job-name="$run" --output="runs/gpu/logs/%x_%j.out" --chdir="$REPO_DIR")
[ -n "${SBATCH_PARTITION:-}" ] && opts+=(--partition="$SBATCH_PARTITION")
[ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(--account="$SBATCH_ACCOUNT")
[ -n "${SBATCH_QOS:-}" ] && opts+=(--qos="$SBATCH_QOS")
[ -n "${SBATCH_GRES:-}" ] && opts+=(--gres="$SBATCH_GRES")
[ -n "${SBATCH_TIME:-}" ] && opts+=(--time="$SBATCH_TIME")
[ -n "${SBATCH_CPUS:-}" ] && opts+=(--cpus-per-task="$SBATCH_CPUS")
[ -n "${SBATCH_MEM:-}" ] && opts+=(--mem="$SBATCH_MEM")
# shellcheck disable=SC2206  # 공백으로 나눠 여러 옵션으로 넘긴다
[ -n "${SBATCH_EXTRA:-}" ] && opts+=($SBATCH_EXTRA)
if [ -n "${SUBMIT_AFTER:-}" ]; then
  opts+=(--dependency="afterany:$SUBMIT_AFTER")
fi
opts+=(--export="ALL,CHAIN_INDEX=${CHAIN_INDEX:-1},PREV_START=${PREV_START:-}")
cmd=(sbatch --parsable "${opts[@]}" gpu_train/train.sbatch "$run")

if [ "${SBATCH_DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi

# REPO_DIR가 이 저장소가 아니면 로그와 결과가 다른 곳에 쌓인다
if [ "$(cd "$REPO_DIR" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
  echo "site.env의 REPO_DIR($REPO_DIR)가 이 저장소($(pwd -P))와 다르다" >&2
  exit 1
fi

# 사람이 제출할 때만: 같은 이름의 작업이 이미 대기 중이거나 실행 중이면 제출하지 않는다
if [ -z "${SUBMIT_AFTER:-}" ]; then
  queued=$(squeue -h -u "${USER:-$(id -un)}" -n "$run" -o "%i %T")
  if [ -n "$queued" ]; then
    echo "이미 대기 중이거나 실행 중이다. 제출하지 않는다:" >&2
    echo "$queued" >&2
    exit 1
  fi
fi

mkdir -p runs/gpu/logs
jobid=$("${cmd[@]}")
jobid=${jobid%%;*}   # --parsable은 "작업ID;클러스터"로 줄 수 있다
echo "제출: $run 작업 $jobid (사슬 ${CHAIN_INDEX:-1}번, 로그 runs/gpu/logs/${run}_${jobid}.out)" >&2
echo "$jobid"
