#!/bin/bash
# C4 수 품질 탐색 작업(quality.sbatch)을 sbatch로 제출한다. GPU를 쓰지 않으므로 기본으로 --gres를 붙이지 않는다.
# 사용: bash gpu_train/submit_quality.sh base R1 R8 R2
#       SBATCH_DRY_RUN=1 bash gpu_train/submit_quality.sh base R1 R8 R2   (제출하지 않고 sbatch 명령만 출력)
# 환경변수: QUALITY_CPUS (기본 16, Stockfish 프로세스 수), QUALITY_MEM (기본 16G),
#           QUALITY_GRES (파티션이 GPU 요청을 요구하면 gpu:1 등), QUALITY_TIME (없으면 site.env의 SBATCH_TIME)
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "사용: bash gpu_train/submit_quality.sh 모델ID [...]" >&2; exit 1; }
site_env=${SITE_ENV:-gpu_train/site.env}
[ -f "$site_env" ] || { echo "$site_env가 없다. site.env.example을 복사해 채운다" >&2; exit 1; }
source "$site_env"
[ -n "${REPO_DIR:-}" ] || { echo "$site_env의 REPO_DIR가 비어 있다" >&2; exit 1; }

declare -A seen=()
for model in "$@"; do
  [[ "$model" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "모델 ID에 쓸 수 없는 문자가 있다: $model" >&2; exit 1; }
  [ -z "${seen[$model]:-}" ] || { echo "모델 ID $model이 겹친다" >&2; exit 1; }
  seen[$model]=1
done

opts=(--job-name=quality --output="runs/gpu/logs/%x_%j.out" --chdir="$REPO_DIR")
[ -n "${SBATCH_PARTITION:-}" ] && opts+=(--partition="$SBATCH_PARTITION")
[ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(--account="$SBATCH_ACCOUNT")
[ -n "${SBATCH_QOS:-}" ] && opts+=(--qos="$SBATCH_QOS")
[ -n "${QUALITY_GRES:-}" ] && opts+=(--gres="$QUALITY_GRES")
time_limit=${QUALITY_TIME:-${SBATCH_TIME:-}}
[ -n "$time_limit" ] && opts+=(--time="$time_limit")
opts+=(--cpus-per-task="${QUALITY_CPUS:-16}" --mem="${QUALITY_MEM:-16G}")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA:-}" ] && opts+=($SBATCH_EXTRA)
cmd=(sbatch --parsable "${opts[@]}" gpu_train/quality.sbatch "$@")

if [ "${SBATCH_DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi

if [ "$(cd "$REPO_DIR" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
  echo "$site_env의 REPO_DIR($REPO_DIR)가 이 저장소($(pwd -P))와 다르다" >&2
  exit 1
fi

mkdir -p runs/gpu/logs
jobid=$("${cmd[@]}")
jobid=${jobid%%;*}
echo "제출: C4 탐색 작업 $jobid (모델 $*, 로그 runs/gpu/logs/quality_${jobid}.out, 결과 runs/gpu/quality/)" >&2
echo "$jobid"
