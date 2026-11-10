#!/bin/bash
# 학습 폴더 재평가(eval.sbatch)를 sbatch로 제출한다. site.env의 Slurm 값으로 옵션을 붙이고, 빈 값은 옵션을 붙이지 않는다.
# 사용: bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8
#       SBATCH_DRY_RUN=1 bash gpu_train/submit_eval.sh runs/gpu/R1   (제출하지 않고 sbatch 명령만 출력)
# 옵션을 붙이는 방식은 submit.sh와 같다. 평가는 짧으므로 이어서 제출(--dependency)은 하지 않는다.
# SITE_ENV로 설정 파일 경로를 바꿀 수 있다 (기본 gpu_train/site.env).
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "사용: bash gpu_train/submit_eval.sh 학습폴더 [학습폴더 ...]" >&2; exit 1; }
site_env=${SITE_ENV:-gpu_train/site.env}
[ -f "$site_env" ] || { echo "$site_env가 없다. site.env.example을 복사해 채운다" >&2; exit 1; }
source "$site_env"
[ -n "${REPO_DIR:-}" ] || { echo "$site_env의 REPO_DIR가 비어 있다" >&2; exit 1; }

# 평가할 수 없는 폴더는 제출 전에 거부한다. 결과 파일 이름(폴더 이름)이 겹쳐도 거부한다
declare -A seen=()
for dir in "$@"; do
  for f in config.json log.jsonl adapter.pt; do
    [ -f "$dir/$f" ] || { echo "$dir/$f가 없다. 제출하지 않는다" >&2; exit 1; }
  done
  name=$(basename "$dir")
  [ -z "${seen[$name]:-}" ] || { echo "결과 이름 $name이 겹친다: ${seen[$name]}, $dir" >&2; exit 1; }
  seen[$name]=$dir
done

opts=(--job-name=eval --output="runs/gpu/logs/%x_%j.out" --chdir="$REPO_DIR")
[ -n "${SBATCH_PARTITION:-}" ] && opts+=(--partition="$SBATCH_PARTITION")
[ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(--account="$SBATCH_ACCOUNT")
[ -n "${SBATCH_QOS:-}" ] && opts+=(--qos="$SBATCH_QOS")
[ -n "${SBATCH_GRES:-}" ] && opts+=(--gres="$SBATCH_GRES")
[ -n "${SBATCH_TIME:-}" ] && opts+=(--time="$SBATCH_TIME")
[ -n "${SBATCH_CPUS:-}" ] && opts+=(--cpus-per-task="$SBATCH_CPUS")
[ -n "${SBATCH_MEM:-}" ] && opts+=(--mem="$SBATCH_MEM")
# shellcheck disable=SC2206  # 공백으로 나눠 여러 옵션으로 넘긴다
[ -n "${SBATCH_EXTRA:-}" ] && opts+=($SBATCH_EXTRA)
cmd=(sbatch --parsable "${opts[@]}" gpu_train/eval.sbatch "$@")

if [ "${SBATCH_DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi

# REPO_DIR가 이 저장소가 아니면 로그와 결과가 다른 곳에 쌓인다
if [ "$(cd "$REPO_DIR" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
  echo "$site_env의 REPO_DIR($REPO_DIR)가 이 저장소($(pwd -P))와 다르다" >&2
  exit 1
fi

mkdir -p runs/gpu/logs
jobid=$("${cmd[@]}")
jobid=${jobid%%;*}   # --parsable은 "작업ID;클러스터"로 줄 수 있다
echo "제출: 재평가 작업 $jobid (대상 $#개, 로그 runs/gpu/logs/eval_${jobid}.out, 결과 runs/gpu/eval/)" >&2
echo "$jobid"
