#!/bin/bash
# 짝 대국 작업(match.sbatch)을 sbatch로 제출한다. site.env의 Slurm 값으로 옵션을 붙이고, 빈 값은 옵션을 붙이지 않는다.
# 사용: bash gpu_train/submit_match.sh 이름:모델:상대:노드수:오프닝 [...]
#   예비 보정: bash gpu_train/submit_match.sh cal1_n100:base:sf:100:0-9 cal1_n300:base:sf:300:0-9 ...
#   SBATCH_DRY_RUN=1 bash gpu_train/submit_match.sh ...   (제출하지 않고 sbatch 명령만 출력)
# 계산 노드를 고정하려면 site.env의 SBATCH_EXTRA에 --nodelist=노드 이름을 넣는다 (Task #12·#13: moana-u1).
# Stockfish 프로세스 수는 할당 CPU − 1이다. CPU 수는 MATCH_CPUS(없으면 site.env의 SBATCH_CPUS)로 정한다.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "사용: bash gpu_train/submit_match.sh 이름:모델:상대:노드수:오프닝 [...]" >&2; exit 1; }
site_env=${SITE_ENV:-gpu_train/site.env}
[ -f "$site_env" ] || { echo "$site_env가 없다. site.env.example을 복사해 채운다" >&2; exit 1; }
source "$site_env"
[ -n "${REPO_DIR:-}" ] || { echo "$site_env의 REPO_DIR가 비어 있다" >&2; exit 1; }

# 묶음 표기를 제출 전에 확인한다. 이름이 겹치거나, 어댑터가 없거나, 노드 수가 상대와 맞지 않으면 거부한다
declare -A seen=()
needs_sf=0
for spec in "$@"; do
  IFS=: read -r name model opponent nodes openings extra <<< "$spec"
  [ -n "$name" ] && [ -n "$model" ] && [ -n "$opponent" ] && [ -n "$nodes" ] && [ -n "$openings" ] && [ -z "${extra:-}" ] \
    || { echo "묶음 표기가 틀렸다: $spec (이름:모델:상대:노드수:오프닝)" >&2; exit 1; }
  [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "이름에 쓸 수 없는 문자가 있다: $name" >&2; exit 1; }
  [ -z "${seen[$name]:-}" ] || { echo "이름 $name이 겹친다" >&2; exit 1; }
  seen[$name]=1
  [[ "$openings" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || { echo "오프닝 범위가 틀렸다: $openings" >&2; exit 1; }
  case "$opponent" in
    sf)   [[ "$nodes" =~ ^[0-9]+$ ]] || { echo "상대가 sf이면 노드 수가 필요하다: $spec" >&2; exit 1; }; needs_sf=1 ;;
    base) [ "$nodes" = "-" ] || { echo "상대가 base이면 노드 수는 -다: $spec" >&2; exit 1; } ;;
    *)    echo "상대는 base 또는 sf다: $spec" >&2; exit 1 ;;
  esac
  [ "$model" != "$opponent" ] || { echo "평가 대상과 상대가 같다: $spec" >&2; exit 1; }
  if [ "$model" != base ] && [ "${SBATCH_DRY_RUN:-0}" != 1 ]; then
    [ -f "runs/gpu/$model/adapter.pt" ] || { echo "runs/gpu/$model/adapter.pt가 없다" >&2; exit 1; }
  fi
done
if [ $needs_sf = 1 ] && [ "${SBATCH_DRY_RUN:-0}" != 1 ]; then
  [ -x runs/tools/stockfish/stockfish ] || { echo "Stockfish가 없다. bash gpu_train/install_stockfish.sh" >&2; exit 1; }
fi

cpus=${MATCH_CPUS:-${SBATCH_CPUS:-}}
opts=(--job-name=match --output="runs/gpu/logs/%x_%j.out" --chdir="$REPO_DIR")
[ -n "${SBATCH_PARTITION:-}" ] && opts+=(--partition="$SBATCH_PARTITION")
[ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(--account="$SBATCH_ACCOUNT")
[ -n "${SBATCH_QOS:-}" ] && opts+=(--qos="$SBATCH_QOS")
[ -n "${SBATCH_GRES:-}" ] && opts+=(--gres="$SBATCH_GRES")
[ -n "${SBATCH_TIME:-}" ] && opts+=(--time="$SBATCH_TIME")
[ -n "$cpus" ] && opts+=(--cpus-per-task="$cpus")
[ -n "${SBATCH_MEM:-}" ] && opts+=(--mem="$SBATCH_MEM")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA:-}" ] && opts+=($SBATCH_EXTRA)
cmd=(sbatch --parsable "${opts[@]}" gpu_train/match.sbatch "$@")

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
echo "제출: 대국 작업 $jobid (묶음 $#개, 로그 runs/gpu/logs/match_${jobid}.out, 결과 runs/gpu/match/)" >&2
echo "$jobid"
