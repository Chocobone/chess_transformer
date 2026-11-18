#!/bin/bash
#SBATCH --output=runs/gpu/logs/%x_%j.out
# 기사 하나의 베이스 q 캐시(base_q_270M.npz)를 만드는 작업 본문. 제출은 gpu_train/submit_cache.sh가 한다 (Task #14).
#   입력 확인 → (검증을 통과한 캐시가 있으면 건너뜀) → 기록 → scripts/cache_base_q.py → 검증 → SHA-256 기록
# 끊기면 같은 명령으로 다시 낸다. 그 기사의 캐시를 처음부터 다시 만든다. 검증에 실패한 캐시는 지운다.
# Slurm 밖에서도 실행할 수 있다: bash gpu_train/cache.sbatch tal
# 종료 코드: 0 완료(또는 이미 있음), 1 실패, 2 입력 확인 실패
#
# 환경변수 (모두 선택)
#   SITE_ENV                설정 파일 (기본 gpu_train/site.env)
#   CACHE_BASE              베이스 (기본 base_model/270M.pt)
#   CACHE_DEVICE            scripts/cache_base_q.py --device (기본: CUDA가 있으면 cuda)
#   CACHE_DATA_ROOT         데이터 루트 (기본 data/processed. 테스트용)
#   CACHE_LOG_ROOT          작업 기록(meta/)과 GPU 기록(logs/)을 둘 곳 (기본 runs/gpu. 테스트용)
#   CACHE_SKIP_INPUT_CHECK  1이면 매니페스트 대조를 건너뛴다 (테스트 전용)
set -uo pipefail

[ $# -eq 1 ] || { echo "사용: bash gpu_train/cache.sbatch 기사(tal|karpov|smyslov)" >&2; exit 1; }
player=$1
case "$player" in tal|karpov|smyslov) ;; *) echo "알 수 없는 기사: $player" >&2; exit 1 ;; esac
in_slurm=0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  in_slurm=1   # sbatch는 스크립트를 복사해 돌리므로 위치 대신 --chdir(저장소 루트)를 믿는다
else
  cd "$(dirname "$0")/.." || exit 1
fi

# ── 준비 ──────────────────────────────────────────────
site_env=${SITE_ENV:-gpu_train/site.env}
if [ -f "$site_env" ]; then
  source "$site_env"
elif [ $in_slurm = 1 ]; then
  echo "$PWD/$site_env가 없다" >&2; exit 1
fi
if [ -n "${CONDA_SH:-}" ]; then
  set +u   # conda 스크립트는 정의되지 않은 변수를 참조한다
  source "$CONDA_SH" && conda activate "$CONDA_ENV" || { echo "conda 환경 $CONDA_ENV 활성화 실패" >&2; exit 1; }
  set -u
fi

base=${CACHE_BASE:-base_model/270M.pt}
data_dir=${CACHE_DATA_ROOT:-data/processed}/$player
cache=$data_dir/base_q_$(basename "$base" .pt).npz

# 베이스와 이 기사의 데이터만 매니페스트와 대조한다 (캐시는 아직 없을 수 있다)
if [ "${CACHE_SKIP_INPUT_CHECK:-0}" = 1 ]; then
  echo "입력 확인을 건너뛴다 (CACHE_SKIP_INPUT_CHECK=1, 테스트 전용)"
else
  lines=$(grep -E "  ($base|$data_dir/(train\.jsonl|val\.jsonl|stats\.json))\$" gpu_train/inputs.sha256)
  [ "$(echo "$lines" | grep -c .)" = 4 ] || { echo "inputs.sha256에 $base와 $data_dir 입력 3개가 모두 있어야 한다" >&2; exit 2; }
  echo "$lines" | sha256sum --quiet -c - || { echo "입력 확인 실패: fetch_inputs.sh를 다시 실행한다" >&2; exit 2; }
  echo "입력 확인 통과: $base, $data_dir/{train.jsonl,val.jsonl,stats.json}"
fi

log_root=${CACHE_LOG_ROOT:-runs/gpu}
mkdir -p "$log_root/logs" "$log_root/meta"
jobid=${SLURM_JOB_ID:-local_$(date '+%Y%m%d_%H%M%S')}
meta=$log_root/meta/cache_${player}_${jobid}.txt
note() { echo "=== $(date '+%F %T') $*" | tee -a "$meta"; }
check() { python -u -m gpu_train.check_cache --player "$player" --base "$base" --data-dir "$data_dir"; }

if [ -f "$cache" ]; then
  if check; then
    note "$cache 이미 있고 검증 통과. 건너뛴다 (SHA-256 $(sha256sum "$cache" | cut -c1-64))"
    exit 0
  fi
  note "$cache 검증 실패. 지우고 다시 만든다"
  rm -f "$cache"
fi

extra=(--player "$player" --base "$base" --data-dir "$data_dir")
[ -n "${CACHE_DEVICE:-}" ] && extra+=(--device "$CACHE_DEVICE")

# ── 기록 ──────────────────────────────────────────────
{
  echo "캐시: $cache"
  echo "시작: $(date '+%F %T')"
  echo "호스트: $(hostname)"
  echo "작업 ID: $jobid  노드: ${SLURM_JOB_NODELIST:-}"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '(git 없음)')"
  echo "git status --porcelain (비어 있으면 미커밋 변경 없음):"
  git status --porcelain 2>/dev/null | sed 's/^/  /'
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo '(nvidia-smi 없음)')"
  echo "torch: $(python -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| GPU 보임', torch.cuda.is_available())")"
  echo "입력 SHA-256:"
  sha256sum "$base" "$data_dir/train.jsonl" "$data_dir/val.jsonl" | sed 's/^/  /'
  echo "명령: python -u scripts/cache_base_q.py ${extra[*]}"
} >> "$meta"

# ── GPU 사용량 (60초마다) ───────────────────────────────
if command -v nvidia-smi > /dev/null; then
  ( while :; do
      echo "$(date '+%F %T'), $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
      sleep 60
    done ) >> "$log_root/logs/gpu_cache_${player}_${jobid}.log" 2>&1 &
  monitor=$!
  trap 'kill $monitor 2>/dev/null' EXIT
fi

# ── 캐시 만들기와 검증 ──────────────────────────────────
note "$player 캐시 시작"
python -u scripts/cache_base_q.py "${extra[@]}" 2>&1 | tee -a "$meta"
rc=${PIPESTATUS[0]}
if [ "$rc" != 0 ]; then
  note "$player 캐시 실패 (종료 코드 $rc)"
  rm -f "$cache"
  exit 1
fi
if ! check 2>&1 | tee -a "$meta"; then
  note "$player 캐시 검증 실패. 지운다"
  rm -f "$cache"
  exit 1
fi
note "$player 완료: $(stat -c %s "$cache") bytes, SHA-256 $(sha256sum "$cache" | cut -c1-64)"
exit 0
