#!/bin/bash
#SBATCH --output=runs/gpu/logs/%x_%j.out
# 학습 하나의 작업 본문. 제출은 gpu_train/submit.sh가 한다 (파티션, GPU, 시간 등은 submit.sh가 붙인다).
#   입력 확인 → 완료 확인 → 다음 작업 예약 → 기록 → 학습(실패하면 재시도) → 완료 표시
# 시간 제한으로 끊기면 예약해 둔 다음 작업이 체크포인트에서 이어 간다. 학습이 끝나면 다음 작업을 취소한다.
# Slurm 밖에서도 실행할 수 있다 (이어서 제출 없이): bash gpu_train/train.sbatch smoke
# 종료 코드: 0 완료(또는 이미 완료), 1 실패·중단, 2 입력 확인 실패
set -uo pipefail

[ $# -eq 1 ] || { echo "사용: bash gpu_train/train.sbatch 학습ID" >&2; exit 1; }
run=$1
in_slurm=0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  in_slurm=1   # sbatch는 스크립트를 복사해 돌리므로 위치 대신 --chdir(저장소 루트)를 믿는다
else
  cd "$(dirname "$0")/.." || exit 1
fi

# ── 준비 ──────────────────────────────────────────────
if [ -f gpu_train/site.env ]; then
  source gpu_train/site.env
elif [ $in_slurm = 1 ]; then
  echo "$PWD/gpu_train/site.env가 없다" >&2; exit 1
fi
if [ -n "${CONDA_SH:-}" ]; then
  set +u   # conda 스크립트는 정의되지 않은 변수를 참조한다
  source "$CONDA_SH" && conda activate "$CONDA_ENV" || { echo "conda 환경 $CONDA_ENV 활성화 실패" >&2; exit 1; }
  set -u
fi

out=$(python -m gpu_train.runs out "$run") || exit 1
args=$(python -m gpu_train.runs args "$run") || exit 1
read -ra train_cmd <<< "$args"
bash gpu_train/check_inputs.sh || exit 2

mkdir -p "$out" runs/gpu/logs runs/gpu/meta
if [ -f "$out/DONE" ]; then
  echo "=== $run 이미 완료: $(cat "$out/DONE")"
  exit 0
fi

jobid=${SLURM_JOB_ID:-local_$(date '+%Y%m%d_%H%M%S')}
chain=${CHAIN_INDEX:-1}
meta=runs/gpu/meta/${run}_${jobid}.txt
note() { echo "=== $(date '+%F %T') $*" | tee -a "$meta"; }
ckpt_step() {
  [ -f "$out/checkpoint.pt" ] || { echo 0; return; }
  python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" \
    "$out/checkpoint.pt" 2>/dev/null || echo 0
}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start=$(ckpt_step)

# ── 기록 ──────────────────────────────────────────────
{
  echo "학습 ID: $run"
  echo "시작: $(date '+%F %T')"
  echo "호스트: $(hostname)"
  echo "작업 ID: $jobid  사슬 번호: $chain / ${CHAIN_MAX:-1}  노드: ${SLURM_JOB_NODELIST:-}"
  echo "앞 작업 시작 step: ${PREV_START:-}"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '(git 없음)')"
  echo "git status --porcelain (비어 있으면 미커밋 변경 없음):"
  git status --porcelain 2>/dev/null | sed 's/^/  /'
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo '(nvidia-smi 없음)')"
  echo "torch: $(python -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| GPU 보임', torch.cuda.is_available())")"
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  echo "명령: ${train_cmd[*]}"
  echo "시작 체크포인트 step: $start"
} >> "$meta"

# ── 전진 확인과 다음 작업 예약 (Slurm 안에서만) ──────────
next=
cancel_next() {
  if [ -n "$next" ]; then
    scancel "$next" && note "다음 작업 $next 취소"
  fi
}
if [ $in_slurm = 1 ]; then
  if [ "$chain" -gt 1 ] && [ -n "${PREV_START:-}" ] && [ "$start" -le "$PREV_START" ]; then
    note "$run 중단: 앞 작업 동안 체크포인트가 전진하지 않았다 (step $PREV_START → $start). 다음 작업을 예약하지 않는다"
    exit 1
  fi
  if [ "$chain" -lt "${CHAIN_MAX:-1}" ]; then
    next=$(SUBMIT_AFTER=$SLURM_JOB_ID CHAIN_INDEX=$((chain + 1)) PREV_START=$start bash gpu_train/submit.sh "$run") \
      && note "다음 작업 $next 예약 (사슬 $((chain + 1))번, 이 작업이 끝난 뒤 시작)" \
      || { next=; note "다음 작업 예약 실패. 이 작업이 시간 제한으로 끊기면 사람이 다시 제출한다"; }
  else
    note "사슬 상한 ${CHAIN_MAX:-1}에 이르러 다음 작업을 예약하지 않는다"
  fi
fi

# ── GPU 사용량 (60초마다) ───────────────────────────────
if command -v nvidia-smi > /dev/null; then
  ( while :; do
      echo "$(date '+%F %T'), $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
      sleep 60
    done ) >> "runs/gpu/logs/gpu_${run}.log" 2>&1 &
  monitor=$!
  trap 'kill $monitor 2>/dev/null' EXIT
fi
# 시간 제한(SIGTERM)으로 끊길 때는 재시도하지 않고, 예약한 다음 작업이 이어 가게 둔다
terminated=0
trap 'terminated=1' TERM

# ── 학습과 재시도 ──────────────────────────────────────
stuck=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  before=$(ckpt_step)
  note "$run 시도 $attempt 시작 (체크포인트 step $before)"
  "${train_cmd[@]}"
  rc=$?
  after=$(ckpt_step)
  if [ $terminated = 1 ]; then
    note "$run 시간 제한 등으로 종료 신호를 받았다 (종료 코드 $rc, 체크포인트 step $after). 다음 작업 ${next:-(없음)}이 이어 간다"
    exit 1
  fi
  if [ $rc -eq 0 ]; then
    echo "$(date '+%F %T') 작업 $jobid (시도 $attempt회, step $after)" > "$out/DONE"
    note "$run 완료 (시도 $attempt회, step $after)"
    cancel_next
    exit 0
  fi
  note "$run 시도 $attempt 실패 (종료 코드 $rc, 체크포인트 step $after)"
  if [ "$after" -gt "$before" ]; then stuck=0; else stuck=$((stuck + 1)); fi
  if [ $stuck -ge 3 ]; then
    note "$run 중단: 체크포인트 전진 없이 연속 3회 실패"
    cancel_next
    exit 1
  fi
  sleep 20
  if [ $terminated = 1 ]; then note "$run 종료 신호를 받아 재시도하지 않는다"; exit 1; fi
done
note "$run 중단: 시도 10회 소진"
cancel_next
exit 1
