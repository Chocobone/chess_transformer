#!/bin/bash
#SBATCH --output=runs/gpu/logs/%x_%j.out
# 학습 폴더 여러 개를 다시 평가하는 작업 본문. 제출은 gpu_train/submit_eval.sh가 한다.
#   입력 확인 → 기록 → 폴더마다 gpu_train.eval_adapter --baseline → 판정 표
# 폴더마다 val 전체를 다시 재서 그 폴더 log.jsonl의 마지막 eval·baseline 줄과 비교한다.
# 하나가 실패해도 나머지는 평가한다. 평가는 짧아 재개·재시도·이어서 제출을 하지 않는다.
# Slurm 밖에서도 실행할 수 있다: bash gpu_train/eval.sbatch runs/gpu/R1 runs/gpu/R2
# 종료 코드: 0 모두 통과, 1 하나라도 실패·오류, 2 입력 확인 실패
#
# 환경변수 (모두 선택)
#   SITE_ENV               설정 파일 (기본 gpu_train/site.env)
#   EVAL_OUT               결과 폴더 (기본 runs/gpu/eval)
#   EVAL_DEVICE            eval_adapter --device (기본: CUDA가 있으면 cuda)
#   EVAL_NUM_WORKERS       eval_adapter --num-workers (기본 2)
#   EVAL_SKIP_INPUT_CHECK  1이면 check_inputs.sh를 건너뛴다 (테스트 전용)
set -uo pipefail

[ $# -ge 1 ] || { echo "사용: bash gpu_train/eval.sbatch 학습폴더 [학습폴더 ...]" >&2; exit 1; }
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

if [ "${EVAL_SKIP_INPUT_CHECK:-0}" = 1 ]; then
  echo "입력 확인을 건너뛴다 (EVAL_SKIP_INPUT_CHECK=1, 테스트 전용)"
else
  bash gpu_train/check_inputs.sh || exit 2
fi

out=${EVAL_OUT:-runs/gpu/eval}
mkdir -p "$out"
jobid=${SLURM_JOB_ID:-local_$(date '+%Y%m%d_%H%M%S')}
meta=$out/meta_${jobid}.txt
note() { echo "=== $(date '+%F %T') $*" | tee -a "$meta"; }

extra=()
[ -n "${EVAL_DEVICE:-}" ] && extra+=(--device "$EVAL_DEVICE")
extra+=(--num-workers "${EVAL_NUM_WORKERS:-2}")

# ── 기록 ──────────────────────────────────────────────
{
  echo "재평가 대상: $*"
  echo "시작: $(date '+%F %T')"
  echo "호스트: $(hostname)"
  echo "작업 ID: $jobid  노드: ${SLURM_JOB_NODELIST:-}"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '(git 없음)')"
  echo "git status --porcelain (비어 있으면 미커밋 변경 없음):"
  git status --porcelain 2>/dev/null | sed 's/^/  /'
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo '(nvidia-smi 없음)')"
  echo "torch: $(python -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| GPU 보임', torch.cuda.is_available())")"
  echo "옵션: ${extra[*]}"
} >> "$meta"

# ── GPU 사용량 (60초마다) ───────────────────────────────
if command -v nvidia-smi > /dev/null; then
  ( while :; do
      echo "$(date '+%F %T'), $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
      sleep 60
    done ) >> "$out/gpu_${jobid}.log" 2>&1 &
  monitor=$!
  trap 'kill $monitor 2>/dev/null' EXIT
fi

# ── 폴더마다 평가 ──────────────────────────────────────
# 판정은 결과 JSON의 pass로 읽는다. JSON이 없으면 평가 중 오류다 (판정 실패와 구분한다)
declare -A seen=()
results=()
failed=0
for dir in "$@"; do
  name=$(basename "$dir")
  if [ -n "${seen[$name]:-}" ]; then
    note "$dir 건너뜀: 결과 이름 $name이 ${seen[$name]}와 겹친다"
    results+=("$name 오류(이름 겹침)"); failed=1; continue
  fi
  seen[$name]=$dir
  json=$out/$name.json
  rm -f "$json"
  note "$dir 평가 시작"
  python -u -m gpu_train.eval_adapter "$dir" --baseline --json "$json" "${extra[@]}" 2>&1 | tee "$out/$name.txt"
  rc=${PIPESTATUS[0]}
  if [ -f "$json" ]; then
    verdict=$(python -c "import json, sys; print('통과' if json.load(open(sys.argv[1], encoding='utf-8'))['pass'] else '실패')" "$json")
  else
    verdict="오류(종료 코드 $rc)"
  fi
  [ "$verdict" = 통과 ] || failed=1
  note "$dir 끝: $verdict"
  results+=("$name $verdict")
done

# ── 판정 표 ────────────────────────────────────────────
{
  echo "== 판정 ($(date '+%F %T'))"
  printf '  %s\n' "${results[@]}"
} | tee -a "$meta"
exit $failed
