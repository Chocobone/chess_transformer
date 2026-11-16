#!/bin/bash
#SBATCH --output=runs/gpu/logs/%x_%j.out
# 짝 대국 여러 묶음을 차례로 두는 작업 본문 (Task #13, 평가 기준 문서 5.3절). 제출은 gpu_train/submit_match.sh가 한다.
#   준비 → Stockfish 확인(버전, bench) → 기록 → 묶음마다 python -m src.match.play → 결과 표
# 묶음 표기: 이름:모델:상대:노드수:오프닝  (상대가 base이면 노드수는 -)
#   예: cal_n100:base:sf:100:0-9   C1_R1:R1:base:-:0-199   C2_R1:R1:sf:300:0-199
# 결과는 {MATCH_OUT}/{이름}/ (games.pgn, index.jsonl, config.json, runs.jsonl)에 쌓인다.
# 끝난 대국은 건너뛰므로, 끊기면 같은 명령으로 다시 낸다. 하나가 실패해도 나머지 묶음은 둔다.
# 종료 코드: 0 모두 끝남, 1 하나라도 실패, 2 준비 실패
#
# 환경변수 (모두 선택)
#   SITE_ENV           설정 파일 (기본 gpu_train/site.env)
#   MATCH_OUT          결과 폴더 (기본 runs/gpu/match)
#   MATCH_CONCURRENT   동시 대국 수 (기본 64)
#   MATCH_SF_WORKERS   Stockfish 프로세스 수 (기본: 할당 CPU 수 − 1, 최소 1)
#   MATCH_DEVICE       --device (기본: CUDA가 있으면 cuda)
set -uo pipefail

[ $# -ge 1 ] || { echo "사용: bash gpu_train/match.sbatch 이름:모델:상대:노드수:오프닝 [...]" >&2; exit 1; }
in_slurm=0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  in_slurm=1
else
  cd "$(dirname "$0")/.." || exit 1
fi

# ── 준비 ──────────────────────────────────────────────
site_env=${SITE_ENV:-gpu_train/site.env}
if [ -f "$site_env" ]; then
  source "$site_env"
elif [ $in_slurm = 1 ]; then
  echo "$PWD/$site_env가 없다" >&2; exit 2
fi
if [ -n "${CONDA_SH:-}" ]; then
  set +u
  source "$CONDA_SH" && conda activate "$CONDA_ENV" || { echo "conda 환경 $CONDA_ENV 활성화 실패" >&2; exit 2; }
  set -u
fi

sf=runs/tools/stockfish/stockfish
[ -x "$sf" ] || { echo "$sf가 없다. bash gpu_train/install_stockfish.sh로 설치한다" >&2; exit 2; }
[ -f base_model/136M.pt ] || { echo "base_model/136M.pt가 없다" >&2; exit 2; }

out=${MATCH_OUT:-runs/gpu/match}
mkdir -p "$out"
jobid=${SLURM_JOB_ID:-local_$(date '+%Y%m%d_%H%M%S')}
meta=$out/meta_${jobid}.txt
note() { echo "=== $(date '+%F %T') $*" | tee -a "$meta"; }

cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
workers=${MATCH_SF_WORKERS:-$(( cpus > 1 ? cpus - 1 : 1 ))}
concurrent=${MATCH_CONCURRENT:-64}
common=(--concurrent "$concurrent" --sf-workers "$workers")
[ -n "${MATCH_DEVICE:-}" ] && common+=(--device "$MATCH_DEVICE")

# ── 기록 ──────────────────────────────────────────────
{
  echo "묶음: $*"
  echo "시작: $(date '+%F %T')"
  echo "호스트: $(hostname)"
  echo "작업 ID: $jobid  노드: ${SLURM_JOB_NODELIST:-}  CPU: $cpus"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '(git 없음)')"
  echo "git status --porcelain (비어 있으면 미커밋 변경 없음):"
  git status --porcelain 2>/dev/null | sed 's/^/  /'
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo '(nvidia-smi 없음)')"
  echo "torch: $(python -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| GPU 보임', torch.cuda.is_available())")"
  echo "CPU: $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/.*: *//')"
  echo "Stockfish: $(printf 'uci\nquit\n' | "$sf" | grep '^id name')  SHA-256 $(sha256sum "$(realpath "$sf")" | cut -d' ' -f1)"
  echo "Stockfish 명령어 집합: $(printf 'compiler\nquit\n' | "$sf" | grep -i 'Compilation architecture' | sed 's/.*: *//')"
  echo "Stockfish bench: $("$sf" bench 2>&1 | grep -E '^Nodes searched' | sed 's/.*: *//') (로컬 2497913과 같아야 한다)"
  echo "옵션: ${common[*]}"
} >> "$meta"

if command -v nvidia-smi > /dev/null; then
  ( while :; do
      echo "$(date '+%F %T'), $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
      sleep 60
    done ) >> "$out/gpu_${jobid}.log" 2>&1 &
  monitor=$!
  trap 'kill $monitor 2>/dev/null' EXIT
fi

# ── 묶음마다 대국 ──────────────────────────────────────
results=()
failed=0
for spec in "$@"; do
  IFS=: read -r name model opponent nodes openings <<< "$spec"
  args=(--model "$model" --opponent "$opponent" --openings "$openings" --out "$out/$name" "${common[@]}")
  [ "$nodes" != "-" ] && args+=(--nodes "$nodes")
  note "$name 시작 ($spec)"
  python -u -m src.match.play "${args[@]}" 2>&1 | tee -a "$out/$name.txt"
  rc=${PIPESTATUS[0]}
  games=$(wc -l < "$out/$name/index.jsonl" 2>/dev/null || echo 0)
  if [ $rc = 0 ]; then verdict="끝남 (${games}국)"; else verdict="실패 (종료 코드 $rc, ${games}국)"; failed=1; fi
  note "$name $verdict"
  results+=("$name $verdict")
done

{
  echo "== 결과 ($(date '+%F %T'))"
  printf '  %s\n' "${results[@]}"
} | tee -a "$meta"
exit $failed
