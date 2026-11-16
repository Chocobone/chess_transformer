#!/bin/bash
#SBATCH --output=runs/gpu/logs/%x_%j.out
# C4 Stockfish 기준 수 품질 탐색 작업 본문 (Task #13, 평가 기준 문서 5.4절). 제출은 gpu_train/submit_quality.sh가 한다.
#   준비 → Stockfish 확인 → 기록 → python -m src.match.quality
# 모델 1순위 수는 Task #12 덤프(runs/gpu/dump)에서 읽고, 국면 FEN은 val 기록(data/processed)에서 읽는다. GPU를 쓰지 않는다.
# 탐색 결과를 캐시하므로 끊기면 같은 명령으로 다시 낸다. 모델을 더해 다시 내면 새 수만 탐색한다.
# 종료 코드: 0 끝남, 1 실패, 2 준비 실패
#
# 환경변수 (모두 선택)
#   SITE_ENV         설정 파일 (기본 gpu_train/site.env)
#   QUALITY_DEPTH    탐색 깊이 (기본 18. 평가 기준 문서에 고정한 값)
#   QUALITY_OUT      결과 폴더 (기본 runs/gpu/quality)
#   QUALITY_DUMP     덤프 폴더 (기본 runs/gpu/dump)
#   QUALITY_WORKERS  Stockfish 프로세스 수 (기본: 할당 CPU 수)
set -uo pipefail

[ $# -ge 1 ] || { echo "사용: bash gpu_train/quality.sbatch 모델ID [...]   (예: base R1 R8 R2)" >&2; exit 1; }
in_slurm=0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  in_slurm=1
else
  cd "$(dirname "$0")/.." || exit 1
fi

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
dump=${QUALITY_DUMP:-runs/gpu/dump}
[ -f "$dump/positions.npz" ] || { echo "$dump/positions.npz가 없다 (Task #12 덤프)" >&2; exit 2; }
for model in "$@"; do
  [ -f "$dump/$model.npz" ] || { echo "$dump/$model.npz가 없다" >&2; exit 2; }
done

out=${QUALITY_OUT:-runs/gpu/quality}
mkdir -p "$out"
jobid=${SLURM_JOB_ID:-local_$(date '+%Y%m%d_%H%M%S')}
meta=$out/meta_${jobid}.txt
depth=${QUALITY_DEPTH:-18}
workers=${QUALITY_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}

{
  echo "모델: $*  깊이: $depth  프로세스: $workers"
  echo "시작: $(date '+%F %T')"
  echo "호스트: $(hostname)  작업 ID: $jobid  노드: ${SLURM_JOB_NODELIST:-}"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '(git 없음)')"
  echo "git status --porcelain (비어 있으면 미커밋 변경 없음):"
  git status --porcelain 2>/dev/null | sed 's/^/  /'
  echo "CPU: $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/.*: *//')"
  echo "Stockfish: $(printf 'uci\nquit\n' | "$sf" | grep '^id name')  SHA-256 $(sha256sum "$(realpath "$sf")" | cut -d' ' -f1)"
  echo "Stockfish 명령어 집합: $(printf 'compiler\nquit\n' | "$sf" | grep -i 'Compilation architecture' | sed 's/.*: *//')"
  echo "Stockfish bench: $("$sf" bench 2>&1 | grep -E '^Nodes searched' | sed 's/.*: *//') (로컬 2497913과 같아야 한다)"
} >> "$meta"

python -u -m src.match.quality --dump "$dump" --models "$@" --depth "$depth" --workers "$workers" --out "$out" \
  2>&1 | tee -a "$out/quality_${jobid}.txt"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') 끝: 종료 코드 $rc" | tee -a "$meta"
[ $rc = 0 ] || exit 1
