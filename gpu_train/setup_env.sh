#!/bin/bash
# conda 환경을 만들고 torch와 requirements.txt를 설치한 뒤 pytest로 확인한다. 로그인 노드에서 실행한다.
# 사용: bash gpu_train/setup_env.sh   (먼저 gpu_train/site.env를 채운다)
# 이미 환경이 있으면 만들지 않고 설치와 확인만 다시 한다.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f gpu_train/site.env ] || { echo "gpu_train/site.env가 없다. site.env.example을 복사해 채운다" >&2; exit 1; }
source gpu_train/site.env
[ -n "${CONDA_SH:-}" ] || { echo "site.env의 CONDA_SH가 비어 있다" >&2; exit 1; }
source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "== conda 환경 $CONDA_ENV 있음"
else
  echo "== conda 환경 $CONDA_ENV 만들기"
  conda env create -n "$CONDA_ENV" -f gpu_train/environment.yml
fi
conda activate "$CONDA_ENV"

echo "== torch 설치 (${TORCH_INDEX_URL:-PyPI 기본 휠})"
python -m pip install --upgrade pip
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
else
  python -m pip install torch
fi

echo "== requirements.txt 설치"
python -m pip install -r requirements.txt

echo "== 확인"
python -c "import sys, torch, chess; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| chess', chess.__version__)"
# 로그인 노드에는 GPU가 없을 수 있다. GPU 인식은 smoke 작업이 확인한다.
python -c "import torch; print('GPU 보임' if torch.cuda.is_available() else 'GPU 안 보임 (로그인 노드면 정상)')"
python -m pytest -q

mkdir -p runs/gpu/meta
freeze=runs/gpu/meta/env_$(date '+%Y%m%d_%H%M%S').txt
python -m pip freeze > "$freeze"
echo "== 완료. 설치 목록: $freeze"
