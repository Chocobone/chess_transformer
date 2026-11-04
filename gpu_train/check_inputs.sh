#!/bin/bash
# 입력 파일이 gpu_train/inputs.sha256과 같은지 확인한다. 하나라도 없거나 다르면 종료 코드 1.
# 사용: bash gpu_train/check_inputs.sh   (어디서 실행해도 저장소 루트 기준으로 본다)
set -euo pipefail
cd "$(dirname "$0")/.."

if sha256sum --quiet -c gpu_train/inputs.sha256; then
  echo "입력 확인 통과: $(grep -cv '^#' gpu_train/inputs.sha256)개"
else
  echo "입력 확인 실패: fetch_inputs.sh를 다시 실행하거나 위 파일을 다시 옮긴다 (gpu_train/README.md 5절)" >&2
  exit 1
fi
