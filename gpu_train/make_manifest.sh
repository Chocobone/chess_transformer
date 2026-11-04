#!/bin/bash
# 서버로 옮길 입력 파일의 SHA-256 목록(gpu_train/inputs.sha256)을 만든다. 로컬에서 실행하고 결과를 커밋한다.
# 사용: bash gpu_train/make_manifest.sh   (저장소 루트에서)
set -euo pipefail
cd "$(dirname "$0")/.."

files=(base_model/136M.pt)
for player in tal karpov smyslov; do
  for name in train.jsonl val.jsonl stats.json base_q_136M.npz; do
    files+=("data/processed/$player/$name")
  done
done

out=gpu_train/inputs.sha256
{
  echo "# 입력 파일 ${#files[@]}개. 확인: bash gpu_train/check_inputs.sh"
  echo "# 만든 날: $(date '+%F'), 커밋: $(git rev-parse --short HEAD)"
  for f in "${files[@]}"; do echo "# $(stat -c %s "$f") bytes  $f"; done
  sha256sum "${files[@]}"
} > "$out"
echo "$out: ${#files[@]}개, 합계 $(du -cb "${files[@]}" | tail -1 | cut -f1) bytes"
