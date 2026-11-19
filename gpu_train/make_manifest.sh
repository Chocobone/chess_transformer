#!/bin/bash
# 서버로 옮길 입력 파일의 SHA-256 목록(gpu_train/inputs.sha256)을 만든다. 로컬에서 실행하고 결과를 커밋한다.
# 사용: bash gpu_train/make_manifest.sh               (저장소 루트에서, 17개)
#       bash gpu_train/make_manifest.sh --base-only   (270M 캐시를 빼고 14개. 서버에서 캐시를 만들기 전 업로드용)
# 앞 13개(136M 입력)의 순서는 그대로 두고, 270M 입력은 그 뒤에 붙인다 (Task #14).
set -euo pipefail
cd "$(dirname "$0")/.."

base_only=0
case "${1:-}" in
  "") ;;
  --base-only) base_only=1 ;;
  *) echo "사용: bash gpu_train/make_manifest.sh [--base-only]" >&2; exit 1 ;;
esac

files=(base_model/136M.pt)
for player in tal karpov smyslov; do
  for name in train.jsonl val.jsonl stats.json base_q_136M.npz; do
    files+=("data/processed/$player/$name")
  done
done
files+=(base_model/270M.pt)
if [ $base_only = 0 ]; then
  for player in tal karpov smyslov; do
    files+=("data/processed/$player/base_q_270M.npz")
  done
fi

out=gpu_train/inputs.sha256
{
  echo "# 입력 파일 ${#files[@]}개. 확인: bash gpu_train/check_inputs.sh"
  echo "# 만든 날: $(date '+%F'), 커밋: $(git rev-parse --short HEAD)"
  for f in "${files[@]}"; do echo "# $(stat -c %s "$f") bytes  $f"; done
  sha256sum --text "${files[@]}"   # Git Bash(Windows)도 "해시  경로" 형식으로 쓴다 (기본은 "해시 *경로")
} > "$out"
echo "$out: ${#files[@]}개, 합계 $(du -cb "${files[@]}" | tail -1 | cut -f1) bytes"
