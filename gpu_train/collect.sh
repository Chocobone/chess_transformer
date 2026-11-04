#!/bin/bash
# 학습 하나의 결과를 tar.gz 하나로 묶는다. 로컬로 가져와 mydocs/test_result/에 정리할 때 쓴다.
# 사용: bash gpu_train/collect.sh R1   → runs/gpu/collect/R1_{시각}.tar.gz
# 넣는 것: 어댑터·설정·로그·DONE, 작업 출력(runs/gpu/logs/R1_*.out), GPU 로그, 기록 파일, 마지막 설치 목록, SUMMARY.txt
# checkpoint.pt(재개용)는 넣지 않는다. DONE이 없어도 경고만 하고 묶는다 (중간 회수용).
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -eq 1 ] || { echo "사용: bash gpu_train/collect.sh 학습ID" >&2; exit 1; }
run=$1
out=$(python -m gpu_train.runs out "$run")
[ -f "$out/log.jsonl" ] || { echo "$out/log.jsonl이 없다. 학습이 시작되지 않았다" >&2; exit 1; }
[ -f "$out/DONE" ] || echo "경고: $out/DONE이 없다. 끝나지 않은 학습을 묶는다" >&2

name=${run}_$(date '+%Y%m%d_%H%M%S')
stage=runs/gpu/collect/.$name
mkdir -p "$stage"
trap 'rm -rf "${stage:?}"' EXIT

shopt -s nullglob
files=()
for f in "$out"/{adapter.pt,config.json,log.jsonl,DONE} runs/gpu/logs/"${run}"_*.out \
         runs/gpu/logs/gpu_"${run}".log runs/gpu/meta/"${run}"_*.txt; do
  [ -f "$f" ] && files+=("$f")
done
envs=(runs/gpu/meta/env_*.txt)
[ ${#envs[@]} -gt 0 ] && files+=("${envs[-1]}")   # 이름에 시각이 들어 있어 마지막이 최신이다
cp --parents "${files[@]}" "$stage/"

{
  echo "학습 ID: $run"
  echo "묶은 시각: $(date '+%F %T')  호스트: $(hostname)"
  echo "완료: $(cat "$out/DONE" 2>/dev/null || echo '(DONE 없음, 미완료)')"
  if [ -f "$out/adapter.pt" ]; then
    echo "adapter.pt: $(stat -c %s "$out/adapter.pt") bytes  SHA-256 $(sha256sum "$out/adapter.pt" | cut -d' ' -f1)"
  else
    echo "adapter.pt: (없음)"
  fi
  echo
  python - "$out/log.jsonl" <<'EOF'
import json, sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
def show(label, record):
    print(f"{label}: {json.dumps(record, ensure_ascii=False) if record else '(없음)'}")
show("baseline", next((r for r in records if r["type"] == "baseline"), None))
show("마지막 eval", next((r for r in reversed(records) if r["type"] == "eval"), None))
show("마지막 train", next((r for r in reversed(records) if r["type"] == "train"), None))
resumes = [r["step"] for r in records if r["type"] == "resume"]
print(f"재개: {len(resumes)}회 {resumes}")
EOF
  echo
  echo "== 작업과 시도 (기록 파일의 === 줄)"
  for f in runs/gpu/meta/"${run}"_*.txt; do
    echo "-- $(basename "$f")"
    grep '^===' "$f" || echo "(시도 기록 없음)"
  done
} > "$stage/SUMMARY.txt"

tarball=runs/gpu/collect/$name.tar.gz
tar -czf "$tarball" -C "$stage" .
echo "묶음: $tarball ($(stat -c %s "$tarball") bytes)"
tar -tzf "$tarball" | grep -v '/$' | sed 's|^\./|  |'
