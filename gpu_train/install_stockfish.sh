#!/bin/bash
# Stockfish 19 정적 빌드 바이너리를 SHA-256으로 확인하고 runs/tools/stockfish/에 설치한다 (로컬 개발 컨테이너와 GPU 서버 공통).
# 사용: bash gpu_train/install_stockfish.sh [바이너리 경로]
#   경로를 주지 않으면 runs/tools/stockfish-dist/stockfish-19-x86-64-avx2-static을 쓴다.
#   로컬: bash gpu_train/build_stockfish.sh로 만든다.
#   서버: 로컬에서 만든 파일을 같은 경로로 옮긴 뒤 실행한다 (scp 등). setup_env.sh가 이 스크립트를 부른다.
# 결과: runs/tools/stockfish/stockfish (실행 파일 링크). src/match/stockfish.py가 기본으로 이 경로를 찾는다.
#
# 공식 리눅스 바이너리는 glibc 2.35를 요구해 서버에서 돌지 않아, 같은 버전 소스를 정적 링크로 빌드한 파일 하나를
# 로컬과 서버가 함께 쓴다 (Task #13 4단계). 버전은 평가 기준 문서(evaluation_criteria.md 5.4절)에 고정한 값이다.
# bench 노드 수가 공식 바이너리와 같아야(2497913) 탐색이 같다.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=stockfish-19-x86-64-avx2-static
SHA256=1d65ab6d2cf0a35d89c5643588502aaf25458d4a10ca8774e940cd8865784632
BENCH=2497913
SOURCE=${1:-runs/tools/stockfish-dist/$NAME}
DEST=runs/tools/stockfish

# 이 빌드는 AVX2·BMI1을 쓴다. 지원하지 않는 CPU에서는 잘못된 명령어로 죽으므로 먼저 확인한다
flags=$(grep -m1 '^flags' /proc/cpuinfo || true)
for flag in avx2 bmi1 popcnt sse4_1; do
  [[ " $flags " == *" $flag "* ]] || { echo "이 CPU($(hostname))는 $flag를 지원하지 않는다. 다른 ARCH로 빌드해야 한다" >&2; exit 1; }
done

if [ -f "$DEST/$NAME" ] && [ "$(sha256sum "$DEST/$NAME" | cut -d' ' -f1)" = "$SHA256" ]; then
  echo "== 이미 설치됨: $DEST/$NAME"
else
  [ -f "$SOURCE" ] || { echo "$SOURCE가 없다. 로컬에서 build_stockfish.sh로 만든 파일을 이 경로로 옮긴다" >&2; exit 1; }
  echo "$SHA256  $SOURCE" | sha256sum -c -
  mkdir -p "$DEST"
  rm -rf "$DEST/release"   # 이전 방식(공식 바이너리)의 흔적
  cp "$SOURCE" "$DEST/$NAME"
  chmod +x "$DEST/$NAME"
fi
ln -sfn "$NAME" "$DEST/stockfish"

echo "== 확인"
binary=$(realpath "$DEST/stockfish")
echo "실행 파일: $binary"
echo "SHA-256: $(sha256sum "$binary" | cut -d' ' -f1)"
printf 'uci\nquit\n' | "$DEST/stockfish" | grep '^id name'
printf 'compiler\nquit\n' | "$DEST/stockfish" | grep -i -E 'compiled|arch' || true
nodes=$("$DEST/stockfish" bench 2>&1 | grep -E '^Nodes searched' | sed 's/.*: *//')
echo "bench: $nodes"
[ "$nodes" = "$BENCH" ] || { echo "bench가 $BENCH와 다르다. 탐색이 로컬과 다르므로 진행하지 않는다" >&2; exit 1; }
