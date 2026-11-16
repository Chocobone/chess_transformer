#!/bin/bash
# Stockfish 19를 공식 릴리스 소스로 정적 링크 빌드한다 (로컬 개발 컨테이너에서, root 또는 g++가 있는 리눅스).
# 사용: bash gpu_train/build_stockfish.sh   → runs/tools/stockfish-dist/stockfish-19-x86-64-avx2-static
#
# 왜 직접 빌드하나: 공식 리눅스 바이너리는 glibc 2.35를 요구해 GPU 서버(glibc 2.32 미만)에서 돌지 않는다 (Task #13 4단계).
# 정적 링크 바이너리는 glibc에 의존하지 않아 로컬과 서버가 같은 파일(같은 SHA-256)을 쓴다. 서버로는 파일을 옮긴다.
# 소스: 공식 릴리스 묶음(sf_19, SHA-256 확인)에 들어 있는 src/. NNUE 네트는 Stockfish 스크립트가 받아 SHA-256을 확인한다.
# 탐색이 공식 바이너리와 같은지는 bench 노드 수(2497913)로 확인한다.
# ARCH는 x86-64-avx2 (AVX2·BMI1 필요). 서버 계산 노드 CPU가 지원하는지 먼저 확인한다.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=sf_19
ASSET=stockfish-linux-x86-64-universal.tar.gz
SHA256=9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f
URL="https://github.com/official-stockfish/Stockfish/releases/download/$TAG/$ASSET"
ARCH=x86-64-avx2
BENCH=2497913
WORK=runs/tools/sfbuild
DIST=runs/tools/stockfish-dist
NAME=stockfish-19-$ARCH-static

command -v g++ > /dev/null && command -v make > /dev/null || {
  echo "g++와 make가 필요하다. 개발 컨테이너에서는 root로 apt-get install -y build-essential" >&2; exit 1; }

rm -rf "$WORK"
mkdir -p "$WORK" "$DIST"
echo "== 소스 받기: $URL"
curl -fsSL --retry 3 -o "$WORK/$ASSET" "$URL"
echo "$SHA256  $WORK/$ASSET" | sha256sum -c -
tar -xzf "$WORK/$ASSET" -C "$WORK"

cd "$WORK/stockfish/src"
echo "== 네트 받기"
make net
echo "== 빌드 (ARCH=$ARCH, 정적 링크, $(g++ --version | head -1))"
make -j"$(nproc)" build ARCH=$ARCH COMP=gcc EXTRALDFLAGS=-static > ../../build.log 2>&1 || { tail -20 ../../build.log; exit 1; }
cd - > /dev/null

cp "$WORK/stockfish/src/stockfish" "$DIST/$NAME"
binary=$DIST/$NAME
echo "== 확인"
linkage=$(ldd "$binary" 2>&1 || true)   # 정적 파일이면 ldd가 종료 코드 1을 낸다
[[ "$linkage" == *"not a dynamic executable"* ]] || { echo "정적 링크가 아니다: $linkage" >&2; exit 1; }
printf 'uci\nquit\n' | "$binary" | grep '^id name'
nodes=$("$binary" bench 2>&1 | grep -E '^Nodes searched' | sed 's/.*: *//')
echo "bench: $nodes"
[ "$nodes" = "$BENCH" ] || { echo "bench가 $BENCH와 다르다" >&2; exit 1; }
echo "실행 파일: $binary"
echo "SHA-256: $(sha256sum "$binary" | cut -d' ' -f1)"
echo "네트 SHA-256: $(sha256sum "$WORK"/stockfish/src/nn-*.nnue | cut -d' ' -f1)"
