#!/bin/bash
# Stockfish 공식 릴리스(리눅스 x86-64 universal)를 받아 SHA-256을 확인하고 runs/tools/stockfish/에 푼다.
# 사용: bash gpu_train/install_stockfish.sh   (저장소 루트 기준, 로컬 개발 컨테이너와 GPU 서버에서 같게 쓴다)
# 결과: runs/tools/stockfish/stockfish (실행 파일 링크). src/match/stockfish.py가 기본으로 이 경로를 찾는다.
# conda-forge에는 stockfish 패키지가 없어 공식 바이너리를 쓴다 (Task #13 1단계).
# 버전은 평가 기준 문서(evaluation_criteria.md 5.4절)에 고정한 값이다. 바꾸면 모든 대국·C4를 다시 해야 한다.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=sf_19
ASSET=stockfish-linux-x86-64-universal.tar.gz
SHA256=9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f
URL="https://github.com/official-stockfish/Stockfish/releases/download/$TAG/$ASSET"
DEST=runs/tools/stockfish

mkdir -p "$DEST"
if [ -x "$DEST/stockfish" ]; then
  echo "== 이미 설치됨: $DEST/stockfish"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  echo "== 받기: $URL"
  curl -fsSL --retry 3 -o "$tmp/$ASSET" "$URL"
  echo "$SHA256  $tmp/$ASSET" | sha256sum -c -
  # 압축 안에 stockfish/ 폴더(소스, 실행 파일)가 있어 release/ 아래에 푼다
  rm -rf "$DEST/release"
  mkdir -p "$DEST/release"
  tar -xzf "$tmp/$ASSET" -C "$DEST/release"
  # 실행 파일 이름은 stockfish-linux-x86-64-universal (sf_19). 하나만 있어야 한다
  mapfile -t found < <(find "$DEST/release" -type f -name 'stockfish-linux-*')
  [ ${#found[@]} -eq 1 ] || { echo "실행 파일을 하나로 정하지 못했다: ${found[*]:-없음}" >&2; exit 1; }
  chmod +x "${found[0]}"
  ln -sfn "$(realpath --relative-to="$DEST" "${found[0]}")" "$DEST/stockfish"
fi

echo "== 확인"
binary=$(realpath "$DEST/stockfish")
echo "실행 파일: $binary"
echo "SHA-256: $(sha256sum "$binary" | cut -d' ' -f1)"
printf 'uci\nquit\n' | "$DEST/stockfish" | grep '^id name'
printf 'compiler\nquit\n' | "$DEST/stockfish" | grep -i -E 'compiled|arch|settings' || true
# bench 노드 수는 빌드(CPU 명령어 집합)와 무관하게 같아야 한다. 로컬과 서버의 값을 대조해 같은 탐색인지 확인한다
echo "bench: $("$DEST/stockfish" bench 2>&1 | grep -E '^Nodes searched')"
