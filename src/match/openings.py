"""C1·C2 대국의 오프닝 200개 선별 (평가 기준 문서 5.3절, Task #13 구현 계획서 해석 7).

출처는 Stockfish 공식 오프닝 저장소 `official-stockfish/books`(CC0)의 `noob_4moves.epd`다. 양쪽 4수(8 ply) 뒤 백 차례
국면 1,837,365개이며, 수순 없이 FEN만 있다 (작업지시자 결정, 2026-11-16. `8moves_v3`는 16 ply라 기준과 맞지 않았다).

선별 규칙:
1. 책의 모든 줄을 읽어 FEN 앞 4필드(배치·차례·캐슬링·앙파상)로 중복을 없앤다. 첫 등장 순서를 지킨다
2. 규칙에 맞고(백 차례, 합법수 있음) 대국이 끝나지 않은 국면만 남긴다
3. `default_rng(seed)`의 순열 순서대로 고정 깊이로 평가해, 백 기준 |cp| ≤ limit_cp(메이트 점수 제외)인 국면을
   n개 모일 때까지 고른다. 오프닝 번호는 고른 순서다 (순열에서 차례로 고르므로 무작위 표본과 같다)

사용 (개발 컨테이너, Stockfish 설치 뒤):
    python -m src.match.openings --depth 18 --out src/match/openings_200.json
"""

import argparse
import datetime
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import chess
import chess.engine
import numpy as np

BOOK_REPO = "official-stockfish/books"
BOOK_COMMIT = "77ba28be2480f3b26a915b43b924841e107212fa"
BOOK_FILE = "noob_4moves.epd"
BOOK_ZIP_SHA256 = "ca8e161fc207e79abfc6fd0e2df3cddfe4b2614a3d871135150ff1c8e1bd9802"
BOOK_URL = f"https://raw.githubusercontent.com/{BOOK_REPO}/{BOOK_COMMIT}/{BOOK_FILE}.zip"
BOOK_PATH = Path("runs/tools/books") / f"{BOOK_FILE}.zip"
BOOK_LICENSE = "CC0-1.0"

N_OPENINGS = 200
LIMIT_CP = 50
OPENING_PLY = 8
SEED = 0
DEFAULT_OUT = Path(__file__).with_name("openings_200.json")


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def fetch_book(path: Path = BOOK_PATH) -> Path:
    """책 zip을 고정 커밋에서 받는다 (이미 있으면 그대로). SHA-256이 다르면 거부한다."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BOOK_URL) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BOOK_ZIP_SHA256:
        raise ValueError(f"오프닝 책 SHA-256이 다르다: {digest} (고정 {BOOK_ZIP_SHA256})")
    return path


def read_epd_lines(lines: Iterable[str]) -> list[str]:
    """EPD 줄 → 중복 없는 FEN 목록 (첫 등장 순서). 수 정보(반수 시계, 수 번호)가 없으면 '0 5'를 붙인다."""
    fens, seen = [], set()
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 4:
            continue
        key = " ".join(fields[:4])
        if key in seen:
            continue
        seen.add(key)
        clocks = fields[4:6] if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit() else ["0", "5"]
        fens.append(" ".join([*fields[:4], *clocks]))
    return fens


def read_book(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive, archive.open(BOOK_FILE) as raw:
        return read_epd_lines(io.TextIOWrapper(raw, encoding="ascii"))


def playable(fen: str) -> bool:
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    return board.is_valid() and board.turn == chess.WHITE and not board.is_game_over()


def select(fens: list[str], evaluate: Callable[[chess.Board], chess.engine.PovScore], n: int = N_OPENINGS,
           limit_cp: int = LIMIT_CP, seed: int = SEED) -> tuple[list[dict], dict]:
    """해석 7의 3. evaluate(board) → 점수. 반환: (고른 오프닝, 집계)."""
    order = np.random.default_rng(seed).permutation(len(fens))
    chosen, counts = [], {"evaluated": 0, "unplayable": 0, "mate": 0, "out_of_range": 0}
    for i in order:
        if len(chosen) == n:
            break
        fen = fens[int(i)]
        if not playable(fen):
            counts["unplayable"] += 1
            continue
        board = chess.Board(fen)
        score = evaluate(board).white()
        counts["evaluated"] += 1
        if score.is_mate():
            counts["mate"] += 1
            continue
        cp = score.score()
        if abs(cp) > limit_cp:
            counts["out_of_range"] += 1
            continue
        chosen.append({"index": len(chosen), "fen": fen, "cp_white": cp, "book_row": int(i)})
    if len(chosen) < n:
        raise ValueError(f"조건에 맞는 국면이 {len(chosen)}개뿐이다 (필요 {n})")
    return chosen, counts


def load_openings(path: Path = DEFAULT_OUT) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["openings"]


def parse_range(text: str) -> list[int]:
    """'0-199' 또는 '0-9,20,30-39' → 번호 목록."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="오프닝 200개 선별 (평가 기준 문서 5.3절)")
    parser.add_argument("--depth", type=int, required=True, help="평가 깊이 (C4와 같은 값)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--book", type=Path, default=BOOK_PATH)
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("-n", type=int, default=N_OPENINGS)
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.match.stockfish import Stockfish

    args = build_parser().parse_args(argv)
    fens = read_book(fetch_book(args.book))
    print(f"책 {BOOK_FILE}: 중복 없는 국면 {len(fens):,}개", flush=True)
    started = datetime.datetime.now()
    with Stockfish(args.stockfish) as sf:
        openings, counts = select(fens, lambda board: sf.evaluate(board, args.depth)[1], n=args.n)
        meta = {
            "source": {"repo": BOOK_REPO, "commit": BOOK_COMMIT, "file": BOOK_FILE, "zip_sha256": BOOK_ZIP_SHA256,
                       "license": BOOK_LICENSE, "unique_positions": len(fens)},
            "stockfish": {"version": sf.version, "binary_sha256": hashlib.sha256(sf.binary.resolve().read_bytes()).hexdigest(),
                          "threads": 1, "hash_mb": sf.hash_mb},
            "rule": {"depth": args.depth, "limit_cp": LIMIT_CP, "seed": SEED, "n": args.n, "ply": OPENING_PLY},
            "counts": counts,
            "created": started.isoformat(timespec="seconds"),
            "seconds": round((datetime.datetime.now() - started).total_seconds(), 1),
        }
    args.out.write_text(json.dumps({"meta": meta, "openings": openings}, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
