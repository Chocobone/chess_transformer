"""Task #13 1단계: Stockfish 연결, 승률 변환, 오프닝 선별."""

import io
import json
import math
import zipfile
from pathlib import Path

import chess
import chess.engine
import pytest

from src.match import openings
from src.match.stockfish import WIN_K, Stockfish, cp_to_win, find_binary, win_prob

MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8#

try:
    find_binary()
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False
needs_sf = pytest.mark.skipif(not HAVE_SF, reason="Stockfish 실행 파일이 없다 (gpu_train/install_stockfish.sh)")


def pov(score: chess.engine.Score, turn: chess.Color = chess.WHITE) -> chess.engine.PovScore:
    return chess.engine.PovScore(score, turn)


# ---- 승률 변환 ----

def test_cp_to_win_values():
    assert cp_to_win(0) == 0.5
    assert cp_to_win(100) == pytest.approx(1 / (1 + math.exp(-WIN_K * 100)))
    assert cp_to_win(100) == pytest.approx(0.5910, abs=1e-4)  # 손 계산: 1 / (1 + 0.69196)
    for cp in (10, 57, 300, 1500):
        assert cp_to_win(cp) + cp_to_win(-cp) == pytest.approx(1.0)
    assert cp_to_win(50) < cp_to_win(51)


def test_win_prob_pov_and_mate():
    score = pov(chess.engine.Cp(120), chess.WHITE)
    assert win_prob(score, chess.WHITE) == pytest.approx(cp_to_win(120))
    assert win_prob(score, chess.BLACK) == pytest.approx(cp_to_win(-120))
    assert win_prob(pov(chess.engine.Mate(3)), chess.WHITE) == 1.0
    assert win_prob(pov(chess.engine.Mate(3)), chess.BLACK) == 0.0
    assert win_prob(pov(chess.engine.Mate(-2)), chess.WHITE) == 0.0
    assert win_prob(pov(chess.engine.Mate(-0)), chess.WHITE) == 0.0  # 이미 메이트당함
    assert win_prob(pov(chess.engine.MateGiven), chess.WHITE) == 1.0


# ---- 오프닝 ----

def test_read_epd_lines_dedup_and_clocks():
    lines = [
        "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5\n",
        "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 7\n",  # 앞 4필드가 같다 → 중복
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - c0 \"x\";\n",  # 수 정보 없음 → 0 5
        "\n",
    ]
    fens = openings.read_epd_lines(lines)
    assert fens == [
        "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 5",
    ]


def test_read_book_from_zip(tmp_path):
    path = tmp_path / "book.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(openings.BOOK_FILE, MIDDLEGAME + "\n" + MIDDLEGAME + "\n")
    assert openings.read_book(path) == [MIDDLEGAME]


def test_playable():
    assert openings.playable(MIDDLEGAME)
    assert not openings.playable(MIDDLEGAME.replace(" w ", " b "))  # 흑 차례
    assert not openings.playable("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")  # 메이트
    assert not openings.playable("not a fen")


def fake_evaluate(values: dict[str, chess.engine.Score]):
    def evaluate(board: chess.Board) -> chess.engine.PovScore:
        return pov(values[board.fen()], chess.WHITE)
    return evaluate


def make_fens(n: int) -> list[str]:
    """서로 다른 합법 국면 n개 (백 차례). 흑 킹 h8, 백 폰 a2에 백 킹 위치만 바꾼다."""
    fens = []
    for square in range(chess.A3, chess.H8):
        board = chess.Board(None)
        board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(square, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over():
            fens.append(board.fen())
        if len(fens) == n:
            break
    assert len(fens) == n
    return fens


def test_select_filters_and_is_reproducible():
    fens = make_fens(40)
    values = {}
    for i, fen in enumerate(fens):
        values[fen] = (chess.engine.Mate(2) if i % 7 == 0 else chess.engine.Cp(-50 if i % 2 else 51 + i))
    chosen, counts = openings.select(fens, fake_evaluate(values), n=5, limit_cp=50, seed=0)
    again, _ = openings.select(fens, fake_evaluate(values), n=5, limit_cp=50, seed=0)
    assert chosen == again
    assert [c["index"] for c in chosen] == list(range(5))
    assert all(c["cp_white"] == -50 for c in chosen)  # |cp| ≤ 50만 (51 이상, 메이트는 빠진다)
    assert counts["evaluated"] == 5 + counts["mate"] + counts["out_of_range"]
    other, _ = openings.select(fens, fake_evaluate(values), n=5, limit_cp=50, seed=1)
    assert [c["fen"] for c in other] != [c["fen"] for c in chosen]


def test_select_raises_when_not_enough():
    fens = make_fens(10)
    values = {fen: chess.engine.Cp(300) for fen in fens}
    with pytest.raises(ValueError):
        openings.select(fens, fake_evaluate(values), n=1)


def test_parse_range():
    assert openings.parse_range("0-3") == [0, 1, 2, 3]
    assert openings.parse_range("0-1,5,8-9") == [0, 1, 5, 8, 9]


def test_committed_openings_file():
    """커밋한 오프닝 200개: 합법, 백 차례, 8 ply 뒤(수 번호 5), FEN 중복 없음, |cp| ≤ 50, 번호 순서."""
    path = openings.DEFAULT_OUT
    if not path.exists():
        pytest.skip("오프닝 파일을 아직 만들지 않았다")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["openings"]
    assert len(rows) == openings.N_OPENINGS
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert len({openings.fen_key(r["fen"]) for r in rows}) == len(rows)
    for r in rows:
        board = chess.Board(r["fen"])
        assert board.is_valid() and board.turn == chess.WHITE and not board.is_game_over()
        assert board.fullmove_number == 5
        assert abs(r["cp_white"]) <= openings.LIMIT_CP
    meta = data["meta"]
    assert meta["source"]["commit"] == openings.BOOK_COMMIT and meta["source"]["license"] == "CC0-1.0"
    assert meta["stockfish"]["version"] == "Stockfish 19"


# ---- Stockfish (실행 파일이 있을 때) ----

@needs_sf
def test_stockfish_version_and_determinism():
    with Stockfish() as sf:
        assert sf.version == "Stockfish 19"
        board = chess.Board(MIDDLEGAME)
        first = sf.move(board, nodes=5000)
        assert first in board.legal_moves
        sf.move(chess.Board(), nodes=5000)  # 다른 국면을 사이에 넣어도
        assert sf.move(board, nodes=5000) == first  # 해시를 비우므로 같은 수


@needs_sf
def test_stockfish_evaluate_best_and_restricted():
    with Stockfish() as sf:
        board = chess.Board(MATE_IN_1)
        best, score = sf.evaluate(board, depth=8)
        assert best == chess.Move.from_uci("a1a8")
        assert win_prob(score, board.turn) == 1.0
        other = chess.Move.from_uci("g1f1")
        move, restricted = sf.evaluate(board, depth=8, only=other)
        assert move == other
        assert win_prob(restricted, board.turn) < 1.0


@needs_sf
def test_stockfish_rejects_other_version():
    with pytest.raises(RuntimeError):
        Stockfish(require_version="Stockfish 0")
