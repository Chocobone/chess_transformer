"""Task #13 3단계: C4 수 품질 탐색 (가짜 Stockfish로 캐시·손실·자르기 확인)."""

import json
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

from src.match import quality
from src.match.stockfish import cp_to_win
from src.model.actions import ordered_legal_moves
from src.players import PLAYERS

FENS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/pp3pp1/2p1p2p/3p4/3P4/3Q1NP1/PPP1PP1P/RNB1KB1R w KQkq - 0 5",
    "8/8/8/4k3/8/8/4K3/4R3 b - - 0 60",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
]
PLAYER_OF = [0, 0, 1, 1, 2, 2]


def best_of(board: chess.Board) -> chess.Move:
    return min(board.legal_moves, key=lambda m: m.uci())


class FakeSF:
    """최선수 = UCI 문자열이 가장 작은 수, 점수 100cp. 다른 수는 100 − 10 × (UCI 순위)cp.
    CHEAT 국면에서는 허용 탐색 점수가 최선수보다 높게 나온다 (탐색 불일치 → 음수 손실)."""
    calls: list = []
    cheat: set = set()

    def __init__(self, binary=None):
        self.version, self.hash_mb, self.binary = "fake", 16, Path(__file__)

    def evaluate(self, board, depth, only=None):
        FakeSF.calls.append((board.fen(), only.uci() if only else None, depth))
        ranked = sorted(board.legal_moves, key=lambda m: m.uci())
        move = only or ranked[0]
        cp = 100 - 10 * ranked.index(move)
        if only is not None and board.fen() in FakeSF.cheat:
            cp = 150
        return move, chess.engine.PovScore(chess.engine.Cp(cp), board.turn)

    def close(self):
        pass


@pytest.fixture
def dump(tmp_path) -> tuple[Path, Path]:
    """FENS로 만든 작은 덤프와 val 기록. 모델 A는 1순위가 최선수, B는 두 번째 수, C는 세 번째 수."""
    root, out = tmp_path / "data", tmp_path / "dump"
    out.mkdir()
    counts, games, plies = [], [], []
    for p, name in enumerate(PLAYERS):
        (root / name).mkdir(parents=True)
        with open(root / name / "val.jsonl", "w", encoding="utf-8") as f:
            for i, fen in enumerate(FENS):
                if PLAYER_OF[i] == p:
                    f.write(json.dumps({"fen": fen, "game_id": 10 + i, "ply": 5 * i}) + "\n")
    for i, fen in enumerate(FENS):
        counts.append(chess.Board(fen).legal_moves.count())
        games.append(10 + i)
        plies.append(5 * i)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.savez(out / "positions.npz", player=np.array(PLAYER_OF, dtype=np.int8), game=np.array(games, dtype=np.int32),
             ply=np.array(plies, dtype=np.int16), offsets=offsets)
    for model, rank in (("A", 0), ("B", 1), ("C", 2)):
        log_pi = np.full(offsets[-1], -10.0, dtype=np.float32)
        for i, fen in enumerate(FENS):
            board = chess.Board(fen)
            ranked = sorted(board.legal_moves, key=lambda m: m.uci())
            slot = ordered_legal_moves(board).index(ranked[min(rank, len(ranked) - 1)])
            log_pi[offsets[i] + slot] = -0.1
        np.savez(out / f"{model}.npz", log_pi=log_pi)
    return out, root


def searcher(workers=2, depth=7):
    return quality.Searcher(workers, depth, factory=FakeSF)


def test_load_fens_and_top1(dump):
    out, root = dump
    fens = quality.load_fens(out, root)
    assert fens == FENS
    assert quality.top1_moves(out, "A", fens) == [best_of(chess.Board(f)).uci() for f in FENS]


def test_load_fens_rejects_mismatch(dump):
    out, root = dump
    with open(root / "tal" / "val.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"fen": chess.STARTING_FEN, "game_id": 99, "ply": 0}) + "\n")
    with pytest.raises(ValueError):
        quality.load_fens(out, root)


def test_compute_loss_cache_and_clip(dump, tmp_path):
    out, root = dump
    fens = quality.load_fens(out, root)
    FakeSF.calls, FakeSF.cheat = [], {FENS[1]}
    s = searcher()
    try:
        result = quality.compute(out, ["A", "B"], tmp_path / "q", s, fens)
    finally:
        s.close()
    # 최선수 탐색 6회 + B의 수 탐색 6회 (A는 모두 최선수라 추가 탐색 없음)
    assert len(FakeSF.calls) == 12 and all(c[2] == 7 for c in FakeSF.calls)
    a = np.load(tmp_path / "q" / "A.npz")
    b = np.load(tmp_path / "q" / "B.npz")
    assert a["loss"].tolist() == [0.0] * 6 and a["move_is_best"].all()
    expected = cp_to_win(100) - cp_to_win(90)
    for i in range(6):
        if i == 1:  # 탐색 불일치 → 음수 손실을 0으로 자른다
            assert b["loss"][i] == 0.0 and b["clipped"][i]
        elif chess.Board(FENS[i]).legal_moves.count() > 1:
            assert b["loss"][i] == pytest.approx(expected) and not b["clipped"][i]
    assert result["summary"]["B"]["clipped"] == 1

    # 다시 실행: 모델 C만 새로 탐색한다 (최선수와 B의 수는 캐시)
    FakeSF.calls = []
    s = searcher()
    try:
        quality.compute(out, ["A", "B", "C"], tmp_path / "q", s, fens)
    finally:
        s.close()
    assert len(FakeSF.calls) == 6 and all(c[1] is not None for c in FakeSF.calls)
    c = np.load(tmp_path / "q" / "C.npz")
    assert c["loss"][0] == pytest.approx(cp_to_win(100) - cp_to_win(80))
    best = [json.loads(line) for line in (tmp_path / "q" / "best.jsonl").read_text().splitlines()]
    assert len(best) == 6 and sorted(r["i"] for r in best) == list(range(6))


def test_score_json_mate_and_cp():
    pov = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
    assert quality.score_json(pov, chess.WHITE) == {"mate": 2}
    assert quality.score_json(pov, chess.BLACK) == {"mate": -2}
    assert quality.score_json(chess.engine.PovScore(chess.engine.Cp(30), chess.BLACK), chess.WHITE) == {"cp": -30}
