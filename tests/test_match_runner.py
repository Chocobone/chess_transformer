"""Task #13 2단계: 묶음 정책, 대국 종료 판정, 동시 대국 실행과 기록."""

import io
import json
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

from src.match import game as game_module
from src.match.game import GameSpec, MatchGame, pair_specs
from src.match.policy import BASE_ID, PolicyBank, check_order
from src.match.runner import SF_ID, run
from src.model.lora import LoRALinear, inject_lora, save_adapter
from src.model.transformer import ActionValueTransformer, TransformerConfig
from src.play.engine import Engine, ModelSpec
from src.match.stockfish import find_binary
from src.model.actions import ordered_legal_moves

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)
SETTINGS = {"rank": 4, "alpha": 8, "dropout": 0.0, "targets": ["q", "k", "v", "o"]}
MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
OPENINGS = [
    {"index": 0, "fen": "rnbqkbnr/pp3pp1/2p1p2p/3p4/3P4/3Q1NP1/PPP1PP1P/RNB1KB1R w KQkq - 0 5"},
    {"index": 1, "fen": "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"},
    {"index": 2, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 5"},
    {"index": 3, "fen": MIDDLEGAME.replace("- 4 4", "- 0 5")},
]

try:
    find_binary()
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False


def tiny_base() -> ActionValueTransformer:
    torch.manual_seed(1)
    return ActionValueTransformer(TINY).eval()


def loader(path: Path, device: torch.device) -> ActionValueTransformer:
    return tiny_base().to(device)


def make_adapter(path: Path, seed: int) -> Path:
    model = tiny_base()
    inject_lora(model, SETTINGS["rank"], SETTINGS["alpha"], SETTINGS["dropout"], tuple(SETTINGS["targets"]))
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.1)
    save_adapter(model, path, SETTINGS)
    return path


@pytest.fixture
def adapters(tmp_path) -> dict[str, Path]:
    return {"R1": make_adapter(tmp_path / "R1.pt", 10), "R8": make_adapter(tmp_path / "R8.pt", 11)}


@pytest.fixture
def bank(adapters) -> PolicyBank:
    return PolicyBank(Path("136M.pt"), adapters, device="cpu", loader=loader)


class FirstMove:
    """가짜 Stockfish: 행동 순서가 아닌 UCI 문자열 순서로 첫 합법수."""
    name = SF_ID
    headers = {"Engine": "fake", "Nodes": "0"}

    def __init__(self):
        self.calls = 0

    def moves(self, boards):
        self.calls += len(boards)
        return [min(board.legal_moves, key=lambda m: m.uci()) for board in boards]


# ---- 묶음 정책 ----

def test_policy_matches_play_engine(adapters):
    """대국 앱 Engine과 같은 π (같은 소형 베이스, 같은 어댑터)."""
    specs = [ModelSpec("base-136M", "b", "136M", Path("136M.pt")),
             ModelSpec("tal-136M", "t", "136M", Path("136M.pt"), adapters["R1"])]
    engine = Engine(specs, device="cpu", loader=lambda path, device: tiny_base().to(device))
    bank = PolicyBank(Path("136M.pt"), adapters, device="cpu", loader=loader)
    fens = [MIDDLEGAME, OPENINGS[0]["fen"], chess.STARTING_FEN]
    for bank_id, engine_id in ((BASE_ID, "base-136M"), ("R1", "tal-136M")):
        probs = bank.probabilities(bank_id, [chess.Board(f) for f in fens])
        for fen, pi in zip(fens, probs):
            analysis = engine.analyze(engine_id, fen, top_k=1000)
            expected = {c.uci: c.prob for c in analysis.candidates}
            assert len(expected) == len(pi)
            for move, p in zip(ordered_legal_moves(chess.Board(fen)), pi.tolist()):
                assert p == pytest.approx(expected[move.uci()], abs=1e-6)
            check_order(chess.Board(fen))


def test_policy_batch_and_chunk_independent(adapters):
    boards = [chess.Board(row["fen"]) for row in OPENINGS] + [chess.Board(MIDDLEGAME)]
    whole = PolicyBank(Path("136M.pt"), adapters, device="cpu", loader=loader)
    small = PolicyBank(Path("136M.pt"), adapters, device="cpu", loader=loader, max_rows=10)
    batched = whole.probabilities("R8", boards)
    chunked = small.probabilities("R8", boards)
    single = [whole.probabilities("R8", [b])[0] for b in boards]
    for a, b, c in zip(batched, chunked, single):
        assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)
        assert a.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_policy_switches_adapters(bank):
    board = [chess.Board(MIDDLEGAME)]
    base1 = bank.probabilities(BASE_ID, board)[0]
    r1 = bank.probabilities("R1", board)[0]
    base2 = bank.probabilities(BASE_ID, board)[0]
    assert not torch.allclose(base1, r1)
    assert torch.equal(base1, base2)
    with pytest.raises(ValueError):
        bank.probabilities("R9", board)


def test_policy_rejects_mismatched_settings(tmp_path, adapters):
    other = tiny_base()
    inject_lora(other, 2, 4, 0.0, ("q", "v"))
    save_adapter(other, tmp_path / "other.pt", {"rank": 2, "alpha": 4, "dropout": 0.0, "targets": ["q", "v"]})
    with pytest.raises(ValueError):
        PolicyBank(Path("136M.pt"), {"R1": adapters["R1"], "X": tmp_path / "other.pt"}, loader=loader)


# ---- 대국 상태와 종료 ----

def spec_for(fen: str, color=chess.WHITE) -> GameSpec:
    return GameSpec(0, fen, "R1", BASE_ID, color)


def push_uci(game: MatchGame, *ucis: str) -> None:
    for uci in ucis:
        game.push(chess.Move.from_uci(uci), None)


def test_seed_and_pairs():
    specs = pair_specs(OPENINGS, [0, 3], "R1", SF_ID)
    assert [(s.opening, s.model_color, s.seed) for s in specs] == [
        (0, chess.WHITE, 0), (0, chess.BLACK, 1), (3, chess.WHITE, 6), (3, chess.BLACK, 7)]
    assert (specs[0].white, specs[0].black) == ("R1", SF_ID)
    assert (specs[1].white, specs[1].black) == (SF_ID, "R1")


def test_termination_checkmate_and_score():
    game = MatchGame(spec_for("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", chess.BLACK))
    push_uci(game, "d8h4")
    assert game.outcome() == ("0-1", "checkmate")
    assert game.model_score() == 1.0  # 평가 대상이 흑


def test_termination_stalemate_and_insufficient():
    assert MatchGame(spec_for("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")).outcome() == ("1/2-1/2", "stalemate")
    assert MatchGame(spec_for("8/8/8/8/8/8/k7/K6b w - - 0 1")).outcome() == ("1/2-1/2", "insufficient_material")


def test_termination_threefold_actual_only():
    game = MatchGame(spec_for(chess.STARTING_FEN))
    push_uci(game, "g1f3", "g8f6", "f3g1", "f6g8")  # 시작 국면 2번째
    assert game.outcome() is None
    push_uci(game, "g1f3", "g8f6", "f3g1")
    assert game.outcome() is None  # 다음 수로 3회가 될 수 있을 뿐이다 (claim_draw 쓰지 않음)
    push_uci(game, "f6g8")
    assert game.outcome() == ("1/2-1/2", "threefold_repetition")


def test_termination_fifty_moves():
    game = MatchGame(spec_for("8/8/8/4k3/8/8/4K3/4R3 w - - 99 80"))
    assert game.outcome() is None
    push_uci(game, "e1a1")
    assert game.outcome() == ("1/2-1/2", "fifty_moves")


def test_termination_max_ply(monkeypatch):
    monkeypatch.setattr(game_module, "MAX_PLY", 12)  # 오프닝 8 ply + 4 ply
    game = MatchGame(spec_for(chess.STARTING_FEN))
    push_uci(game, "g1f3", "g8f6", "f3g1")
    assert game.outcome() is None and game.total_ply == 11
    push_uci(game, "f6g8")
    assert game.outcome() == ("1/2-1/2", "max_ply")
    assert game.model_score() == 0.5


def test_push_rejects_illegal():
    with pytest.raises(ValueError):
        push_uci(MatchGame(spec_for(chess.STARTING_FEN)), "e2e5")


# ---- 실행과 기록 ----

def read_index(out: Path) -> dict:
    rows = [json.loads(line) for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    return {(r["opening"], r["model_color"]): r for r in rows}


def read_pgns(out: Path) -> dict:
    games = {}
    with open(out / "games.pgn", encoding="utf-8") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            key = (int(game.headers["Opening"]), game.headers["ModelColor"])
            games[key] = (game.headers, [m.uci() for m in game.mainline_moves()])
    return games


def test_run_c1_reproducible_and_independent_of_concurrency(tmp_path, bank):
    specs = pair_specs(OPENINGS, [0, 1, 2, 3], "R1", BASE_ID)
    run(specs, bank, tmp_path / "a", concurrent=1, log_every=0)
    run(specs, bank, tmp_path / "b", concurrent=8, log_every=0)
    run(specs, bank, tmp_path / "c", concurrent=8, log_every=0)
    a, b, c = read_pgns(tmp_path / "a"), read_pgns(tmp_path / "b"), read_pgns(tmp_path / "c")
    assert len(a) == 8
    assert {k: v[1] for k, v in a.items()} == {k: v[1] for k, v in b.items()} == {k: v[1] for k, v in c.items()}


def test_run_records_and_pgn_replay(tmp_path, bank):
    out = tmp_path / "m"
    specs = pair_specs(OPENINGS, [0, 1], "R8", SF_ID)
    summary = run(specs, bank, out, opponent=FirstMove(), concurrent=4, log_every=0)
    assert summary["played"] == 4 and summary["sf_moves"] > 0 and summary["model_moves"] > 0
    index, pgns = read_index(out), read_pgns(out)
    assert set(index) == set(pgns) == {(0, "white"), (0, "black"), (1, "white"), (1, "black")}
    for key, (headers, moves) in pgns.items():
        row = index[key]
        assert headers["FEN"] == OPENINGS[key[0]]["fen"] and headers["SetUp"] == "1"
        assert headers["Seed"] == str(row["seed"]) == str(2 * key[0] + (key[1] == "black"))
        assert headers["Termination"] == row["termination"] and headers["Result"] == row["result"]
        assert headers["Engine"] == "fake" and headers["Concurrent"] == "4"
        model_white = key[1] == "white"
        assert (headers["White"], headers["Black"]) == (("R8", SF_ID) if model_white else (SF_ID, "R8"))
        assert headers["WhiteSha256" if model_white else "BlackSha256"] == bank.sha256["R8"]
        board = chess.Board(headers["FEN"])
        for uci in moves:
            board.push_uci(uci)
        replay = MatchGame(GameSpec(key[0], headers["FEN"], "R8", SF_ID, chess.WHITE if model_white else chess.BLACK))
        for move in board.move_stack:
            replay.push(move, None)
        assert replay.outcome() == (row["result"], row["termination"])
        assert len(moves) == row["ply"] == int(headers["PlyCount"])
    runs = (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(runs) == 1


def test_run_resume_skips_done(tmp_path, bank):
    specs = pair_specs(OPENINGS, [0, 1, 2], "R1", BASE_ID)
    out = tmp_path / "r"
    run(specs[:2], bank, out, concurrent=2, log_every=0)  # 중간에 끊긴 것처럼 일부만
    summary = run(specs, bank, out, concurrent=4, log_every=0)
    assert summary["skipped"] == 2 and summary["played"] == 4
    index = read_index(out)
    assert len(index) == 6 and len((out / "index.jsonl").read_text().splitlines()) == 6
    fresh = tmp_path / "f"
    run(specs, bank, fresh, concurrent=4, log_every=0)
    assert {k: v[1] for k, v in read_pgns(out).items()} == {k: v[1] for k, v in read_pgns(fresh).items()}
    again = run(specs, bank, out, concurrent=4, log_every=0)
    assert again["played"] == 0 and again["skipped"] == 6


def test_run_requires_opponent_for_sf(tmp_path, bank):
    with pytest.raises(ValueError):
        run(pair_specs(OPENINGS, [0], "R1", SF_ID), bank, tmp_path / "x", log_every=0)


@pytest.mark.skipif(not HAVE_SF, reason="Stockfish 실행 파일이 없다")
def test_run_with_real_stockfish_pool_is_worker_independent(tmp_path, bank):
    from src.match.runner import StockfishPool
    specs = pair_specs(OPENINGS, [0, 1], "R1", SF_ID)
    results = []
    for workers in (1, 3):
        pool = StockfishPool(nodes=200, workers=workers)
        try:
            run(specs, bank, tmp_path / f"w{workers}", opponent=pool, concurrent=4, log_every=0)
        finally:
            pool.close()
        pgns = read_pgns(tmp_path / f"w{workers}")
        assert all(h["Engine"] == "Stockfish 19" and h["Nodes"] == "200" for h, _ in pgns.values())
        results.append({k: v[1] for k, v in pgns.items()})
    assert results[0] == results[1]
