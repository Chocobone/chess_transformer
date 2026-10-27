from collections import Counter
from pathlib import Path

import chess
import pytest
import torch

from src.model.actions import ordered_legal_moves
from src.model.lora import LoRALinear, inject_lora, load_adapter, save_adapter
from src.model.transformer import ActionValueTransformer, TransformerConfig
from src.play.engine import TEMPERATURE, Engine, ModelSpec, default_specs, parse_board
from src.train.policy import legal_action_tokens, policy_from_q

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)
SETTINGS = {"rank": 4, "alpha": 8, "dropout": 0.0, "targets": ["q", "k", "v", "o"]}
# 크기별 소형 베이스의 seed. 실제 체크포인트 대신 loader가 이 seed로 모델을 만든다
BASE_SEEDS = {"9M": 0, "136M": 1, "270M": 2}

FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
PROMOTION = "8/P7/8/8/8/8/8/k6K w - - 0 1"
EN_PASSANT = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
CASTLING = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def tiny_base(size: str) -> ActionValueTransformer:
    torch.manual_seed(BASE_SEEDS[size])
    return ActionValueTransformer(TINY).eval()


def loader(path: Path, device: torch.device) -> ActionValueTransformer:
    return tiny_base(Path(path).stem).to(device)


def make_adapter(path: Path, seed: int) -> Path:
    model = tiny_base("136M")
    inject_lora(model, SETTINGS["rank"], SETTINGS["alpha"], SETTINGS["dropout"], tuple(SETTINGS["targets"]))
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.1)
    save_adapter(model, path, SETTINGS)
    return path


@pytest.fixture
def specs(tmp_path) -> list[ModelSpec]:
    base = {size: Path(f"{size}.pt") for size in BASE_SEEDS}
    return [
        ModelSpec("base-9M", "베이스 (9M)", "9M", base["9M"]),
        ModelSpec("base-136M", "베이스 (136M)", "136M", base["136M"]),
        ModelSpec("base-270M", "베이스 (270M)", "270M", base["270M"]),
        ModelSpec("tal-136M", "탈 (136M)", "136M", base["136M"], make_adapter(tmp_path / "tal.pt", 10)),
        ModelSpec("karpov-136M", "카르포프 (136M)", "136M", base["136M"], make_adapter(tmp_path / "karpov.pt", 11)),
    ]


@pytest.fixture
def engine(specs) -> Engine:
    return Engine(specs, device="cpu", loader=loader)


def reference(model: ActionValueTransformer, fen: str) -> dict[str, tuple[float, float]]:
    """학습 코드 경로로 직접 계산한 수별 (π, q)."""
    tokens, owner, _ = legal_action_tokens([fen])
    with torch.inference_mode():
        q = model.win_prob(tokens)
        log_pi, _ = policy_from_q(q, owner, TEMPERATURE)
    moves = ordered_legal_moves(chess.Board(fen))
    return {move.uci(): (float(log_pi[0, i].exp()), float(q[i])) for i, move in enumerate(moves)}


def all_moves(engine: Engine, model_id: str, fen: str) -> dict[str, tuple[float, float]]:
    analysis = engine.analyze(model_id, fen, top_k=1000)
    return {c.uci: (c.prob, c.q) for c in analysis.candidates}


def adapter_model(spec: ModelSpec) -> ActionValueTransformer:
    model = tiny_base("136M")
    load_adapter(model, spec.adapter)
    return model.eval()


def test_policy_matches_training_code(engine, specs):
    by_id = {spec.id: spec for spec in specs}
    for fen in (chess.STARTING_FEN, MIDDLEGAME):
        assert all_moves(engine, "base-270M", fen) == reference(tiny_base("270M"), fen)
        assert all_moves(engine, "tal-136M", fen) == reference(adapter_model(by_id["tal-136M"]), fen)
        probs = [prob for prob, _ in all_moves(engine, "base-9M", fen).values()]
        assert sum(probs) == pytest.approx(1.0, abs=1e-5)


def test_choose_returns_legal_moves(engine):
    generator = torch.Generator().manual_seed(0)
    for fen in (chess.STARTING_FEN, PROMOTION, EN_PASSANT, CASTLING, MIDDLEGAME):
        board = chess.Board(fen)
        for model_id in ("base-136M", "tal-136M", "base-9M"):
            for _ in range(5):
                move, analysis = engine.choose(model_id, fen, generator)
                assert chess.Move.from_uci(move.uci) in board.legal_moves
                assert move.san == board.san(chess.Move.from_uci(move.uci))
                assert analysis.legal_count == board.legal_moves.count()


def test_sampling_is_reproducible(engine):
    def play(seed: int) -> list[str]:
        generator = torch.Generator().manual_seed(seed)
        return [engine.choose("tal-136M", MIDDLEGAME, generator)[0].uci for _ in range(30)]

    assert play(7) == play(7)
    assert play(7) != play(8)


def test_sampling_follows_policy(engine):
    generator = torch.Generator().manual_seed(0)
    draws = 2000
    counts = Counter(engine.choose("base-136M", MIDDLEGAME, generator)[0].uci for _ in range(draws))
    for uci, (prob, _) in all_moves(engine, "base-136M", MIDDLEGAME).items():
        assert abs(counts[uci] / draws - prob) < 0.05


def test_adapter_switching(engine, specs):
    by_id = {spec.id: spec for spec in specs}
    expected = {
        "tal-136M": reference(adapter_model(by_id["tal-136M"]), MIDDLEGAME),
        "karpov-136M": reference(adapter_model(by_id["karpov-136M"]), MIDDLEGAME),
        "base-136M": reference(tiny_base("136M"), MIDDLEGAME),
    }
    assert expected["tal-136M"] != expected["karpov-136M"] != expected["base-136M"]
    for model_id in ("tal-136M", "karpov-136M", "tal-136M", "base-136M", "karpov-136M", "base-9M", "tal-136M", "base-136M"):
        result = all_moves(engine, model_id, MIDDLEGAME)
        if model_id in expected:
            assert result == expected[model_id], model_id


def test_only_one_size_is_loaded(engine):
    engine.analyze("base-9M", MIDDLEGAME)
    assert engine._size == "9M" and engine._zero is None
    engine.analyze("tal-136M", MIDDLEGAME)
    assert engine._size == "136M" and engine._active == "tal-136M"
    engine.analyze("base-270M", MIDDLEGAME)
    assert engine._size == "270M" and engine._active is None


def test_candidates_are_sorted_and_truncated(engine):
    analysis = engine.analyze("tal-136M", MIDDLEGAME, top_k=5)
    probs = [c.prob for c in analysis.candidates]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)
    assert analysis.win_prob == max(q for _, q in all_moves(engine, "tal-136M", MIDDLEGAME).values())

    analysis = engine.analyze("tal-136M", chess.STARTING_FEN, top_k=100)
    assert len(analysis.candidates) == analysis.legal_count == 20


def test_errors(engine):
    with pytest.raises(ValueError):
        engine.analyze("smyslov-136M", MIDDLEGAME)
    with pytest.raises(ValueError):
        engine.analyze("base-136M", FOOLS_MATE)
    with pytest.raises(ValueError):
        engine.analyze("base-136M", "not a fen")
    with pytest.raises(ValueError):
        parse_board("8/8/8/8/8/8/8/8 w - - 0 1")  # 왕이 없음
    with pytest.raises(ValueError):
        engine.compare(["base-136M", "nope"], MIDDLEGAME)


def test_compare(engine):
    ids = ["karpov-136M", "base-136M", "tal-136M"]
    results = engine.compare(ids, MIDDLEGAME, top_k=3)
    assert [r.model_id for r in results] == ids
    for result in results:
        assert result == engine.analyze(result.model_id, MIDDLEGAME, top_k=3)


def test_mismatched_adapter_settings(tmp_path, specs):
    model = tiny_base("136M")
    inject_lora(model, rank=2, alpha=4)
    save_adapter(model, tmp_path / "other.pt", {**SETTINGS, "rank": 2, "alpha": 4})
    other = ModelSpec("smyslov-136M", "스미슬로프 (136M)", "136M", Path("136M.pt"), tmp_path / "other.pt")
    with pytest.raises(ValueError):
        Engine([*specs, other], device="cpu", loader=loader)


def test_default_specs(tmp_path):
    base_dir, runs_dir = tmp_path / "base_model", tmp_path / "runs"
    base_dir.mkdir()
    for size in ("9M", "136M"):
        (base_dir / f"{size}.pt").touch()
    for player in ("tal", "smyslov"):
        (runs_dir / f"{player}_136M").mkdir(parents=True)
        (runs_dir / f"{player}_136M" / "adapter.pt").touch()

    specs = default_specs(base_dir, runs_dir)
    assert [spec.id for spec in specs] == ["base-9M", "base-136M", "tal-136M", "smyslov-136M"]
    assert specs[2].adapter == runs_dir / "tal_136M" / "adapter.pt" and specs[2].label == "탈 (136M)"

    (base_dir / "136M.pt").unlink()
    assert [spec.id for spec in default_specs(base_dir, runs_dir)] == ["base-9M"]
