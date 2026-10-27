import io
import random
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch
from fastapi.testclient import TestClient

from src.model.lora import LoRALinear, inject_lora, save_adapter
from src.model.transformer import ActionValueTransformer, TransformerConfig
from src.play.engine import Analysis, Candidate, Engine, ModelSpec
from src.play.game import Game, TurnError
from src.play.server import create_app

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)
SETTINGS = {"rank": 4, "alpha": 8, "dropout": 0.0, "targets": ["q", "k", "v", "o"]}
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def tiny_base(seed: int) -> ActionValueTransformer:
    torch.manual_seed(seed)
    return ActionValueTransformer(TINY).eval()


def loader(path: Path, device: torch.device) -> ActionValueTransformer:
    return tiny_base({"9M": 0, "136M": 1}[Path(path).stem]).to(device)


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> Engine:
    adapter = tmp_path_factory.mktemp("adapters") / "tal.pt"
    model = tiny_base(1)
    inject_lora(model, SETTINGS["rank"], SETTINGS["alpha"], SETTINGS["dropout"], tuple(SETTINGS["targets"]))
    torch.manual_seed(10)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.1)
    save_adapter(model, adapter, SETTINGS)
    specs = [
        ModelSpec("base-9M", "베이스 (9M)", "9M", Path("9M.pt")),
        ModelSpec("base-136M", "베이스 (136M)", "136M", Path("136M.pt")),
        ModelSpec("tal-136M", "탈 (136M)", "136M", Path("136M.pt"), adapter),
    ]
    return Engine(specs, device="cpu", loader=loader)


def client_for(engine: Engine, seed: int = 0) -> TestClient:
    return TestClient(create_app(engine, seed))


@pytest.fixture
def client(engine) -> TestClient:
    return client_for(engine)


def replay(moves: list[dict]) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push_uci(move["uci"])
    return board


def fake_move(board: chess.Board, uci: str) -> tuple[Candidate, Analysis]:
    """Game 단위 테스트에서 모델 대신 정해진 수를 두기 위한 후보."""
    move = chess.Move.from_uci(uci)
    candidate = Candidate(uci, board.san(move), 1.0, 0.5)
    return candidate, Analysis("base-9M", [candidate], 0.5, board.legal_moves.count())


def test_models(client):
    body = client.get("/api/models").json()
    assert [m["id"] for m in body["models"]] == ["base-9M", "base-136M", "tal-136M"]
    assert body["compare"] == ["base-136M", "tal-136M"]  # 없는 카르포프·스미슬로프는 빠진다


def test_play_a_game(client):
    rng = random.Random(0)
    state = client.post("/api/game", json={"model": "tal-136M", "color": "white"}).json()
    assert state["user_color"] == "white" and state["turn"] == "white" and len(state["legal_moves"]) == 20
    for _ in range(40):
        state = client.post("/api/game/move", json={"uci": rng.choice(state["legal_moves"])}).json()
        assert state["legal_moves"] == [] and state["moves"][-1]["by"] == "user"
        if state["outcome"]:
            break
        state = client.post("/api/game/engine").json()
        last = state["moves"][-1]
        assert last["by"] == "model" and 0 < last["prob"] <= 1
        analysis = state["last_analysis"]
        assert analysis["model_id"] == "tal-136M" and 1 <= len(analysis["candidates"]) <= 5
        assert last["analysis"] == analysis and state["moves"][-2]["analysis"] is None
        board = replay(state["moves"])
        assert state["fen"] == board.fen() == last["fen"]
        if state["outcome"]:
            break
        assert set(state["legal_moves"]) == {m.uci() for m in board.legal_moves}
    assert client.get("/api/game").json() == state


def test_user_plays_black(client):
    state = client.post("/api/game", json={"model": "base-9M", "color": "black"}).json()
    assert state["turn"] == "white" and state["legal_moves"] == []
    assert client.post("/api/game/move", json={"uci": "e7e5"}).status_code == 409
    state = client.post("/api/game/engine").json()
    assert state["turn"] == "black" and len(state["moves"]) == 1 and state["legal_moves"]
    assert client.post("/api/game/engine").status_code == 409


def test_random_color(engine):
    colors = set()
    for seed in range(10):
        state = client_for(engine, seed).post("/api/game", json={"model": "base-9M", "color": "random"}).json()
        colors.add(state["user_color"])
    assert colors == {"white", "black"}


def test_rejects_bad_requests(client):
    assert client.get("/api/game").status_code == 404
    assert client.post("/api/game/move", json={"uci": "e2e4"}).status_code == 404
    assert client.post("/api/game", json={"model": "nope", "color": "white"}).status_code == 400
    assert client.post("/api/game", json={"model": "base-9M", "color": "red"}).status_code == 400
    client.post("/api/game", json={"model": "base-9M", "color": "white"})
    assert client.post("/api/game/move", json={"uci": "e2e5"}).status_code == 400
    assert client.post("/api/game/move", json={"uci": "zz"}).status_code == 400
    assert client.post("/api/game/engine").status_code == 409
    client.post("/api/game/move", json={"uci": "e2e4"})
    assert client.post("/api/game/move", json={"uci": "d2d4"}).status_code == 409


def test_undo(client):
    client.post("/api/game", json={"model": "base-9M", "color": "white"})
    assert client.post("/api/game/undo").json()["moves"] == []  # 첫 수 전에는 그대로
    client.post("/api/game/move", json={"uci": "e2e4"})
    client.post("/api/game/engine")
    client.post("/api/game/move", json={"uci": "d2d4"})
    client.post("/api/game/engine")
    state = client.post("/api/game/undo").json()
    assert [m["uci"] for m in state["moves"]][0] == "e2e4" and len(state["moves"]) == 2
    assert state["turn"] == "white" and state["legal_moves"]
    # 모델 수를 받기 전이면 사용자 수 하나만 되돌린다
    client.post("/api/game/move", json={"uci": "g1f3"})
    assert len(client.post("/api/game/undo").json()["moves"]) == 2


def test_undo_when_user_is_black(client):
    client.post("/api/game", json={"model": "base-9M", "color": "black"})
    client.post("/api/game/engine")
    assert len(client.post("/api/game/undo").json()["moves"]) == 1  # 모델의 첫 수만 있으면 그대로
    state = client.post("/api/game/move", json={"uci": client.get("/api/game").json()["legal_moves"][0]}).json()
    state = client.post("/api/game/engine").json()
    state = client.post("/api/game/undo").json()
    assert len(state["moves"]) == 1 and state["turn"] == "black"


def test_resign(client):
    client.post("/api/game", json={"model": "base-9M", "color": "black"})
    state = client.post("/api/game/resign").json()
    assert state["outcome"] == {"result": "1-0", "reason": "resign", "winner": "white"}
    assert client.post("/api/game/engine").status_code == 409
    assert client.post("/api/game/resign").status_code == 409
    assert client.post("/api/game/undo").json()["outcome"] == {"result": "1-0", "reason": "resign", "winner": "white"}


def test_pgn(client):
    rng = random.Random(1)
    state = client.post("/api/game", json={"model": "tal-136M", "color": "white"}).json()
    for _ in range(6):
        state = client.post("/api/game/move", json={"uci": rng.choice(state["legal_moves"])}).json()
        state = client.post("/api/game/engine").json()
    client.post("/api/game/resign")
    response = client.get("/api/game/pgn")
    assert "attachment" in response.headers["content-disposition"]
    game = chess.pgn.read_game(io.StringIO(response.text))
    assert [m.uci() for m in game.mainline_moves()] == [m["uci"] for m in state["moves"]]
    assert game.headers["Result"] == "0-1" and game.headers["Black"] == "탈 (136M)"


def test_analyze(client):
    body = client.post("/api/analyze", json={"fen": chess.STARTING_FEN}).json()
    assert [r["model_id"] for r in body["results"]] == ["base-136M", "tal-136M"]
    assert all(len(r["candidates"]) == 3 and r["label"] for r in body["results"])
    body = client.post("/api/analyze", json={"fen": chess.STARTING_FEN, "models": ["base-9M"]}).json()
    assert [r["model_id"] for r in body["results"]] == ["base-9M"]
    assert client.post("/api/analyze", json={"fen": FOOLS_MATE}).status_code == 400
    assert client.post("/api/analyze", json={"fen": chess.STARTING_FEN, "models": ["nope"]}).status_code == 400


def test_same_seed_same_replies(engine):
    def play(seed: int) -> list[str]:
        client = client_for(engine, seed)
        client.post("/api/game", json={"model": "tal-136M", "color": "white"})
        for uci in ("e2e4", "g1f3", "f1c4", "d2d3"):
            if client.post("/api/game/move", json={"uci": uci}).status_code != 200:
                break
            client.post("/api/game/engine")
        return [m["uci"] for m in client.get("/api/game").json()["moves"]]

    assert play(3) == play(3)


# --- Game 단위 테스트: 모델 수를 정해 두고 종료 판정을 본다 ---


def test_checkmate():
    game = Game("base-9M", "베이스 (9M)", chess.BLACK)
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        if game.user_to_move:
            game.push_user(uci)
        else:
            game.push_model(*fake_move(game.board, uci))
    assert game.outcome() == {"result": "0-1", "reason": "checkmate", "winner": "black"}
    state = game.state()
    assert state["check"] == "e1" and state["legal_moves"] == []
    candidate = Candidate("e2e4", "e4", 1.0, 0.5)
    with pytest.raises(TurnError):
        game.push_model(candidate, Analysis("base-9M", [candidate], 0.5, 20))


def test_threefold_repetition():
    game = Game("base-9M", "베이스 (9M)", chess.WHITE)
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in shuffle * 2:
        assert game.outcome() is None
        if game.user_to_move:
            game.push_user(uci)
        else:
            game.push_model(*fake_move(game.board, uci))
    assert game.outcome() == {"result": "1/2-1/2", "reason": "threefold_repetition", "winner": None}
    assert game.records[-1].repetition and not game.records[-2].repetition
    with pytest.raises(TurnError):
        game.push_user("g1f3")
    game.undo()  # 무르면 다시 진행된다
    assert game.outcome() is None


def test_repetition_is_not_claimed_early():
    """다음 한 수로 3회 반복이 될 수 있을 뿐인 국면은 끝나지 않는다."""
    game = Game("base-9M", "베이스 (9M)", chess.WHITE)
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in (shuffle * 2)[:7]:
        if game.user_to_move:
            game.push_user(uci)
        else:
            game.push_model(*fake_move(game.board, uci))
    assert game.board.can_claim_threefold_repetition() and game.outcome() is None


def test_promotion():
    game = Game("base-9M", "베이스 (9M)", chess.WHITE)
    game.board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    with pytest.raises(ValueError):
        game.push_user("a7a8")  # 승진 기물 없음
    game.push_user("a7a8q")
    assert game.records[-1].san == "a8=Q+"
