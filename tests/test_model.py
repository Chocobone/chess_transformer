import chess
import numpy as np
import pytest
import torch

from src.model.actions import ACTION_TO_MOVE, MOVE_TO_ACTION, NUM_ACTIONS, ordered_legal_moves
from src.model.convert import convert, key_mapping
from src.model.tokenizer import SEQUENCE_LENGTH, tokenize
from src.model.transformer import (
    ActionValueTransformer,
    TransformerConfig,
    config_for,
    encode_position,
    load_base,
    save_base,
)

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)

# 문자 → 토큰 번호 (원본 _CHARACTERS의 순서)
D = 30  # '.'
W, B_SIDE = 29, 11
P_, N_, R_, K_, Q_ = 18, 19, 20, 21, 22  # 흑 기물 p n r k q
WP, WB, WN, WR, WQ, WK = 23, 24, 25, 26, 27, 28  # 백 기물 P B N R Q K
BB = 11  # 흑 비숍 'b'는 파일 'b'와 같은 토큰이다


def test_tokenize_starting_position():
    back_black = [R_, N_, BB, Q_, K_, BB, N_, R_]
    back_white = [WR, WN, WB, WQ, WK, WB, WN, WR]
    expected = (
        [W] + back_black + 8 * [P_] + 32 * [D] + 8 * [WP] + back_white
        + [WK, WQ, K_, Q_]  # 캐슬링 KQkq
        + [D, D]  # 앙파상 없음
        + [0, D, D]  # 반수 0
        + [1, D, D]  # 전수 1
    )
    tokens = tokenize(chess.STARTING_FEN)
    assert tokens.shape == (SEQUENCE_LENGTH,)
    assert tokens.tolist() == expected


def test_tokenize_en_passant_partial_castling_and_counters():
    tokens = tokenize("4k2r/8/8/3pP3/8/8/8/4K3 b k d6 12 105").tolist()
    assert tokens[0] == B_SIDE
    assert tokens[1:9] == [D, D, D, D, K_, D, D, R_]
    assert tokens[65:69] == [K_, D, D, D]  # 캐슬링 'k' + 패딩
    assert tokens[69:71] == [13, 6]  # 'd', '6'
    assert tokens[71:74] == [1, 2, D]
    assert tokens[74:77] == [1, 0, 5]


def test_tokenize_no_castling():
    assert tokenize("8/8/8/8/8/8/8/K6k w - - 0 1")[65:69].tolist() == 4 * [D]


def test_action_space():
    assert NUM_ACTIONS == 1968
    # a1에서 퀸이 가는 칸이 칸 번호 순으로 먼저, 그다음 나이트가 가는 칸
    assert ACTION_TO_MOVE[:3] == ["a1b1", "a1c1", "a1d1"]
    assert ACTION_TO_MOVE[21:23] == ["a1c2", "a1b3"]
    # 프로모션은 맨 뒤, 7랭크 h파일의 왼쪽 잡기가 마지막
    assert ACTION_TO_MOVE[-8:] == [
        "h7h8q", "h7h8r", "h7h8b", "h7h8n", "h7g8q", "h7g8r", "h7g8b", "h7g8n",
    ]
    assert sum(len(move) == 5 for move in ACTION_TO_MOVE) == 2 * (8 + 14) * 4


@pytest.mark.parametrize("move", ["e2e4", "g1f3", "e1g1", "e8c8", "e7e8q", "a2b1n", "h7g8r"])
def test_action_round_trip(move):
    assert ACTION_TO_MOVE[MOVE_TO_ACTION[move]] == move


def test_every_legal_move_has_an_action():
    board = chess.Board()
    for uci in ["e2e4", "d7d5", "e4d5", "g8f6", "f1b5", "c7c6", "d5c6", "a7a6", "c6b7", "a6b5"]:
        assert all(move.uci() in MOVE_TO_ACTION for move in board.legal_moves)
        board.push_uci(uci)
    promotions = [move.uci() for move in board.legal_moves if move.promotion]
    assert len(promotions) == 8 and all(uci in MOVE_TO_ACTION for uci in promotions)


def test_ordered_legal_moves_sorted_by_action():
    moves = ordered_legal_moves(chess.Board())
    actions = [MOVE_TO_ACTION[move.uci()] for move in moves]
    assert len(moves) == 20 and actions == sorted(actions)


def test_encode_position():
    tokens, moves = encode_position(chess.STARTING_FEN)
    assert tokens.shape == (20, 79)
    assert (tokens[:, :77] == tokenize(chess.STARTING_FEN)).all()
    assert tokens[:, 77].tolist() == [MOVE_TO_ACTION[move.uci()] for move in moves]
    assert (tokens[:, 78] == 0).all()


def test_forward_shapes_and_normalization():
    torch.manual_seed(0)
    model = ActionValueTransformer(TINY).eval()
    tokens = torch.from_numpy(encode_position(chess.STARTING_FEN)[0])
    with torch.no_grad():
        log_probs = model(tokens)
        win_probs = model.win_prob(tokens)
    assert log_probs.shape == (20, 128)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(20), atol=1e-5)
    assert win_probs.shape == (20,) and ((win_probs > 0) & (win_probs < 1)).all()


def test_return_token_is_ignored():
    """마지막 리턴 토큰은 shift로 버려지므로 출력에 영향을 주지 않는다."""
    torch.manual_seed(0)
    model = ActionValueTransformer(TINY).eval()
    tokens = torch.from_numpy(encode_position(chess.STARTING_FEN)[0])
    changed = tokens.clone()
    changed[:, -1] = 100
    with torch.no_grad():
        assert torch.equal(model(tokens), model(changed))


# 9M은 실제 체크포인트의 파라미터 수와 대조한 값이고, 나머지는 같은 구조에서 계산한 값이다
@pytest.mark.parametrize("size, expected", [("9M", 8_954_240), ("136M", 136_479_872), ("270M", 270_730_368)])
def test_parameter_counts(size, expected):
    config = config_for(size)
    model = ActionValueTransformer(config)
    assert sum(p.numel() for p in model.parameters()) == expected


def _random_haiku_params(config: TransformerConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = ActionValueTransformer(config).state_dict()
    params = {}
    for key, (haiku_key, transpose) in key_mapping(config).items():
        shape = tuple(state[key].shape)
        params[haiku_key] = (0.2 * rng.standard_normal(shape[::-1] if transpose else shape)).astype(np.float32)
    return params


def _haiku_forward(params: dict[str, np.ndarray], config: TransformerConfig, targets: np.ndarray) -> np.ndarray:
    """원본 `transformer_decoder`의 수식을 Haiku 파라미터 이름 그대로 numpy로 옮긴 기준 구현."""

    def name(base, index):
        return base if index == 0 else f"{base}_{index}"

    def layer_norm(x, index):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        key = name("layer_norm", index)
        return (x - mean) / np.sqrt(var + 1e-5) * params[f"{key}/scale"] + params[f"{key}/offset"]

    def softmax(x):
        x = np.exp(x - x.max(-1, keepdims=True))
        return x / x.sum(-1, keepdims=True)

    batch, length = targets.shape
    inputs = np.concatenate([np.zeros((batch, 1), dtype=targets.dtype), targets], axis=1)[:, :-1]
    h = params["embed/embeddings"][inputs] * np.sqrt(config.embedding_dim)
    h = h + params["embed_1/embeddings"][np.arange(length)]

    heads = config.num_heads
    per_head = config.embedding_dim // heads
    for i in range(config.num_layers):
        x = layer_norm(h, 2 * i)
        module = name("multi_head_dot_product_attention", i)
        q = (x @ params[f"{module}/linear/w"]).reshape(batch, length, heads, per_head)
        k = (x @ params[f"{module}/linear_1/w"]).reshape(batch, length, heads, per_head)
        v = (x @ params[f"{module}/linear_2/w"]).reshape(batch, length, heads, per_head)
        attention = softmax(np.einsum("bthd,bThd->bhtT", q, k) / np.sqrt(per_head))
        out = np.einsum("bhtT,bThd->bthd", attention, v).reshape(batch, length, -1)
        h = h + out @ params[f"{module}/linear_3/w"]

        x = layer_norm(h, 2 * i + 1)
        split_1 = x @ params[f"{name('linear', 3 * i)}/w"]
        split_2 = x @ params[f"{name('linear', 3 * i + 1)}/w"]
        gate = split_1 / (1 + np.exp(-split_1)) * split_2
        h = h + gate @ params[f"{name('linear', 3 * i + 2)}/w"]

    h = layer_norm(h, 2 * config.num_layers)
    head = name("linear", 3 * config.num_layers)
    logits = h @ params[f"{head}/w"] + params[f"{head}/b"]
    logits = logits - logits.max(-1, keepdims=True)
    return (logits - np.log(np.exp(logits).sum(-1, keepdims=True)))[:, -1]


def test_converted_model_matches_haiku_formulas():
    params = _random_haiku_params(TINY)
    model = convert(params, TINY).eval()
    tokens, _ = encode_position("r3k2r/pppq1ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 12")
    expected = _haiku_forward({k: v.astype(np.float64) for k, v in params.items()}, TINY, tokens)
    with torch.no_grad():
        actual = model(torch.from_numpy(tokens)).numpy()
    assert np.abs(actual - expected).max() < 1e-4


def test_convert_rejects_missing_and_extra_keys():
    params = _random_haiku_params(TINY)
    missing = dict(params)
    del missing["linear_1/w"]
    with pytest.raises(ValueError, match="없는 키"):
        convert(missing, TINY)
    with pytest.raises(ValueError, match="남는 키"):
        convert({**params, "linear_99/w": np.zeros(1, dtype=np.float32)}, TINY)


def test_convert_rejects_wrong_size():
    with pytest.raises(ValueError):
        convert(_random_haiku_params(TINY), TransformerConfig(embedding_dim=32, num_layers=3, num_heads=4))


def test_save_and_load_base(tmp_path):
    torch.manual_seed(0)
    model = ActionValueTransformer(TINY).eval()
    save_base(model, tmp_path / "base.pt")
    loaded = load_base(tmp_path / "base.pt").eval()
    tokens = torch.from_numpy(encode_position(chess.STARTING_FEN)[0])
    with torch.no_grad():
        assert torch.equal(model(tokens), loaded(tokens))
