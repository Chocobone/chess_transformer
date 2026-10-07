import math

import chess
import pytest
import torch

from src.model.actions import ACTION_TO_MOVE, MOVE_TO_ACTION
from src.model.lora import inject_lora
from src.model.transformer import ActionValueTransformer, TransformerConfig
from src.train.losses import imitation_loss, strength_loss, style_loss, total_loss
from src.train.policy import legal_action_tokens, pad_by_owner, policy_from_q, target_index

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)

FENS = [
    chess.STARTING_FEN,  # 합법수 20개
    "8/8/8/8/8/8/8/K6k w - - 0 1",  # 3개
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # 27개
]
COUNTS = [20, 3, 27]


def test_legal_action_tokens():
    tokens, owner, action_ids = legal_action_tokens(FENS)
    assert tokens.shape == (sum(COUNTS), 79)
    assert torch.bincount(owner).tolist() == COUNTS
    for index, fen in enumerate(FENS):
        ids = action_ids[owner == index].tolist()
        assert ids == sorted(ids)
        assert {ACTION_TO_MOVE[i] for i in ids} == {move.uci() for move in chess.Board(fen).legal_moves}
    assert torch.equal(tokens[:, -2], action_ids)

    with pytest.raises(ValueError):
        legal_action_tokens(["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"])  # 스테일메이트


def test_pad_by_owner():
    owner = torch.tensor([0, 0, 1, 2, 2, 2])
    padded, mask = pad_by_owner(torch.arange(6.0), owner, fill=-1.0)
    assert padded.tolist() == [[0, 1, -1], [2, -1, -1], [3, 4, 5]]
    assert mask.tolist() == [[True, True, False], [True, False, False], [True, True, True]]


def test_policy_sums_to_one_and_padding_is_zero():
    _, owner, _ = legal_action_tokens(FENS)
    torch.manual_seed(0)
    log_pi, mask = policy_from_q(torch.rand(len(owner)), owner, temperature=0.02)
    assert log_pi.shape == mask.shape == (3, 27)
    assert mask.sum(dim=1).tolist() == COUNTS
    pi = log_pi.exp()
    assert torch.allclose(pi.sum(dim=1), torch.ones(3), atol=1e-6)
    assert (pi[~mask] == 0).all()


def test_policy_matches_hand_calculation():
    q = torch.tensor([0.5, 0.6, 0.3])
    owner = torch.tensor([0, 0, 1])
    log_pi, _ = policy_from_q(q, owner, temperature=0.1)
    expected = 1 / (1 + math.exp(-1.0))  # 수 2개, q 차이 0.1, T 0.1
    assert log_pi[0, 1].exp().item() == pytest.approx(expected, abs=1e-6)
    assert log_pi[1, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_target_index():
    _, owner, action_ids = legal_action_tokens(FENS)
    targets = torch.tensor([MOVE_TO_ACTION[move] for move in ("e2e4", "a1b2", "f1b5")])
    index = target_index(action_ids, owner, targets)
    padded, _ = pad_by_owner(action_ids, owner, fill=-1)
    assert padded.gather(1, index[:, None]).squeeze(1).tolist() == targets.tolist()

    with pytest.raises(ValueError):
        target_index(action_ids, owner, torch.tensor([MOVE_TO_ACTION["e2e5"]] * 3))


def policy(rows: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """확률 표(패딩은 0)를 log_pi와 mask로 바꾼다."""
    pi = torch.tensor(rows)
    return pi.log(), pi > 0


def test_imitation_loss():
    log_pi, _ = policy([[0.5, 0.25, 0.25], [0.1, 0.9, 0.0]])
    target = torch.tensor([0, 1])
    expected = -(math.log(0.5) + math.log(0.9)) / 2
    assert imitation_loss(log_pi, target).item() == pytest.approx(expected, abs=1e-6)

    # gamma: 베이스가 이미 0.75로 두는 수는 가중치 0.25, 0.5로 두는 수는 0.5
    base_log_pi, _ = policy([[0.75, 0.125, 0.125], [0.5, 0.5, 0.0]])
    weighted = -(0.25 * math.log(0.5) + 0.5 * math.log(0.9)) / 2
    assert imitation_loss(log_pi, target, base_log_pi, gamma=1.0).item() == pytest.approx(weighted, abs=1e-6)
    squared = -(0.25**2 * math.log(0.5) + 0.5**2 * math.log(0.9)) / 2
    assert imitation_loss(log_pi, target, base_log_pi, gamma=2.0).item() == pytest.approx(squared, abs=1e-6)

    with pytest.raises(ValueError):
        imitation_loss(log_pi, target, gamma=1.0)


def test_strength_loss():
    log_pi, mask = policy([[0.5, 0.25, 0.25], [0.1, 0.9, 0.0]])
    # 패딩 자리의 q_base는 무시되어야 한다 (0.99가 최댓값으로 잡히면 안 된다)
    q_base = torch.tensor([[0.60, 0.59, 0.58], [0.50, 0.20, 0.99]])

    # 모든 수가 밴드 안: 첫 국면 regret 최대 0.02, 둘째 0.30
    assert strength_loss(log_pi, q_base, mask, tau=0.5).item() == 0.0
    # 밴드 밖 수에 확률이 실리면 양수: 둘째 국면만 0.9 × (0.30 − 0.05)
    expected = (0.0 + 0.9 * 0.25) / 2
    assert strength_loss(log_pi, q_base, mask, tau=0.05).item() == pytest.approx(expected, abs=1e-6)
    # tau 0: Σ π · regret
    expected = (0.25 * 0.01 + 0.25 * 0.02 + 0.9 * 0.30) / 2
    assert strength_loss(log_pi, q_base, mask, tau=0.0).item() == pytest.approx(expected, abs=1e-6)


def test_style_loss():
    log_pi, mask = policy([[0.5, 0.25, 0.25], [0.1, 0.9, 0.0]])
    assert style_loss(log_pi, None, None, mask).item() == 0.0

    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, -1.0], [9.0, 9.0]]]
    )
    w_p = torch.tensor([1.0, 2.0])
    # 점수: [1, 2, 3], [2, -2, 패딩]
    expected = -((0.5 * 1 + 0.25 * 2 + 0.25 * 3) + (0.1 * 2 + 0.9 * -2)) / 2
    assert style_loss(log_pi, features, w_p, mask).item() == pytest.approx(expected, abs=1e-6)


def test_total_loss_combines_terms():
    log_pi, mask = policy([[0.5, 0.25, 0.25], [0.1, 0.9, 0.0]])
    target = torch.tensor([0, 1])
    q_base = torch.tensor([[0.60, 0.59, 0.58], [0.50, 0.20, 0.0]])
    features = torch.ones(2, 3, 2)
    w_p = torch.tensor([1.0, 2.0])

    loss, parts = total_loss(log_pi, mask, target, q_base, tau=0.05, lam=2.0, alpha=0.5, features=features, w_p=w_p)
    expected = parts["imitation"] + 0.5 * parts["style"] + 2.0 * parts["strength"]
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert parts["style"].item() == pytest.approx(-3.0, abs=1e-6)

    # 기본값(alpha 0, 특성 없음)에서는 모방 항 + 기력 항
    loss, parts = total_loss(log_pi, mask, target, q_base, tau=0.05)
    assert parts["style"].item() == 0.0
    assert loss.item() == pytest.approx((parts["imitation"] + parts["strength"]).item(), abs=1e-6)


def test_gradients_are_finite_with_padding():
    torch.manual_seed(0)
    _, owner, _ = legal_action_tokens(FENS)
    q = torch.rand(len(owner), requires_grad=True)
    log_pi, mask = policy_from_q(q, owner, temperature=0.02)
    q_base, _ = pad_by_owner(torch.rand(len(owner)), owner)
    loss, _ = total_loss(log_pi, mask, torch.tensor([0, 1, 2]), q_base, tau=0.02)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0


def test_lora_overfits_one_batch():
    """같은 배치를 반복 학습하면 모방 손실이 줄고 목표 수의 확률이 오른다."""
    torch.manual_seed(0)
    model = ActionValueTransformer(TINY)
    tokens, owner, action_ids = legal_action_tokens(FENS)
    targets = torch.tensor([MOVE_TO_ACTION[move] for move in ("g1f3", "a1b2", "f1b5")])
    target = target_index(action_ids, owner, targets)

    temperature = 0.02
    with torch.no_grad():
        q_base, _ = pad_by_owner(model.win_prob(tokens), owner)
        base_log_pi, _ = policy_from_q(model.win_prob(tokens), owner, temperature)

    parameters = inject_lora(model, rank=4, alpha=8)
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    history = []
    for _ in range(60):
        log_pi, mask = policy_from_q(model.win_prob(tokens), owner, temperature)
        # tau를 크게 잡아 기력 항을 끄고 모방 항만 본다
        loss, parts = total_loss(log_pi, mask, target, q_base, tau=1.0)
        history.append(parts["imitation"].item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    initial = imitation_loss(base_log_pi, target).item()
    assert history[0] == pytest.approx(initial, abs=1e-5)  # 주입 직후는 베이스와 같다
    assert history[-1] < 0.5 * initial
