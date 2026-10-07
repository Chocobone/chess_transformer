"""학습 목표: L = L_imit + alpha * L_style + lam * L_strength (제안서 4.3).

모든 함수는 `policy_from_q`가 만든 log_pi[B, A_max]와 mask[B, A_max]를 받고, 국면 평균을 돌려준다.
"""

import torch


def imitation_loss(
    log_pi: torch.Tensor,
    target_idx: torch.Tensor,
    base_log_pi: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    """기사가 둔 수 a*에 대한 교차엔트로피.

    gamma > 0이면 표본마다 (1 - π_base(a*|s))^gamma를 곱한다. 베이스도 당연히 두는 수는 약하게,
    베이스가 예상하지 못한 수는 강하게 학습한다 (제안서 "베이스 대비 잔차"에 대한 구현자의 해석).
    """
    nll = -log_pi.gather(1, target_idx[:, None]).squeeze(1)
    if gamma > 0:
        if base_log_pi is None:
            raise ValueError("gamma > 0이면 base_log_pi가 필요하다")
        base_prob = base_log_pi.gather(1, target_idx[:, None]).squeeze(1).exp()
        nll = (1 - base_prob).clamp_min(0).pow(gamma).detach() * nll
    return nll.mean()


def strength_loss(log_pi: torch.Tensor, q_base: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    """Σ π(a|s) · max(0, regret(s,a) − τ), regret(s,a) = max_a' q_base(s,a') − q_base(s,a).

    베이스 기준으로 최선수보다 τ 넘게 나쁜 수에 실린 확률만 벌한다.
    """
    best = q_base.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True).values
    penalty = (best - q_base - tau).clamp_min(0).masked_fill(~mask, 0.0)
    return (log_pi.exp() * penalty).sum(dim=1).mean()


def style_loss(
    log_pi: torch.Tensor,
    features: torch.Tensor | None,
    w_p: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    """−Σ π(a|s) · (w_p · f̃(s,a)). features[B, A_max, F]가 없으면 0이다."""
    if features is None or w_p is None:
        return log_pi.new_zeros(())
    score = (features @ w_p).masked_fill(~mask, 0.0)
    return -(log_pi.exp() * score).sum(dim=1).mean()


def total_loss(
    log_pi: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
    q_base: torch.Tensor,
    tau: float,
    lam: float = 1.0,
    alpha: float = 0.0,
    gamma: float = 0.0,
    base_log_pi: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    w_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """전체 손실과 항별 값(로그용, 그래프에서 분리됨)을 돌려준다."""
    imitation = imitation_loss(log_pi, target_idx, base_log_pi, gamma)
    strength = strength_loss(log_pi, q_base, mask, tau)
    style = style_loss(log_pi, features, w_p, mask)
    loss = imitation + alpha * style + lam * strength
    parts = {"imitation": imitation.detach(), "strength": strength.detach(), "style": style.detach()}
    return loss, parts
