"""동결 베이스에 붙이는 LoRA 어댑터.

베이스 가중치는 건드리지 않고 선형층마다 저랭크 보정 (alpha / r) * B A 만 학습한다.
B를 0으로 초기화하므로 주입 직후의 출력은 베이스와 같다.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.model.transformer import ActionValueTransformer

ATTENTION_TARGETS = ("q", "k", "v", "o")
MLP_LAYERS = ("split_1", "split_2", "out")


class LoRALinear(nn.Module):
    """W x + (alpha / r) * B A x."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scale * update


def inject_lora(
    model: ActionValueTransformer,
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ATTENTION_TARGETS,
) -> list[nn.Parameter]:
    """모델 전체를 동결하고 대상 선형층을 LoRALinear로 바꾼다. 학습할 파라미터 목록을 돌려준다.

    targets: 어텐션의 "q", "k", "v", "o"와 "mlp"(MLP 선형층 3개) 중에서 고른다.
    """
    unknown = set(targets) - set(ATTENTION_TARGETS) - {"mlp"}
    if unknown:
        raise ValueError(f"알 수 없는 LoRA 대상: {sorted(unknown)}")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("이미 LoRA가 주입된 모델이다")

    model.requires_grad_(False)

    # 새 A/B는 베이스와 같은 장치·자료형에 둔다
    reference = model.head.weight
    for block in model.blocks:
        layers = [(block.attn, name) for name in ATTENTION_TARGETS if name in targets]
        if "mlp" in targets:
            layers += [(block.mlp, name) for name in MLP_LAYERS]
        for parent, name in layers:
            wrapped = LoRALinear(getattr(parent, name), rank, alpha, dropout)
            setattr(parent, name, wrapped.to(reference.device, reference.dtype))

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _is_lora_key(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in ("lora_A", "lora_B")


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """어댑터 파라미터만 담은 state_dict. 베이스 가중치는 들어가지 않는다."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if _is_lora_key(key)}


def load_lora(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """같은 설정으로 inject_lora를 마친 모델에 어댑터를 싣는다. 키가 하나라도 어긋나면 실패한다."""
    expected = {key for key in model.state_dict() if _is_lora_key(key)}
    missing = sorted(expected - set(state_dict))
    unexpected = sorted(set(state_dict) - expected)
    if missing or unexpected:
        raise ValueError(f"어댑터 키 불일치: 누락 {missing}, 잉여 {unexpected}")
    model.load_state_dict(state_dict, strict=False)


def save_adapter(model: nn.Module, path: str | Path, settings: dict) -> None:
    """어댑터 가중치와 주입 설정(rank, alpha, dropout, targets)을 한 파일에 저장한다."""
    torch.save({"settings": settings, "state_dict": lora_state_dict(model)}, path)


def load_adapter(model: ActionValueTransformer, path: str | Path) -> dict:
    """베이스 모델에 저장된 설정대로 LoRA를 주입하고 어댑터를 싣는다. 주입 설정을 돌려준다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    inject_lora(model, settings["rank"], settings["alpha"], settings["dropout"], tuple(settings["targets"]))
    load_lora(model, checkpoint["state_dict"])
    return settings
