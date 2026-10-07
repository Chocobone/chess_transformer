"""searchless_chess action-value Transformer의 PyTorch 이식.

원본(`src/transformer.py`, JAX/Haiku)과 연산이 같아야 공개 체크포인트를 그대로 쓸 수 있다.
구조: 입력 오른쪽 shift → 임베딩(√d 스케일) + 학습형 위치 인코딩 → [pre-LN 어텐션, pre-LN SwiGLU MLP] × L
→ 최종 LN → 선형 → log-softmax. 인과 마스크는 없다.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.model.actions import MOVE_TO_ACTION, NUM_ACTIONS, ordered_legal_moves
from src.model.tokenizer import SEQUENCE_LENGTH, tokenize


@dataclass
class TransformerConfig:
    vocab_size: int = NUM_ACTIONS  # FEN 토큰(0~30)과 행동 토큰이 같은 임베딩 표를 쓴다
    output_size: int = 128  # 승률 구간 수
    embedding_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 8
    max_sequence_length: int = SEQUENCE_LENGTH + 2  # FEN 77 + 행동 1 + 리턴 1
    widening_factor: int = 4


MODEL_SIZES = {
    "9M": dict(num_layers=8, embedding_dim=256, num_heads=8),
    "136M": dict(num_layers=8, embedding_dim=1024, num_heads=8),
    "270M": dict(num_layers=16, embedding_dim=1024, num_heads=8),
}


def config_for(size: str) -> TransformerConfig:
    return TransformerConfig(**MODEL_SIZES[size])


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        shape = (batch, length, self.num_heads, dim // self.num_heads)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(batch, length, dim))


class GatedMLP(nn.Module):
    """SwiGLU: out(silu(split_1(x)) * split_2(x))."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        dim = config.embedding_dim
        hidden = dim * config.widening_factor
        self.split_1 = nn.Linear(dim, hidden, bias=False)
        self.split_2 = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.split_1(x)) * self.split_2(x))


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.embedding_dim)
        self.attn = Attention(config)
        self.ln_mlp = nn.LayerNorm(config.embedding_dim)
        self.mlp = GatedMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        return x + self.mlp(self.ln_mlp(x))


class ActionValueTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_layers))
        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.output_size)

        edges = torch.linspace(0.0, 1.0, config.output_size + 1)
        self.register_buffer("bucket_values", (edges[:-1] + edges[1:]) / 2, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens[N, 79] = FEN 77 + 행동 1 + 리턴 자리 1 → 마지막 위치의 구간 로그확률 [N, 128].

        원본처럼 입력을 오른쪽으로 한 칸 민다. 맨 앞에 0이 들어가고 마지막 리턴 토큰은 버려지므로,
        마지막 위치는 (FEN, 행동)을 보고 리턴 구간을 예측한다.
        """
        inputs = F.pad(tokens[:, :-1], (1, 0), value=0)
        x = self.token_embedding(inputs) * self.config.embedding_dim**0.5
        x = x + self.position_embedding.weight[: inputs.shape[1]]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x[:, -1]))
        return F.log_softmax(logits.float(), dim=-1)

    def win_prob(self, tokens: torch.Tensor) -> torch.Tensor:
        """각 (국면, 행동)의 예상 승률 q(s, a) [N]."""
        return self.forward(tokens).exp() @ self.bucket_values


def encode_position(fen: str) -> tuple[np.ndarray, list[chess.Move]]:
    """한 국면의 모든 합법수에 대한 입력 [합법수 개수, 79]와 그 순서의 합법수 목록."""
    moves = ordered_legal_moves(chess.Board(fen))
    tokens = np.zeros((len(moves), SEQUENCE_LENGTH + 2), dtype=np.int64)
    tokens[:, :SEQUENCE_LENGTH] = tokenize(fen)
    tokens[:, SEQUENCE_LENGTH] = [MOVE_TO_ACTION[move.uci()] for move in moves]
    return tokens, moves


def save_base(model: ActionValueTransformer, path: str | Path) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_base(path: str | Path, device: str | torch.device = "cpu") -> ActionValueTransformer:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ActionValueTransformer(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
