"""개발 환경 점검: 필수 패키지와 입력 기보가 준비되어 있는지 확인한다."""

from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

PGNS = [
    "model_tal/Tal.pgn",
    "model_karpov/Karpov.pgn",
    "model_smyslov/Smyslov.pgn",
]


def test_torch_forward_backward():
    layer = torch.nn.Linear(4, 2)
    loss = layer(torch.randn(3, 4)).sum()
    loss.backward()
    assert layer.weight.grad is not None


def test_chess_legal_moves():
    board = chess.Board()
    assert board.legal_moves.count() == 20
    board.push_uci("e2e4")
    assert board.turn == chess.BLACK


@pytest.mark.parametrize("rel_path", PGNS)
def test_pgn_readable(rel_path):
    path = ROOT / rel_path
    assert path.is_file()
    with open(path, encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)
    assert game is not None
    assert len(list(game.mainline_moves())) > 0


def test_src_packages_importable():
    import src.data
    import src.model
    import src.train
