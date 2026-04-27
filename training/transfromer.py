import random
from collections import deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.tokenize import build_uci_vocabulary, get_legal_move_mask


# ---------------------------
# board encoding
# ---------------------------
def encode_board(board):
    squares = np.zeros(64, dtype=np.int64)
    for i in range(64):
        p = board.piece_at(i)
        if p:
            squares[i] = p.piece_type if p.color else p.piece_type + 6
    return squares


# ---------------------------
# model
# ---------------------------
class ChessTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        d = 128

        self.piece_emb = nn.Embedding(13, d)
        self.pos_emb = nn.Embedding(64, d)

        self.turn_emb = nn.Embedding(2, d)

        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)

        self.policy_head = nn.Linear(d * 64, vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(d * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x, turn, mask=None):
        b = x.size(0)

        pos = torch.arange(64, device=x.device).unsqueeze(0).expand(b, 64)

        x = self.piece_emb(x) + self.pos_emb(pos)
        x = x + self.turn_emb(turn).unsqueeze(1)

        x = self.encoder(x)
        x = x.reshape(b, -1)

        p = self.policy_head(x)
        v = self.value_head(x)

        if mask is not None:
            p = p.masked_fill(mask == 0, -1e9)

        return p, v


# ---------------------------
# MCTS (경량화)
# ---------------------------
class Node:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.P = prior

    def Q(self):
        return 0 if self.N == 0 else self.W / self.N


def ucb(parent, child, c=1.2):
    return child.Q() + c * child.P * (np.sqrt(parent.N) / (1 + child.N))


def expand_batch(nodes, model, vocab, device):
    boards = [n.board for n in nodes]

    states = torch.tensor([encode_board(b) for b in boards]).to(device)
    turns = torch.tensor([int(b.turn) for b in boards]).to(device)
    masks = torch.tensor([get_legal_move_mask(b, vocab) for b in boards]).to(device)

    with torch.no_grad():
        p_logits, values = model(states, turns, masks)

    policies = torch.softmax(p_logits, dim=-1).cpu().numpy()
    values = values.cpu().numpy()

    for i, node in enumerate(nodes):
        b = node.board
        p = policies[i]

        for m in b.legal_moves:
            u = m.uci()
            if u not in vocab:
                continue

            nb = b.copy()
            nb.push(m)

            node.children[u] = Node(nb, node, p[vocab[u]])

    return values


def run_mcts(board, model, vocab, device, sims=32, batch_size=8):
    root = Node(board)
    expand_batch([root], model, vocab, device)

    for _ in range(sims // batch_size):
        leaves = []

        for _ in range(batch_size):
            node = root
            while node.children:
                node = max(node.children.values(), key=lambda c: ucb(node, c))
            leaves.append(node)

        values = expand_batch(leaves, model, vocab, device)

        for node, v in zip(leaves, values):
            while node:
                node.N += 1
                node.W += v
                v = -v
                node = node.parent

    return root


def get_policy(root, vocab):
    pi = np.zeros(len(vocab))
    for m, c in root.children.items():
        pi[vocab[m]] = c.N
    if pi.sum() > 0:
        pi /= pi.sum()
    return pi


# ---------------------------
# self-play
# ---------------------------
def self_play(model, vocab, device):
    board = chess.Board()
    data = []

    while not board.is_game_over():
        root = run_mcts(board, model, vocab, device)

        pi = get_policy(root, vocab)

        moves = list(root.children.keys())
        probs = np.array([root.children[m].N for m in moves])
        probs = probs / probs.sum()

        move = np.random.choice(moves, p=probs)

        data.append((encode_board(board), int(board.turn), pi))

        board.push_uci(move)

    result = board.result()
    z = 1 if result == "1-0" else -1 if result == "0-1" else 0

    final = []
    for s, t, pi in data:
        value = z if t else -z
        final.append((s, t, pi, value))

    return final


# ---------------------------
# training
# ---------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, _ = build_uci_vocabulary()
    model = ChessTransformer(len(vocab)).to(device)

    opt = optim.AdamW(model.parameters(), lr=1e-4)

    buffer = deque(maxlen=5000)

    for epoch in range(50):
        for _ in range(5):
            buffer.extend(self_play(model, vocab, device))

        batch = random.sample(buffer, min(len(buffer), 128))

        s = torch.tensor([b[0] for b in batch]).to(device)
        t = torch.tensor([b[1] for b in batch]).to(device)
        pi = torch.tensor([b[2] for b in batch]).to(device)
        v = torch.tensor([b[3] for b in batch]).unsqueeze(1).to(device)

        masks = torch.tensor([get_legal_move_mask(chess.Board(), vocab) for _ in batch]).to(device)

        p_pred, v_pred = model(s, t, masks)

        loss_p = -(pi * torch.log_softmax(p_pred, dim=-1)).sum(dim=1).mean()
        loss_v = ((v_pred - v) ** 2).mean()

        loss = loss_p + loss_v

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"epoch {epoch} loss {loss.item():.4f}")


if __name__ == "__main__":
    train()
