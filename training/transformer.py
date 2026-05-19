import copy
import os
import random

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from chess_tokenize import build_uci_vocabulary, get_legal_move_mask


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
# PER (Prioritized Experience Replay)
# ---------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        while idx > 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def _retrieve(self, idx, s):
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """capacity를 줄이고 (20k) 우선순위 기반 샘플링으로 오래된 약한 데이터 비중을 낮춤."""

    def __init__(self, capacity=20_000, alpha=0.6, beta_start=0.4, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.epsilon = epsilon
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def add(self, sample):
        self.tree.add(self.max_priority, sample)

    def extend(self, samples):
        for s in samples:
            self.add(s)

    def sample(self, n):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n

        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                continue
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        if not batch:
            return [], [], np.ones(0, dtype=np.float32)

        probs = np.array(priorities) / (self.tree.total() + 1e-8)
        weights = (len(self) * probs + 1e-8) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        return batch, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (abs(float(error)) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def anneal_beta(self, epoch, total_epochs):
        self.beta = min(1.0, 0.4 + 0.6 * epoch / total_epochs)


# ---------------------------
# Opponent Pool
# ---------------------------
class OpponentPool:
    """과거 체크포인트 pool — epoch마다 일정 확률로 현재 모델과 대전."""

    def __init__(self, max_size=5):
        self.pool = []
        self.max_size = max_size

    def add(self, state_dict):
        self.pool.append(copy.deepcopy(state_dict))
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample_opponent(self, vocab_size, device):
        if not self.pool:
            return None
        state = random.choice(self.pool)
        m = ChessTransformer(vocab_size).to(device)
        m.load_state_dict(state)
        m.eval()
        return m


# ---------------------------
# MCTS
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

    states = torch.tensor(np.array([encode_board(b) for b in boards])).to(device)
    turns = torch.tensor(np.array([int(b.turn) for b in boards])).to(device)
    masks = torch.tensor(np.array([get_legal_move_mask(b, vocab) for b in boards])).to(device)

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
    pi = np.zeros(len(vocab), dtype=np.float32)
    for m, c in root.children.items():
        pi[vocab[m]] = c.N
    if pi.sum() > 0:
        pi /= pi.sum()
    return pi


# ---------------------------
# CSV 데이터 로드
# ---------------------------
def load_csv_games(csv_path, vocab):
    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        result = row["Result"]
        z = 1 if result == "1-0" else -1 if result == "0-1" else 0

        board = chess.Board()
        try:
            moves = str(row["UCI_Moves"]).split()
        except Exception:
            continue

        for move_uci in moves:
            if board.is_game_over():
                break

            try:
                move = chess.Move.from_uci(move_uci)
            except Exception:
                break

            if move not in board.legal_moves:
                break

            if move_uci in vocab:
                pi = np.zeros(len(vocab), dtype=np.float32)
                pi[vocab[move_uci]] = 1.0

                mask = get_legal_move_mask(board, vocab)
                value = float(z if board.turn == chess.WHITE else -z)

                samples.append((encode_board(board), int(board.turn), pi, value, mask))

            board.push(move)

    return samples


# ---------------------------
# self-play (temperature annealing + opponent pool)
# ---------------------------
def self_play(current_model, vocab, device, temperature=1.0, opponent_model=None):
    """
    temperature: 높을수록 탐색적, 낮을수록 greedy.
    opponent_model: 있으면 랜덤 컬러를 맡아 대전. current_model 포지션만 buffer에 저장.
    """
    board = chess.Board()
    data = []

    if opponent_model is not None:
        opponent_color = random.choice([chess.WHITE, chess.BLACK])
    else:
        opponent_color = None

    while not board.is_game_over():
        is_opponent = (opponent_color is not None) and (board.turn == opponent_color)
        active_model = opponent_model if is_opponent else current_model

        root = run_mcts(board, active_model, vocab, device)
        pi = get_policy(root, vocab)
        mask = get_legal_move_mask(board, vocab)

        moves = list(root.children.keys())
        if not moves:
            break

        counts = np.array([root.children[m].N for m in moves], dtype=np.float32)

        if temperature > 0:
            probs = counts ** (1.0 / temperature)
        else:
            probs = (counts == counts.max()).astype(np.float32)
        probs = probs / probs.sum()

        move = np.random.choice(moves, p=probs)

        if not is_opponent:
            data.append((encode_board(board), int(board.turn), pi, mask))

        board.push_uci(move)

    result = board.result()
    z = 1 if result == "1-0" else -1 if result == "0-1" else 0

    final = []
    for s, t, pi, mask in data:
        value = float(z if t else -z)
        final.append((s, t, pi, value, mask))

    return final


# ---------------------------
# training
# ---------------------------
def train(data_dir="data"):
    TOTAL_EPOCHS = 50
    WARMUP_EPOCHS = 5
    GAMES_PER_EPOCH = 5
    BATCH_SIZE = 128
    CHECKPOINT_INTERVAL = 10  # opponent pool 저장 주기 (epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    vocab, _ = build_uci_vocabulary()
    model = ChessTransformer(len(vocab)).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    # warm-up 5 epoch → cosine annealing (min lr = 1e-6)
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, TOTAL_EPOCHS - WARMUP_EPOCHS)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    buffer = PrioritizedReplayBuffer(capacity=20_000, alpha=0.6, beta_start=0.4)
    opponent_pool = OpponentPool(max_size=5)

    train_csv = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_csv):
        print("CSV 데이터 로드 중...")
        csv_samples = load_csv_games(train_csv, vocab)
        buffer.extend(csv_samples)
        print(f"  → {len(csv_samples)}개 샘플 로드 완료 (buffer: {len(buffer)})")
    else:
        print(f"CSV 없음 ({train_csv}), self-play만 사용")

    for epoch in range(TOTAL_EPOCHS):
        # temperature annealing: 초반 탐색(1.5) → 후반 수렴(0.1)
        temperature = max(0.1, 1.5 * (1 - epoch / TOTAL_EPOCHS))

        # 50% 확률로 opponent pool과 대전
        opponent = None
        if len(opponent_pool.pool) > 0 and random.random() < 0.5:
            opponent = opponent_pool.sample_opponent(len(vocab), device)

        for _ in range(GAMES_PER_EPOCH):
            buffer.extend(self_play(model, vocab, device, temperature=temperature, opponent_model=opponent))

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            opponent_pool.add(model.state_dict())
            ckpt_path = os.path.join(data_dir, f"checkpoint_epoch{epoch + 1}.pt")
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
            print(f"  → checkpoint 저장: {ckpt_path} (pool size: {len(opponent_pool.pool)})")

    final_path = os.path.join(data_dir, "model_final.pt")
    torch.save({"epoch": TOTAL_EPOCHS, "model": model.state_dict(), "opt": opt.state_dict()}, final_path)
    print(f"학습 완료. 최종 모델 저장: {final_path}")

        if len(buffer) < BATCH_SIZE:
            print(f"epoch {epoch:3d} | 샘플 부족 ({len(buffer)}개), skip")
            scheduler.step()
            continue

        buffer.anneal_beta(epoch, TOTAL_EPOCHS)
        batch, idxs, weights = buffer.sample(BATCH_SIZE)

        if not batch:
            scheduler.step()
            continue

        weights_t = torch.tensor(weights).to(device)

        s = torch.tensor(np.array([b[0] for b in batch])).to(device)
        t = torch.tensor(np.array([b[1] for b in batch])).to(device)
        pi = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)
        v = (
            torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        masks = torch.tensor(np.array([b[4] for b in batch])).to(device)

        p_pred, v_pred = model(s, t, masks)

        loss_p = -(pi * torch.log_softmax(p_pred, dim=-1)).sum(dim=1)
        loss_v = (v_pred - v).squeeze(1) ** 2
        loss = ((loss_p + loss_v) * weights_t).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # TD error로 샘플 우선순위 업데이트
        with torch.no_grad():
            td_errors = (v_pred - v).squeeze(1).abs().cpu().numpy()
        buffer.update_priorities(idxs, td_errors)

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | loss {loss.item():.4f} | "
            f"buffer {len(buffer)} | lr {lr_now:.2e} | temp {temperature:.2f}"
        )


if __name__ == "__main__":
    train()
