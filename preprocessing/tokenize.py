import numpy as np


def build_uci_vocabulary():
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f + r for f in files for r in ranks]

    vocab = {}
    idx = 0

    # ✅ knight moves + queen-like moves만 포함
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    knight_dirs = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

    def inside(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    for f in range(8):
        for r in range(8):
            for dx, dy in directions:
                x, y = f + dx, r + dy
                while inside(x, y):
                    move = f"{files[f]}{ranks[r]}{files[x]}{ranks[y]}"
                    vocab[move] = idx
                    idx += 1
                    x += dx
                    y += dy

            for dx, dy in knight_dirs:
                x, y = f + dx, r + dy
                if inside(x, y):
                    move = f"{files[f]}{ranks[r]}{files[x]}{ranks[y]}"
                    vocab[move] = idx
                    idx += 1

    return vocab, {v: k for k, v in vocab.items()}


def get_legal_move_mask(board, vocab):
    mask = np.zeros(len(vocab), dtype=np.float32)
    for m in board.legal_moves:
        u = m.uci()
        if u in vocab:
            mask[vocab[u]] = 1
    return mask


# ✅ positional bias용 (distance encoding)
def get_relative_position_matrix():
    mat = np.zeros((64, 64), dtype=np.int32)

    for i in range(64):
        x1, y1 = divmod(i, 8)
        for j in range(64):
            x2, y2 = divmod(j, 8)
            mat[i, j] = abs(x1 - x2) + abs(y1 - y2)

    return mat
