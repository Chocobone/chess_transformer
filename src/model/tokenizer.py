"""FEN → 77토큰. google-deepmind/searchless_chess `src/tokenizer.py`의 이식이다.

토큰 구성: 차례 1 + 판 64 + 캐슬링 4 + 앙파상 2 + 반수 3 + 전수 3 = 77.
"""

import numpy as np

# 순서가 곧 토큰 번호다. 원본과 한 글자도 다르면 안 된다.
_CHARACTERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h",
    "p", "n", "r", "k", "q",
    "P", "B", "N", "R", "Q", "K",
    "w", ".",
]
_INDEX = {char: index for index, char in enumerate(_CHARACTERS)}
_PAD = _INDEX["."]

SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> np.ndarray:
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")

    tokens = [_INDEX[side]]
    for char in board.replace("/", ""):
        if char in "12345678":
            tokens.extend(int(char) * [_PAD])
        else:
            tokens.append(_INDEX[char])

    if castling == "-":
        tokens.extend(4 * [_PAD])
    else:
        tokens.extend(_INDEX[char] for char in castling)
        tokens.extend((4 - len(castling)) * [_PAD])

    if en_passant == "-":
        tokens.extend(2 * [_PAD])
    else:
        tokens.extend(_INDEX[char] for char in en_passant)

    for counter in (halfmoves, fullmoves):
        tokens.extend(_INDEX[char] for char in counter)
        tokens.extend((3 - len(counter)) * [_PAD])

    assert len(tokens) == SEQUENCE_LENGTH, fen
    return np.asarray(tokens, dtype=np.int64)
