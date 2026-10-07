"""UCI 수 ↔ 행동 번호(1,968개). searchless_chess `src/utils.py`의 이식이다.

행동 번호는 베이스 모델 임베딩의 행 번호이므로 생성 순서를 원본과 똑같이 유지해야 한다.
"""

import chess

_FILES = "abcdefgh"


def _all_moves() -> list[str]:
    moves = []

    # 일반 수: 각 칸에서 퀸과 나이트가 갈 수 있는 모든 칸 (캐슬링도 킹의 이동으로 포함된다)
    board = chess.BaseBoard.empty()
    for square in range(64):
        next_squares = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        next_squares += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        next_squares += board.attacks(square)
        board.remove_piece_at(square)
        for next_square in next_squares:
            moves.append(chess.square_name(square) + chess.square_name(next_square))

    # 프로모션: 직진, 왼쪽 잡기, 오른쪽 잡기 순
    for rank, next_rank in [("2", "1"), ("7", "8")]:
        for index, file in enumerate(_FILES):
            next_files = [file]
            if file > "a":
                next_files.append(_FILES[index - 1])
            if file < "h":
                next_files.append(_FILES[index + 1])
            for next_file in next_files:
                move = f"{file}{rank}{next_file}{next_rank}"
                moves += [move + piece for piece in "qrbn"]

    return moves


ACTION_TO_MOVE = _all_moves()
MOVE_TO_ACTION = {move: action for action, move in enumerate(ACTION_TO_MOVE)}
NUM_ACTIONS = len(ACTION_TO_MOVE)
assert NUM_ACTIONS == len(MOVE_TO_ACTION) == 1968


def ordered_legal_moves(board: chess.Board) -> list[chess.Move]:
    """합법수를 행동 번호 순으로 정렬한다. 원본 엔진과 같은 순서다."""
    return sorted(board.legal_moves, key=lambda move: MOVE_TO_ACTION[move.uci()])
