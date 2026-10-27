"""대국 상태. 규칙(합법수, 종료 판정, SAN, PGN)은 모두 python-chess에 맡긴다."""

import datetime
from dataclasses import asdict, dataclass

import chess
import chess.pgn

from src.play.engine import Analysis, Candidate

COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}


class TurnError(Exception):
    """차례가 아니거나 대국이 끝나서 할 수 없는 요청."""


@dataclass
class MoveRecord:
    uci: str
    san: str
    by: str  # "user" | "model"
    fen: str  # 이 수를 둔 뒤의 국면 (되돌려 보기용)
    prob: float | None = None  # 모델이 둔 수의 π
    analysis: Analysis | None = None  # 모델이 이 수를 고를 때의 후보
    repetition: bool = False  # 이 수로 같은 국면이 세 번째가 되었는지


class Game:
    def __init__(self, model_id: str, model_label: str, user_color: chess.Color):
        self.model_id = model_id
        self.model_label = model_label
        self.user_color = user_color
        self.board = chess.Board()
        self.records: list[MoveRecord] = []
        self.resigned: chess.Color | None = None
        self.started = datetime.date.today()

    @property
    def user_to_move(self) -> bool:
        return self.board.turn == self.user_color

    def require_turn(self, user: bool) -> None:
        """대국이 진행 중이고 user(True: 사용자, False: 모델)의 차례인지 확인한다. 아니면 TurnError."""
        if self.outcome() is not None:
            raise TurnError("대국이 끝났다")
        if self.user_to_move != user:
            raise TurnError("모델 차례다" if user else "사용자 차례다")

    def push_user(self, uci: str) -> None:
        self.require_turn(user=True)
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError(f"잘못된 수 표기다: {uci}") from error
        if move not in self.board.legal_moves:
            raise ValueError(f"둘 수 없는 수다: {uci}")
        self._push(move, "user")

    def push_model(self, candidate: Candidate, analysis: Analysis) -> None:
        self.require_turn(user=False)
        move = chess.Move.from_uci(candidate.uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"모델이 둘 수 없는 수를 냈다: {candidate.uci}")
        self._push(move, "model", candidate.prob, analysis)

    def undo(self) -> None:
        """사용자가 마지막으로 둔 수까지 되돌린다 (보통 모델 수와 함께 2수). 사용자 수가 없으면 그대로 둔다."""
        if not any(record.by == "user" for record in self.records):
            return
        while True:
            self.board.pop()
            if self.records.pop().by == "user":
                break
        self.resigned = None

    def resign(self) -> None:
        if self.outcome() is not None:
            raise TurnError("대국이 끝났다")
        self.resigned = self.user_color

    def outcome(self) -> dict | None:
        """종료면 {"result", "reason", "winner"}, 아니면 None.

        3회 반복과 50수 규칙은 실제로 그 상태가 되면 자동으로 무승부다. `outcome(claim_draw=True)`는
        다음 한 수로 반복이 될 수 있기만 해도 무승부로 보므로 쓰지 않는다.
        """
        if self.resigned is not None:
            winner = not self.resigned
            return {"result": "1-0" if winner else "0-1", "reason": "resign", "winner": COLOR_NAMES[winner]}
        outcome = self.board.outcome()
        if outcome is not None:
            winner = None if outcome.winner is None else COLOR_NAMES[outcome.winner]
            return {"result": outcome.result(), "reason": outcome.termination.name.lower(), "winner": winner}
        if self.board.is_repetition(3):
            return {"result": "1/2-1/2", "reason": "threefold_repetition", "winner": None}
        if self.board.is_fifty_moves():
            return {"result": "1/2-1/2", "reason": "fifty_moves", "winner": None}
        return None

    def state(self) -> dict:
        outcome = self.outcome()
        last_model = next((r for r in reversed(self.records) if r.by == "model"), None)
        king = self.board.king(self.board.turn)
        return {
            "fen": self.board.fen(),
            "initial_fen": chess.STARTING_FEN,
            "turn": COLOR_NAMES[self.board.turn],
            "user_color": COLOR_NAMES[self.user_color],
            "model": {"id": self.model_id, "label": self.model_label},
            "legal_moves": [m.uci() for m in self.board.legal_moves] if self.user_to_move and outcome is None else [],
            "last_move": self.records[-1].uci if self.records else None,
            "check": chess.square_name(king) if self.board.is_check() and king is not None else None,
            "moves": [
                {"uci": r.uci, "san": r.san, "by": r.by, "fen": r.fen, "prob": r.prob, "repetition": r.repetition}
                for r in self.records
            ],
            "last_analysis": asdict(last_model.analysis) if last_model and last_model.analysis else None,
            "outcome": outcome,
        }

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        user, model = "사용자", self.model_label
        game.headers["Event"] = "모델과 대국"
        game.headers["Site"] = "chess_transformer 대국 앱"
        game.headers["Date"] = self.started.strftime("%Y.%m.%d")
        game.headers["White"] = user if self.user_color == chess.WHITE else model
        game.headers["Black"] = model if self.user_color == chess.WHITE else user
        outcome = self.outcome()
        game.headers["Result"] = outcome["result"] if outcome else "*"
        if outcome:
            game.headers["Termination"] = outcome["reason"]
        return str(game) + "\n"

    def _push(self, move: chess.Move, by: str, prob: float | None = None, analysis: Analysis | None = None) -> None:
        san = self.board.san(move)
        self.board.push(move)
        self.records.append(MoveRecord(move.uci(), san, by, self.board.fen(), prob, analysis, self.board.is_repetition(3)))
