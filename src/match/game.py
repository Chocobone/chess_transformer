"""대국 하나의 상태와 종료 판정 (평가 기준 문서 5.3절, Task #13 2단계).

종료 규칙은 대국 앱 `src/play/game.py`의 `Game.outcome`과 같다: python-chess의 `board.outcome()`(체크메이트,
스테일메이트, 기물 부족, 75수, 5회 반복) 다음에 실제 3회 반복, 50수 규칙. `outcome(claim_draw=True)`는 다음 한 수로
반복이 될 수 있기만 해도 무승부로 보므로 쓰지 않는다. 여기에 400 ply 한도를 더한다 (해석 1: 오프닝 8 ply 포함).
평가값 판정(adjudication)은 하지 않는다.
"""

from dataclasses import dataclass, field

import chess
import chess.pgn
import torch

MAX_PLY = 400
OPENING_PLY = 8


@dataclass(frozen=True)
class GameSpec:
    opening: int  # 오프닝 번호
    fen: str  # 오프닝 국면 (8 ply 뒤, 백 차례)
    model: str  # 평가 대상 모델 ID
    opponent: str  # 상대 ID ("base" 또는 "sf")
    model_color: chess.Color  # 평가 대상의 색

    @property
    def seed(self) -> int:
        """해석 2: seed = 2 × 오프닝 번호 + 색(백 0, 흑 1). 모든 조합이 같은 seed를 쓴다."""
        return 2 * self.opening + (0 if self.model_color == chess.WHITE else 1)

    @property
    def white(self) -> str:
        return self.model if self.model_color == chess.WHITE else self.opponent

    @property
    def black(self) -> str:
        return self.opponent if self.model_color == chess.WHITE else self.model

    @property
    def key(self) -> tuple[int, str]:
        return self.opening, "white" if self.model_color == chess.WHITE else "black"


def pair_specs(openings: list[dict], indices: list[int], model: str, opponent: str) -> list[GameSpec]:
    """짝: 오프닝마다 평가 대상이 백인 대국, 흑인 대국 순서로 2국."""
    by_index = {row["index"]: row for row in openings}
    specs = []
    for i in indices:
        for color in (chess.WHITE, chess.BLACK):
            specs.append(GameSpec(i, by_index[i]["fen"], model, opponent, color))
    return specs


@dataclass
class MatchGame:
    spec: GameSpec
    board: chess.Board = field(init=False)
    generator: torch.Generator = field(init=False)
    probs: list[float | None] = field(default_factory=list)  # 수마다 모델의 π (Stockfish 수는 None)

    def __post_init__(self):
        self.board = chess.Board(self.spec.fen)
        # 해석 2: 한 대국에 생성기 하나. C1에서는 두 모델이 차례로 같은 생성기를 쓴다
        self.generator = torch.Generator().manual_seed(self.spec.seed)

    @property
    def to_move(self) -> str:
        return self.spec.white if self.board.turn == chess.WHITE else self.spec.black

    @property
    def total_ply(self) -> int:
        return OPENING_PLY + len(self.board.move_stack)

    def push(self, move: chess.Move, prob: float | None) -> None:
        if move not in self.board.legal_moves:
            raise ValueError(f"둘 수 없는 수다: {move} ({self.board.fen()})")
        self.board.push(move)
        self.probs.append(prob)

    def outcome(self) -> tuple[str, str] | None:
        """종료면 (결과, 사유), 아니면 None."""
        outcome = self.board.outcome()
        if outcome is not None:
            return outcome.result(), outcome.termination.name.lower()
        if self.board.is_repetition(3):
            return "1/2-1/2", "threefold_repetition"
        if self.board.is_fifty_moves():
            return "1/2-1/2", "fifty_moves"
        if self.total_ply >= MAX_PLY:
            return "1/2-1/2", "max_ply"
        return None

    def model_score(self) -> float:
        """평가 대상 쪽 점수 (1, 0.5, 0)."""
        result = self.outcome()
        if result is None:
            raise ValueError("대국이 끝나지 않았다")
        white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result[0]]
        return white if self.spec.model_color == chess.WHITE else 1.0 - white

    def pgn(self, headers: dict[str, str]) -> str:
        result, reason = self.outcome()
        game = chess.pgn.Game()
        game.setup(chess.Board(self.spec.fen))
        node = game
        for move, prob in zip(self.board.move_stack, self.probs, strict=True):
            node = node.add_variation(move)
            if prob is not None:
                node.comment = f"pi={prob:.4f}"
        game.headers["Event"] = "Task #13 대국"
        game.headers["White"] = self.spec.white
        game.headers["Black"] = self.spec.black
        game.headers["Result"] = result
        game.headers["Opening"] = str(self.spec.opening)
        game.headers["ModelColor"] = self.spec.key[1]
        game.headers["Seed"] = str(self.spec.seed)
        game.headers["Termination"] = reason
        game.headers["PlyCount"] = str(len(self.board.move_stack))
        game.headers["TotalPly"] = str(self.total_ply)
        for key, value in headers.items():
            game.headers[key] = value
        return str(game) + "\n\n"
