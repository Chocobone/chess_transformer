"""Stockfish UCI 연결과 센티폰 → 승률 변환 (평가 기준 문서 5.4절, Task #13).

- 설정은 `Threads = 1`, `Hash` 고정이다. 시간 제한은 쓰지 않고 노드 수(C2) 또는 깊이(C4, 오프닝 선별)로 탐색량을 정한다
- 매 탐색 전에 해시를 비운다(`ucinewgame`, 구현 계획서 해석 3·8). 그래서 한 탐색의 결과가 (수순, 탐색량)만으로 정해지고,
  같은 엔진을 여러 대국이 나눠 써도 순서에 따라 결과가 바뀌지 않는다
- 승률 변환은 W(cp) = 1 / (1 + exp(−0.00368208 · cp))이다 (수행 계획서 결정 4). 베이스 모델(searchless_chess)이
  학습 라벨을 만들 때 쓴 Lichess 변환식과 같다. 메이트 점수는 이기는 쪽 1, 지는 쪽 0이다
"""

import math
import os
import shutil
from pathlib import Path

import chess
import chess.engine

WIN_K = 0.00368208
HASH_MB = 16
DEFAULT_BINARY = Path("runs/tools/stockfish/stockfish")  # gpu_train/install_stockfish.sh가 만든다
VERSION = "Stockfish 19"  # 기준 문서에 고정한 버전. 다른 버전이면 연결을 거부한다


def cp_to_win(cp: float) -> float:
    """센티폰(둘 차례 쪽 기준) → 승률."""
    return 1.0 / (1.0 + math.exp(-WIN_K * cp))


def win_prob(score: chess.engine.PovScore, pov: chess.Color) -> float:
    """Stockfish 점수 → pov 쪽 승률. 메이트는 이기는 쪽 1, 지는 쪽 0 (메이트까지 수와 무관)."""
    relative = score.pov(pov)
    if relative.is_mate():
        # Mate(+n)·MateGiven은 이긴 쪽, Mate(−n)·Mate(−0)은 진 쪽. 점수 비교가 이 순서를 지킨다
        return 1.0 if relative > chess.engine.Cp(0) else 0.0
    return cp_to_win(relative.score())


def find_binary(path: str | Path | None = None) -> Path:
    """실행 파일 경로: 인자 → 환경변수 STOCKFISH → runs/tools/stockfish/stockfish → PATH의 stockfish."""
    candidates = [path, os.environ.get("STOCKFISH"), DEFAULT_BINARY, shutil.which("stockfish")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Stockfish 실행 파일이 없다. bash gpu_train/install_stockfish.sh로 설치하거나 STOCKFISH를 지정한다")


class Stockfish:
    """Stockfish 프로세스 하나. 스레드 안전하지 않다 (동시에 쓰려면 프로세스를 여러 개 만든다)."""

    def __init__(self, binary: str | Path | None = None, hash_mb: int = HASH_MB, require_version: str | None = VERSION):
        self.binary = find_binary(binary)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.binary))
        self.version = self.engine.id.get("name", "")
        if require_version and self.version != require_version:
            self.engine.quit()
            raise RuntimeError(f"Stockfish 버전이 다르다: {self.version!r} (고정 {require_version!r})")
        self.engine.configure({"Threads": 1, "Hash": hash_mb})
        self.hash_mb = hash_mb
        self._game = 0

    def _new_game(self) -> object:
        """python-chess는 game 키가 바뀔 때 ucinewgame을 보낸다. 매번 새 키를 줘서 해시를 비운다."""
        self._game += 1
        return ("fresh", self._game)

    def move(self, board: chess.Board, nodes: int) -> chess.Move:
        """노드 수 제한으로 한 수 (C2). board의 수순 전체가 엔진에 넘어간다 (반복을 안다)."""
        result = self.engine.play(board, chess.engine.Limit(nodes=nodes), game=self._new_game())
        if result.move is None:
            raise RuntimeError(f"Stockfish가 수를 내지 않았다: {board.fen()}")
        return result.move

    def evaluate(self, board: chess.Board, depth: int, only: chess.Move | None = None
                 ) -> tuple[chess.Move, chess.engine.PovScore]:
        """고정 깊이 탐색 → (최선수, 점수). only를 주면 그 수만 허용해 탐색한다 (C4 모델 수, 해석 8)."""
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth), game=self._new_game(),
                                   root_moves=[only] if only is not None else None)
        pv, score = info.get("pv"), info.get("score")
        if not pv or score is None:
            raise RuntimeError(f"Stockfish 탐색 결과가 없다: {board.fen()}")
        if only is not None and pv[0] != only:
            raise RuntimeError(f"허용한 수와 다른 수가 나왔다: {pv[0]} != {only}")
        return pv[0], score

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
