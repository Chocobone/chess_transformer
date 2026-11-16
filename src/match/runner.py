"""동시 대국 실행과 기록 (평가 기준 문서 5.3절, Task #13 2단계).

- 여러 대국을 동시에 진행한다. 한 바퀴마다 모델 차례 국면을 모델별로 묶어 계산하고(해석 10), Stockfish 차례 국면은
  Stockfish 프로세스 풀에 나눈다. Stockfish 계산은 다른 스레드에서 돌려 모델 계산과 겹친다
- 끝난 대국마다 `games.pgn`에 한 판을 덧붙이고 `index.jsonl`에 한 줄을 적는다. 다시 실행하면 `index.jsonl`에 있는
  대국은 건너뛴다. 진행 중이던 대국은 처음부터 다시 둔다
- 실행 기록은 `runs.jsonl`에 실행마다 한 줄씩 남는다 (장비, 버전, 설정, 시간)

사용:
    python -m src.match.play --model R1 --opponent base --openings 0-199 --out runs/gpu/match/C1_R1
    python -m src.match.play --model base --opponent sf --nodes 10000 --openings 0-9 --out runs/gpu/match/cal_base_10000
"""

import datetime
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import chess

from src.match.game import GameSpec, MatchGame
from src.match.policy import PolicyBank

SF_ID = "sf"


class Opponent(Protocol):
    """모델이 아닌 상대 (Stockfish). boards의 각 국면에 수 하나."""
    name: str
    headers: dict[str, str]

    def moves(self, boards: list[chess.Board]) -> list[chess.Move]: ...


class StockfishPool:
    """Stockfish 프로세스 여러 개. 노드 수 고정, 매 수 해시를 비우므로 어느 프로세스가 두든 같은 수다 (해석 3)."""

    name = SF_ID

    def __init__(self, nodes: int, workers: int, binary: str | None = None):
        from src.match.stockfish import Stockfish

        self.nodes = nodes
        self.engines = [Stockfish(binary) for _ in range(workers)]
        self._free: queue.Queue = queue.Queue()
        for engine in self.engines:
            self._free.put(engine)
        self._executor = ThreadPoolExecutor(max_workers=workers)
        first = self.engines[0]
        from src.match.policy import sha256
        self.headers = {"Engine": first.version, "EngineSha256": sha256(first.binary.resolve()),
                        "Nodes": str(nodes), "EngineThreads": "1", "EngineHash": str(first.hash_mb)}

    def _one(self, board: chess.Board) -> chess.Move:
        engine = self._free.get()
        try:
            return engine.move(board, self.nodes)
        finally:
            self._free.put(engine)

    def moves(self, boards: list[chess.Board]) -> list[chess.Move]:
        return list(self._executor.map(self._one, boards))

    def close(self) -> None:
        self._executor.shutdown()
        for engine in self.engines:
            engine.close()


def read_done(out: Path) -> set[tuple[int, str]]:
    path = out / "index.jsonl"
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add((row["opening"], row["model_color"]))
    return done


def run(specs: list[GameSpec], bank: PolicyBank, out: Path, opponent: Opponent | None = None, concurrent: int = 64,
        log_every: int = 50, meta: dict | None = None) -> dict:
    """specs의 대국 가운데 끝나지 않은 것을 모두 둔다. 반환: 이번 실행의 요약."""
    out.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    pending = [spec for spec in specs if spec.key not in done]
    models = {spec.model for spec in specs} | {spec.opponent for spec in specs}
    if SF_ID in models and opponent is None:
        raise ValueError("Stockfish 상대가 필요하다")
    extra = dict(opponent.headers) if opponent is not None and SF_ID in models else {}

    started = time.time()
    summary = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "requested": len(specs),
               "skipped": len(specs) - len(pending), "played": 0, "rounds": 0, "model_moves": 0, "sf_moves": 0,
               "model_seconds": 0.0, "sf_seconds": 0.0, "concurrent": concurrent, **(meta or {})}
    active: list[MatchGame] = []
    queue_ = list(reversed(pending))
    sf_executor = ThreadPoolExecutor(max_workers=1)
    pgn_file = open(out / "games.pgn", "a", encoding="utf-8")
    index_file = open(out / "index.jsonl", "a", encoding="utf-8")
    try:
        while queue_ or active:
            while queue_ and len(active) < concurrent:
                active.append(MatchGame(queue_.pop()))
            summary["rounds"] += 1

            sf_games = [game for game in active if game.to_move == SF_ID]
            sf_future = None
            if sf_games:
                sf_started = time.time()
                sf_future = sf_executor.submit(opponent.moves, [game.board for game in sf_games])

            model_started = time.time()
            by_model: dict[str, list[MatchGame]] = {}
            for game in active:
                if game.to_move != SF_ID:
                    by_model.setdefault(game.to_move, []).append(game)
            for model_id in sorted(by_model):
                games = by_model[model_id]
                chosen = bank.moves(model_id, [g.board for g in games], [g.generator for g in games])
                for game, (move, prob) in zip(games, chosen, strict=True):
                    game.push(move, prob)
                summary["model_moves"] += len(games)
            summary["model_seconds"] += time.time() - model_started

            if sf_future is not None:
                for game, move in zip(sf_games, sf_future.result(), strict=True):
                    game.push(move, None)
                summary["sf_moves"] += len(sf_games)
                summary["sf_seconds"] += time.time() - sf_started

            still = []
            for game in active:
                result = game.outcome()
                if result is None:
                    still.append(game)
                    continue
                headers = {**extra, "Concurrent": str(concurrent),
                           "WhiteSha256": bank.sha256.get(game.spec.white, ""),
                           "BlackSha256": bank.sha256.get(game.spec.black, "")}
                pgn_file.write(game.pgn(headers))
                pgn_file.flush()
                index_file.write(json.dumps({
                    "opening": game.spec.opening, "model_color": game.spec.key[1], "model": game.spec.model,
                    "opponent": game.spec.opponent, "seed": game.spec.seed, "result": result[0],
                    "termination": result[1], "score": game.model_score(), "ply": len(game.board.move_stack),
                }) + "\n")
                index_file.flush()
                summary["played"] += 1
                if log_every and summary["played"] % log_every == 0:
                    print(f"[{datetime.datetime.now():%H:%M:%S}] {summary['played']}/{len(pending)}국", flush=True)
            active = still
    finally:
        pgn_file.close()
        index_file.close()
        sf_executor.shutdown()
        summary["seconds"] = round(time.time() - started, 1)
        summary["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary
