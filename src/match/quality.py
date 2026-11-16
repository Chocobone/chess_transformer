"""C4 Stockfish 기준 수 품질 탐색 (평가 기준 문서 5.4절, Task #13 구현 계획서 해석 8).

S_val 국면마다 Stockfish 최선수를 고정 깊이로 한 번 탐색하고, 모델 1순위 수가 최선수와 다르면 그 수만 허용해 같은
깊이로 한 번 더 탐색한다. 손실 ℓ = W(최선수) − W(모델 수)이고, W는 둘 차례 쪽 승률이다. 탐색 불일치로 ℓ < 0이면
0으로 자르고 그 수를 기록한다. 모델이 늘어도 새 (국면, 수)만 탐색하도록 결과를 캐시한다.

모델 1순위 수는 Task #12 덤프(`{모델}.npz`, `base.npz`)의 log π에서 고른다 (`metrics.segment_argmax`, 동률이면 행동
번호가 작은 수). 합법수 순서는 캐시·덤프와 같은 행동 번호 순(`ordered_legal_moves`)이다. 국면 FEN은 val 기록에서 읽는다.

    python -m src.match.quality --dump runs/gpu/dump --models base R1 R8 R2 --depth 18 --workers 16 --out runs/gpu/quality

`{out}`에 남는 파일:
    best.jsonl       국면 i → 최선수, 점수, W (캐시)
    moves.jsonl      (국면 i, 수) → 점수, W (캐시, 최선수가 아닌 수만)
    {모델}.npz       loss[N] float64, move_is_best[N] bool, clipped[N] bool (잘린 음수)
    quality.json     설정, Stockfish 버전·SHA-256, 모델별 요약, 시간
"""

import argparse
import datetime
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np

from src.eval.metrics import segment_argmax
from src.match.stockfish import Stockfish, win_prob
from src.model.actions import ordered_legal_moves
from src.players import PLAYERS

PLAYER_ORDER = tuple(PLAYERS)  # 덤프(src/eval/dump.py)와 같은 순서


def score_json(score: chess.engine.PovScore, turn: chess.Color) -> dict:
    relative = score.pov(turn)
    if relative.is_mate():
        return {"mate": relative.mate()}
    return {"cp": relative.score()}


def load_fens(dump: Path, data_root: Path | None) -> list[str]:
    """덤프의 국면 순서(기사 순서대로 val 이어 붙임)로 FEN을 읽고, 기사·대국·ply·합법수 수가 덤프와 같은지 확인한다."""
    from src.eval.dump import data_dir_of, read_records

    with np.load(dump / "positions.npz") as data:
        player, game, ply, offsets = data["player"], data["game"], data["ply"], data["offsets"]
    fens, players, games, plies = [], [], [], []
    for index, name in enumerate(PLAYER_ORDER):
        for record in read_records(data_dir_of(name, data_root), "val"):
            fens.append(record["fen"])
            players.append(index)
            games.append(record["game_id"])
            plies.append(record["ply"])
    if len(fens) != len(player):
        raise ValueError(f"val 국면 수({len(fens)})가 positions.npz({len(player)})와 다르다")
    if not (np.array_equal(players, player) and np.array_equal(games, game) and np.array_equal(plies, ply)):
        raise ValueError("val 기록의 기사·대국·ply가 positions.npz와 다르다")
    counts = np.diff(offsets)
    for i in range(0, len(fens), max(1, len(fens) // 500)):  # 표본 확인 (전수는 탐색 때 한다)
        if chess.Board(fens[i]).legal_moves.count() != counts[i]:
            raise ValueError(f"국면 {i}의 합법수 수가 덤프와 다르다")
    return fens


def top1_moves(dump: Path, model: str, fens: list[str]) -> list[str]:
    """모델의 1순위 수 (UCI). 국면마다 합법수 수를 확인한다."""
    with np.load(dump / "positions.npz") as data:
        offsets = data["offsets"]
    with np.load(dump / f"{model}.npz") as data:
        log_pi = data["log_pi"]
    if len(log_pi) != offsets[-1]:
        raise ValueError(f"{model}.npz의 길이가 positions.npz와 다르다")
    choice = segment_argmax(log_pi, offsets)
    moves = []
    for i, fen in enumerate(fens):
        legal = ordered_legal_moves(chess.Board(fen))
        if len(legal) != offsets[i + 1] - offsets[i]:
            raise ValueError(f"국면 {i}의 합법수 수가 덤프와 다르다")
        moves.append(legal[int(choice[i])].uci())
    return moves


def read_cache(path: Path, key) -> dict:
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[key(row)] = row
    return out


class Searcher:
    """Stockfish 프로세스 풀. 매 탐색 해시를 비우므로 어느 프로세스가 맡든 같은 결과다."""

    def __init__(self, workers: int, depth: int, binary: str | None = None, factory=Stockfish):
        self.depth = depth
        self.engines = [factory(binary) for _ in range(workers)]
        self._free: queue.Queue = queue.Queue()
        for engine in self.engines:
            self._free.put(engine)
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def search(self, fen: str, only: str | None) -> dict:
        engine = self._free.get()
        try:
            board = chess.Board(fen)
            move, score = engine.evaluate(board, self.depth, chess.Move.from_uci(only) if only else None)
        finally:
            self._free.put(engine)
        return {"move": move.uci(), "score": score_json(score, board.turn), "w": win_prob(score, board.turn)}

    def close(self) -> None:
        self.executor.shutdown()
        for engine in self.engines:
            engine.close()


def run_searches(searcher: Searcher, jobs: list[tuple], path: Path, label: str, log_every: int = 1000) -> None:
    """jobs: (i, fen, only) 목록. 끝나는 대로 path에 한 줄씩 덧붙인다."""
    lock = threading.Lock()
    done = [0]
    with open(path, "a", encoding="utf-8") as f:
        def one(job):
            i, fen, only = job
            row = {"i": i, **searcher.search(fen, only)}
            if only is not None:
                row["only"] = only
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[0] += 1
                if log_every and done[0] % log_every == 0:
                    print(f"[{datetime.datetime.now():%H:%M:%S}] {label} {done[0]}/{len(jobs)}", flush=True)
        list(searcher.executor.map(one, jobs))


def compute(dump: Path, models: list[str], out: Path, searcher: Searcher, fens: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    top1 = {model: top1_moves(dump, model, fens) for model in models}
    timing = {}

    best = read_cache(out / "best.jsonl", lambda r: r["i"])
    jobs = [(i, fens[i], None) for i in range(len(fens)) if i not in best]
    started = time.time()
    run_searches(searcher, jobs, out / "best.jsonl", "최선수")
    timing["best"] = {"searches": len(jobs), "seconds": round(time.time() - started, 1)}
    best = read_cache(out / "best.jsonl", lambda r: r["i"])

    cached = read_cache(out / "moves.jsonl", lambda r: (r["i"], r["only"]))
    needed = sorted({(i, top1[m][i]) for m in models for i in range(len(fens)) if top1[m][i] != best[i]["move"]})
    jobs = [(i, fens[i], move) for i, move in needed if (i, move) not in cached]
    started = time.time()
    run_searches(searcher, jobs, out / "moves.jsonl", "모델 수")
    timing["moves"] = {"searches": len(jobs), "seconds": round(time.time() - started, 1)}
    cached = read_cache(out / "moves.jsonl", lambda r: (r["i"], r["only"]))

    summary = {}
    for model in models:
        loss = np.zeros(len(fens))
        is_best = np.zeros(len(fens), dtype=bool)
        clipped = np.zeros(len(fens), dtype=bool)
        for i, move in enumerate(top1[model]):
            if move == best[i]["move"]:
                is_best[i] = True
                continue
            value = best[i]["w"] - cached[(i, move)]["w"]
            if value < 0:
                clipped[i] = True
                value = 0.0
            loss[i] = value
        np.savez(out / f"{model}.npz", loss=loss, move_is_best=is_best, clipped=clipped)
        summary[model] = {"mean_loss": float(loss.mean()), "best_share": float(is_best.mean()),
                          "clipped": int(clipped.sum()), "blunder_share": float((loss >= 0.20).mean())}
    return {"summary": summary, "timing": timing, "positions": len(fens)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C4 Stockfish 기준 수 품질 (평가 기준 문서 5.4절)")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="덤프의 모델 ID (base 포함)")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--stockfish", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.match.policy import sha256

    fens = load_fens(args.dump, args.data_root)
    config_path = args.out / "config.json"
    config = {"dump": str(args.dump), "depth": args.depth}
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise SystemExit(f"{config_path}의 설정과 다르다 (캐시를 다른 깊이로 섞지 않는다)")
    args.out.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    started = datetime.datetime.now()
    searcher = Searcher(args.workers, args.depth, args.stockfish)
    try:
        first = searcher.engines[0]
        engine = {"version": first.version, "sha256": sha256(first.binary.resolve()), "threads": 1,
                  "hash_mb": first.hash_mb}
        result = compute(args.dump, args.models, args.out, searcher, fens)
    finally:
        searcher.close()
    result.update({"depth": args.depth, "models": args.models, "workers": args.workers, "engine": engine,
                   "started": started.isoformat(timespec="seconds"),
                   "finished": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(args.out / "quality.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
