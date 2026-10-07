"""PGN → 국면 단위 데이터셋.

대상 기사가 둘 차례인 국면마다 (둔 수 직전의 FEN, 둔 수)를 한 레코드로 기록하고,
판 단위로 train/val을 분할한다.

    python -m src.data.preprocess --player tal [--include-fast] [--val-ratio 0.05] [--seed 0]
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from src.players import PLAYERS, Player

ROOT = Path(__file__).resolve().parents[2]

CLASSICAL = "classical"

# 이벤트명으로 판의 종류를 분류한다. 위에서부터 먼저 맞는 것이 그 판의 event_type이 된다.
# classical이 아닌 판은 기본 제외 대상이다.
EVENT_PATTERNS = [
    ("simul", r"simul|\bsim\d*\b"),
    ("blindfold", r"blind"),
    ("blitz", r"blitz"),
    # g/25, 30' 처럼 제한 시간이 이벤트명에 적힌 판은 속기다
    ("rapid", r"rapid|\brap\b|\bg/?\d+|\d+'"),
    ("tiebreak", r"tiebreak|\btb\b"),
    ("exhibition", r"\bexh"),
    ("media", r"\btv\b|internet"),
    # 컴퓨터의 도움을 받아 두는 대국. 기사 혼자 고른 수가 아니다
    ("advanced", r"advanced|man\+comp"),
]
EVENT_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in EVENT_PATTERNS]

# PGN 파일 일부가 UTF-8이 아니다. latin-1은 어떤 바이트열도 실패 없이 읽는다.
PGN_ENCODING = "latin-1"


def classify_event(event: str) -> str:
    for name, pattern in EVENT_PATTERNS:
        if pattern.search(event):
            return name
    return CLASSICAL


def player_color(player: Player, white: str, black: str) -> chess.Color | None:
    """대상 기사의 색. 양쪽 다 맞거나 둘 다 아니면 None."""
    is_white = player.matches(white)
    is_black = player.matches(black)
    if is_white == is_black:
        return None
    return chess.WHITE if is_white else chess.BLACK


def parse_year(date: str) -> int | None:
    year = date.split(".")[0]
    return int(year) if year.isdigit() else None


def extract_positions(game: chess.pgn.Game, color: chess.Color) -> list[tuple[int, str, str]]:
    """본선을 따라가며 `color` 차례마다 (ply, 둔 수 직전 FEN, 둔 수 UCI)를 모은다."""
    board = game.board()
    positions = []
    for ply, move in enumerate(game.mainline_moves()):
        if board.turn == color:
            positions.append((ply, board.fen(), move.uci()))
        board.push(move)
    return positions


def split_games(game_ids: list[int], val_ratio: float, seed: int) -> set[int]:
    """val에 들어갈 game_id 집합. 판 단위로 나누므로 한 판의 국면이 양쪽에 걸치지 않는다."""
    shuffled = sorted(game_ids)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[: round(len(shuffled) * val_ratio)])


def preprocess(
    player: Player,
    pgn_path: Path,
    out_dir: Path,
    include_fast: bool = False,
    val_ratio: float = 0.05,
    seed: int = 0,
) -> dict:
    # 비합법 수는 game.errors로 집계하므로 python-chess의 로그 출력은 끈다
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    records_by_game: dict[int, list[dict]] = {}
    excluded = Counter()
    excluded_events: dict[str, Counter] = {}
    total_games = 0

    with open(pgn_path, encoding=PGN_ENCODING) as f:
        while (game := chess.pgn.read_game(f)) is not None:
            game_id = total_games
            total_games += 1
            headers = game.headers

            color = player_color(player, headers.get("White", ""), headers.get("Black", ""))
            if color is None:
                excluded["name_mismatch"] += 1
                continue

            event = headers.get("Event", "")
            event_type = classify_event(event)
            if event_type != CLASSICAL and not include_fast:
                excluded[event_type] += 1
                excluded_events.setdefault(event_type, Counter())[event] += 1
                continue

            if game.errors:
                excluded["parse_error"] += 1
                continue

            positions = extract_positions(game, color)
            if not positions:
                excluded["no_moves"] += 1
                continue

            meta = {
                "game_id": game_id,
                "color": "white" if color == chess.WHITE else "black",
                "year": parse_year(headers.get("Date", "")),
                "eco": headers.get("ECO", ""),
                "result": headers.get("Result", "*"),
                "event_type": event_type,
            }
            records_by_game[game_id] = [
                {"fen": fen, "move": move, "ply": ply, **meta} for ply, fen, move in positions
            ]

    val_ids = split_games(list(records_by_game), val_ratio, seed)
    splits = {"train": [], "val": []}
    for game_id, records in records_by_game.items():
        splits["val" if game_id in val_ids else "train"].extend(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    all_records = splits["train"] + splits["val"]
    stats = {
        "pgn": str(pgn_path),
        "include_fast": include_fast,
        "val_ratio": val_ratio,
        "seed": seed,
        "total_games": total_games,
        "excluded_games": dict(excluded),
        "used_games": len(records_by_game),
        "games": {"train": len(records_by_game) - len(val_ids), "val": len(val_ids)},
        "positions": {name: len(records) for name, records in splits.items()},
        "positions_by_color": dict(Counter(r["color"] for r in all_records)),
        "positions_by_event_type": dict(Counter(r["event_type"] for r in all_records)),
        "unique_fens": len({r["fen"] for r in all_records}),
        "excluded_events": {
            event_type: dict(sorted(events.items()))
            for event_type, events in sorted(excluded_events.items())
        },
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--player", required=True, choices=sorted(PLAYERS))
    parser.add_argument("--include-fast", action="store_true",
                        help="속기·다면기·블라인드·시범 계열 판도 포함한다")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="기본값: data/processed/{player}")
    args = parser.parse_args()

    player = PLAYERS[args.player]
    out_dir = args.out_dir or ROOT / "data" / "processed" / args.player
    stats = preprocess(player, ROOT / player.pgn, out_dir,
                       include_fast=args.include_fast, val_ratio=args.val_ratio, seed=args.seed)
    summary = {k: v for k, v in stats.items() if k != "excluded_events"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
