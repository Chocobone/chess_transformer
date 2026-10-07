import json

import chess
import pytest

from src.data.preprocess import (
    classify_event,
    parse_year,
    player_color,
    preprocess,
    split_games,
)
from src.players import PLAYERS, Player

HERO = Player(pgn="", name_prefix="Hero,")

SAMPLE_PGN = """\
[Event "Classic A"]
[Date "1960.01.02"]
[White "Hero, Test"]
[Black "Other, A"]
[Result "1-0"]
[ECO "C20"]

1.e4 e5 2.Nf3 Nc6 1-0

[Event "Classic B"]
[Date "1961.??.??"]
[White "Other, B"]
[Black "Hero,T"]
[Result "0-1"]
[ECO "B00"]

1.e4 d5 2.exd5 Qxd5 0-1

[Event "City blitz"]
[Date "1962.??.??"]
[White "Hero, Test"]
[Black "Other, C"]
[Result "1/2-1/2"]
[ECO "A00"]

1.d4 d5 1/2-1/2

[Event "Broken"]
[Date "1963.??.??"]
[White "Hero, Test"]
[Black "Other, D"]
[Result "1-0"]
[ECO "A00"]

1.e4 e5 2.Ke3 1-0

[Event "Stranger"]
[Date "1964.??.??"]
[White "Heron, X"]
[Black "Other, E"]
[Result "1-0"]
[ECO "A00"]

1.e4 1-0

[Event "Empty"]
[Date "1965.??.??"]
[White "Hero, Test"]
[Black "Other, F"]
[Result "*"]
[ECO "A00"]

*
"""


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def sample_pgn(tmp_path):
    path = tmp_path / "sample.pgn"
    path.write_text(SAMPLE_PGN, encoding="latin-1")
    return path


@pytest.mark.parametrize("name", ["Karpov, Anatoly", "Karpov,Ana", "Karpov,An"])
def test_karpov_name_variants_match(name):
    assert PLAYERS["karpov"].matches(name)


@pytest.mark.parametrize("name", ["Kasparov, Gary", "Kasparov,G", "Karpova, X", "Tal, Mihail"])
def test_karpov_does_not_match_others(name):
    assert not PLAYERS["karpov"].matches(name)


def test_player_color():
    assert player_color(HERO, "Hero, Test", "Other, A") == chess.WHITE
    assert player_color(HERO, "Other, A", "Hero,T") == chess.BLACK
    assert player_color(HERO, "Hero, A", "Hero, B") is None
    assert player_color(HERO, "Other, A", "Other, B") is None


@pytest.mark.parametrize("event, expected", [
    ("URS-ch25", "classical"),
    ("Candidats playoff-4pl", "classical"),
    ("Tallinn m", "classical"),
    ("Simferopol", "classical"),
    ("Moscow-Leningrad m", "classical"),
    ("Wch-blitz", "blitz"),
    ("Amber Rapidplay", "rapid"),
    ("Tilburg rapid20", "rapid"),
    ("Varna rap m", "rapid"),
    ("Match g/25", "rapid"),
    ("Santiago m g25/30", "rapid"),
    ("Budapest m 30'", "rapid"),
    ("Karpov-Boca Simul", "simul"),
    ("Tbilisi sim m", "simul"),
    ("Koszalin sim1", "simul"),
    ("Disney rapid sim", "simul"),
    ("Amber-blind 2nd", "blindfold"),
    ("Jakarta m 25' blindfold", "blindfold"),
    ("GpB Tiebreak", "tiebreak"),
    ("Platja d'Aro Legends TB", "tiebreak"),
    ("The Hague exh3", "exhibition"),
    ("Classical Exhibition Match", "exhibition"),
    ("Moscow TV", "media"),
    ("Internet FIN m", "media"),
    ("Leon Man+Comp m", "advanced"),
])
def test_classify_event(event, expected):
    assert classify_event(event) == expected


def test_parse_year():
    assert parse_year("1961.??.??") == 1961
    assert parse_year("????.??.??") is None


def test_split_is_deterministic_and_sized():
    ids = list(range(100))
    val = split_games(ids, 0.05, seed=0)
    assert len(val) == 5
    assert val == split_games(ids[::-1], 0.05, seed=0)
    assert val != split_games(ids, 0.05, seed=1)


def test_preprocess_extracts_only_player_moves(sample_pgn, tmp_path):
    out = tmp_path / "out"
    stats = preprocess(HERO, sample_pgn, out, val_ratio=0.0)
    records = read_jsonl(out / "train.jsonl")

    assert read_jsonl(out / "val.jsonl") == []
    assert stats["total_games"] == 6
    assert stats["used_games"] == 2
    assert stats["excluded_games"] == {
        "blitz": 1, "parse_error": 1, "name_mismatch": 1, "no_moves": 1,
    }
    assert stats["excluded_events"] == {"blitz": {"City blitz": 1}}
    assert stats["positions_by_color"] == {"white": 2, "black": 2}

    white = [r for r in records if r["game_id"] == 0]
    assert [(r["ply"], r["move"]) for r in white] == [(0, "e2e4"), (2, "g1f3")]
    assert white[0]["fen"] == chess.STARTING_FEN
    assert white[1]["fen"] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    assert white[0]["year"] == 1960 and white[0]["eco"] == "C20"
    assert white[0]["result"] == "1-0" and white[0]["event_type"] == "classical"

    black = [r for r in records if r["game_id"] == 1]
    assert [(r["ply"], r["move"]) for r in black] == [(1, "d7d5"), (3, "d8d5")]
    assert black[0]["fen"] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert black[0]["color"] == "black"

    for r in records:
        assert chess.Move.from_uci(r["move"]) in chess.Board(r["fen"]).legal_moves


def test_include_fast_keeps_event_type(sample_pgn, tmp_path):
    out = tmp_path / "out"
    stats = preprocess(HERO, sample_pgn, out, include_fast=True, val_ratio=0.0)
    assert stats["used_games"] == 3
    assert stats["positions_by_event_type"] == {"classical": 4, "blitz": 1}


def test_split_is_by_game(sample_pgn, tmp_path):
    out = tmp_path / "out"
    stats = preprocess(HERO, sample_pgn, out, val_ratio=0.5)
    train_ids = {r["game_id"] for r in read_jsonl(out / "train.jsonl")}
    val_ids = {r["game_id"] for r in read_jsonl(out / "val.jsonl")}
    assert stats["games"] == {"train": 1, "val": 1}
    assert train_ids and val_ids and not (train_ids & val_ids)
