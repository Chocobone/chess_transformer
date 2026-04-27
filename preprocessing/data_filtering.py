import glob
import os
import random

import chess.pgn
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_folder_to_uci(folder_path, output_dir, sample_prob=0.3):
    search_pattern = os.path.join(folder_path, "*.pgn")
    file_paths = glob.glob(search_pattern)

    all_games = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                except:
                    continue

                moves = list(game.mainline_moves())

                if len(moves) < 10:
                    continue

                # ✅ 랜덤 샘플링
                if random.random() > sample_prob:
                    continue

                uci_moves = " ".join(m.uci() for m in moves)

                all_games.append({
                    "Result": game.headers.get("Result", "1/2-1/2"),
                    "UCI_Moves": uci_moves,
                })

    df = pd.DataFrame(all_games)

    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train, val, test
