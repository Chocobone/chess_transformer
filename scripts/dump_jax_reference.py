"""원본 JAX 모델의 기준 출력 추출. GPU 서버의 원본 searchless_chess 환경에서 실행한다.

    PYTHONPATH=<searchless_chess 저장소의 부모 폴더> python scripts/dump_jax_reference.py \\
        --params base_model/270M_params.npz --size 270M --out base_model/reference_270M.npz

`scripts/reference_fens.txt`의 각 국면에서 모든 합법수에 대한 구간 로그확률을 저장한다.
입력 토큰도 원본 토크나이저로 만들어 함께 저장하므로, `verify_port.py`는 토큰화·행동 번호·
모델 출력을 모두 대조할 수 있다. 가중치는 `export_haiku_params.py`가 만든 npz에서 읽는다.
"""

import argparse
from pathlib import Path

import chess
import haiku as hk
import jax
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine

MODEL_SIZES = {
    "9M": dict(num_layers=8, embedding_dim=256, num_heads=8),
    "136M": dict(num_layers=8, embedding_dim=1024, num_heads=8),
    "270M": dict(num_layers=16, embedding_dim=1024, num_heads=8),
}
NUM_RETURN_BUCKETS = 128
BATCH_SIZE = 256


def build_sequences(fen: str) -> tuple[np.ndarray, list[str]]:
    """원본 `ActionValueEngine.analyse`와 같은 방식으로 입력을 만든다."""
    board = chess.Board(fen)
    moves = engine.get_ordered_legal_moves(board)
    actions = np.array([utils.MOVE_TO_ACTION[m.uci()] for m in moves], dtype=np.int32)[:, None]
    dummy_returns = np.zeros((len(moves), 1), dtype=np.int32)
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    fens = np.stack([tokenized_fen] * len(moves))
    return np.concatenate([fens, actions, dummy_returns], axis=1), [m.uci() for m in moves]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", required=True, help="export_haiku_params.py가 만든 npz")
    parser.add_argument("--size", required=True, choices=sorted(MODEL_SIZES))
    parser.add_argument("--fens", type=Path,
                        default=Path(__file__).with_name("reference_fens.txt"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=NUM_RETURN_BUCKETS,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
        **MODEL_SIZES[args.size],
    )
    predictor = transformer.build_transformer_predictor(config=config)

    params: dict[str, dict[str, np.ndarray]] = {}
    with np.load(args.params) as flat:
        for key in flat.files:
            module, name = key.rsplit("/", 1)
            params.setdefault(module, {})[name] = flat[key]
    predict = jax.jit(predictor.predict)

    fens = args.fens.read_text().split("\n")
    fens = [fen for fen in fens if fen]
    sequences, moves, counts = [], [], []
    for fen in fens:
        fen_sequences, fen_moves = build_sequences(fen)
        sequences.append(fen_sequences)
        moves.extend(fen_moves)
        counts.append(len(fen_moves))
    sequences = np.concatenate(sequences)

    # 배치 크기를 고정해 jit 재컴파일을 피한다. 남는 자리는 0으로 채웠다가 잘라 낸다.
    padded = np.pad(sequences, ((0, -len(sequences) % BATCH_SIZE), (0, 0)))
    outputs = [
        np.asarray(predict(params=params, targets=padded[i:i + BATCH_SIZE], rng=None))[:, -1]
        for i in range(0, len(padded), BATCH_SIZE)
    ]
    log_probs = np.concatenate(outputs)[: len(sequences)].astype(np.float32)

    np.savez(
        args.out,
        fens=np.array(fens),
        counts=np.array(counts, dtype=np.int32),
        moves=np.array(moves),
        sequences=sequences.astype(np.int32),
        log_probs=log_probs,
    )
    print(f"국면 {len(fens)}개, 합법수 {len(sequences)}개 → {args.out}")


if __name__ == "__main__":
    main()
