"""PyTorch 이식 모델을 원본 JAX 기준 출력과 비교한다.

    python scripts/verify_port.py --base base_model/270M.pt --reference base_model/reference_270M.npz

`dump_jax_reference.py`가 만든 기준 파일의 입력 토큰, 합법수 순서, 구간 로그확률을 대조한다.
모든 항목이 통과하면 종료 코드 0, 아니면 1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.transformer import encode_position, load_base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="convert.py가 만든 .pt")
    parser.add_argument("--reference", required=True, help="dump_jax_reference.py가 만든 npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="구간 로그확률 최대 절대 오차 허용치")
    args = parser.parse_args()

    with np.load(args.reference) as reference:
        fens = [str(fen) for fen in reference["fens"]]
        counts = reference["counts"]
        ref_moves = [str(move) for move in reference["moves"]]
        ref_sequences = reference["sequences"].astype(np.int64)
        ref_log_probs = reference["log_probs"]

    encoded = [encode_position(fen) for fen in fens]
    sequences = np.concatenate([tokens for tokens, _ in encoded])
    moves = [move.uci() for _, fen_moves in encoded for move in fen_moves]

    moves_match = moves == ref_moves
    tokens_match = sequences.shape == ref_sequences.shape and bool((sequences == ref_sequences).all())
    print(f"국면 {len(fens)}개, 합법수 {len(ref_moves)}개")
    print(f"합법수 순서 일치: {moves_match}")
    print(f"입력 토큰 일치: {tokens_match}")
    if not (moves_match and tokens_match):
        sys.exit(1)

    model = load_base(args.base, args.device).eval()
    tokens = torch.from_numpy(sequences).to(args.device)
    with torch.no_grad():
        log_probs = torch.cat([model(tokens[i:i + args.batch_size])
                               for i in range(0, len(tokens), args.batch_size)])
        win_probs = (log_probs.exp() @ model.bucket_values).cpu().numpy()
    log_probs = log_probs.cpu().numpy()
    bucket_values = model.bucket_values.cpu().numpy()
    ref_win_probs = np.exp(ref_log_probs) @ bucket_values

    # 확률이 사실상 0인 구간은 로그확률이 매우 작아 오차가 부풀려지므로 확률 오차도 함께 본다
    max_log_prob_error = float(np.abs(log_probs - ref_log_probs).max())
    max_prob_error = float(np.abs(np.exp(log_probs) - np.exp(ref_log_probs)).max())
    max_win_prob_error = float(np.abs(win_probs - ref_win_probs).max())

    best_match = 0
    for start, count in zip(np.cumsum(counts) - counts, counts):
        segment = slice(start, start + count)
        best_match += int(win_probs[segment].argmax() == ref_win_probs[segment].argmax())

    print(f"구간 로그확률 최대 절대 오차: {max_log_prob_error:.3e}")
    print(f"구간 확률 최대 절대 오차: {max_prob_error:.3e}")
    print(f"승률 최대 절대 오차: {max_win_prob_error:.3e}")
    print(f"최선수 일치: {best_match}/{len(fens)}")

    passed = max_log_prob_error <= args.tolerance and best_match == len(fens)
    print("결과: 통과" if passed else f"결과: 실패 (허용치 {args.tolerance:.0e})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
