# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-style chess AI combining a Transformer neural network with Monte Carlo Tree Search (MCTS) and self-play training. The model learns entirely from self-play — no supervised learning from human games is required for training, though a PGN dataset is included for reference.

## Commands

**Install dependencies (no requirements.txt — install manually):**
```bash
pip install torch numpy pandas scikit-learn python-chess
```

**Run the full preprocessing pipeline:**
```bash
bash shell/preprocess.sh
# or directly:
python preprocessing/data_filtering.py
python preprocessing/tokenize.py
```

**Train the model:**
```bash
python training/transfromer.py
# On a SLURM cluster:
sbatch shell/model_training.sh
```

**Full pipeline on SLURM (preprocess → train → rust_test):**
```bash
sbatch shell/model_run.sh
```

**Build the Rust component (optional):**
```bash
cargo build --release
```

There is no test suite or linter configured.

## Architecture

### Data Pipeline

```
PGN files → data_filtering.py → train/val/test CSV (UCI moves)
                                      ↓
                              tokenize.py → UCI vocab (~4,200 moves) + legal move masks
```

- `preprocessing/data_filtering.py`: Parses PGN files, filters games with <10 moves, samples 30% randomly, outputs 80/10/10 train/val/test CSV split.
- `preprocessing/tokenize.py`: Builds a vocabulary of all legal UCI moves (queen-style directions × distances + knight jumps). Also generates `get_legal_move_mask(board, vocab)` and `get_relative_position_matrix()` (Manhattan distance between squares).

### Model (`training/transfromer.py`)

**Board encoding** (`encode_board`): 64-element flat array (row-major a1→h8). White pieces = 1–6, black pieces = 7–12, empty = 0.

**`ChessTransformer`:**
- Piece embedding (13 tokens → 128-dim) + position embedding (64 positions → 128-dim) + turn embedding (2 values → 128-dim)
- 4-layer `TransformerEncoder`, 4 attention heads, d_model=128
- **Policy head**: Linear(128×64 → ~4,200) — move logits
- **Value head**: Linear(128×64 → 128) → ReLU → Linear(128 → 1) → Tanh → scalar in [-1, 1]

**MCTS (`Node`, `run_mcts`, `expand_batch`):**
- UCB formula: `Q(s,a) + c · P(s,a) · sqrt(N(s)) / (1 + N(s,a))`
- 32 simulations per move; batch expansion with batch size 8 for parallel leaf evaluation

**Self-play (`self_play`):**
- Plays complete games using MCTS; returns `(board_state, turn, mcts_policy_vector, game_result)`
- Game result: white win → z=1, black win → z=−1, draw → z=0; value is negated for the opposing player

**Training loop (`train`):**
- 50 epochs; 5 self-play games per epoch
- Replay buffer: `deque(maxlen=5000)`; training batch size 128
- Optimizer: AdamW (lr=1e-4)
- Loss: `loss_policy` (cross-entropy vs. MCTS visit distribution) + `loss_value` (MSE vs. game result)

### Key Design Decisions

- The model is **input as flat 64-token sequence** — each token is one square — so the Transformer attends over board squares directly.
- MCTS uses **neural priors from the policy head** and **neural value from the value head** instead of random rollouts.
- No checkpoint saving or inference entry point exists in the current code; the training loop runs to completion and the model is held in memory.
