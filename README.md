# Chess Transformer

AlphaZero 스타일의 체스 AI 프로젝트입니다. Transformer 기반 신경망과 MCTS(Monte Carlo Tree Search)를 결합하여 자기 대국(Self-Play)을 통해 학습합니다.

## 아키텍처 개요

```
PGN 기보 데이터
     ↓
데이터 필터링 & UCI 변환 (data_filtering.py)
     ↓
UCI 어휘 사전 구축 (tokenize.py)
     ↓
ChessTransformer 학습 (transfromer.py)
  ├── Self-Play → 학습 데이터 생성
  ├── MCTS → 수 선택
  └── Policy Head + Value Head 학습
```

## 모델 구조

`ChessTransformer`는 체스 보드를 64개 칸의 시퀀스로 처리합니다.

| 컴포넌트 | 내용 |
|---|---|
| 입력 | 64개 칸 × 기물 ID (0~12) |
| Embedding | 기물 임베딩 + 위치 임베딩 + 턴 임베딩 (dim=128) |
| Encoder | TransformerEncoder (4 layers, 4 heads) |
| Policy Head | Linear(128×64 → vocab_size) — 각 수의 사전 확률 |
| Value Head | Linear → ReLU → Linear → Tanh — 현재 플레이어 승률 (-1 ~ 1) |

## 프로젝트 구조

```
chess_transformer/
├── preprocessing/
│   ├── data_filtering.py   # PGN → UCI CSV 변환 및 train/val/test 분리
│   ├── tokenize.py         # UCI 어휘 사전, 합법 수 마스크, 상대 위치 행렬
│   └── sample/             # 샘플 PGN 파일 (lichess_elite_2016-09.pgn)
├── training/
│   ├── transfromer.py      # 모델, MCTS, Self-Play, 학습 루프
│   └── readme.md           # 구현 상세 설명
├── shell/
│   ├── preprocess.sh       # 전처리 실행 스크립트
│   ├── model_training.sh   # 학습 실행 스크립트 (SLURM)
│   ├── model_run.sh        # 전체 파이프라인 실행 스크립트 (SLURM)
│   └── rust_test.sh        # Rust 테스트 스크립트
└── .env.example
```

## 설치

```bash
pip install torch numpy pandas scikit-learn python-chess
```

## 사용법

### 1. 데이터 전처리

PGN 파일이 담긴 폴더를 지정하여 UCI 형식의 CSV로 변환합니다.

```python
from preprocessing.data_filtering import preprocess_folder_to_uci

train, val, test = preprocess_folder_to_uci(
    folder_path="path/to/pgn/",
    output_dir="data/",
    sample_prob=0.3   # 전체 기보 중 30% 랜덤 샘플링
)
```

- 10수 미만의 게임은 자동으로 제외됩니다.
- 출력: `data/train.csv`, `data/val.csv`, `data/test.csv`
- 분할 비율: train 80% / val 10% / test 10%

### 2. 모델 학습

```bash
python training/transfromer.py
```

Self-Play를 통해 학습 데이터를 직접 생성하며 50 에폭 학습합니다.

| 하이퍼파라미터 | 값 |
|---|---|
| Optimizer | AdamW (lr=1e-4) |
| Epochs | 50 |
| Self-Play / epoch | 5 게임 |
| Replay Buffer | 최대 5,000 샘플 |
| Batch Size | 128 |
| MCTS Simulations | 32 |

### 3. SLURM 클러스터 실행

```bash
sbatch shell/model_run.sh
```

GPU 1장, CPU 8코어, 메모리 32GB 기준으로 설정되어 있습니다.

## MCTS 동작 방식

1. **Select** — UCB 점수 기준으로 리프 노드까지 탐색
2. **Expand** — 신경망으로 합법 수 전체를 배치 평가
3. **Backup** — 가치(Value)를 루트까지 역전파 (상대방은 부호 반전)
4. **선택** — 방문 횟수(N)에 비례하여 최종 수 샘플링

```
UCB(s, a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

## UCI 어휘 사전

`build_uci_vocabulary()`는 체스판 위에서 가능한 모든 이동을 사전으로 구성합니다.

- Queen 방향 이동 (직선 + 대각선, 최대 7칸)
- Knight 이동 (L자, 8방향)
- 약 4,200개의 UCI 이동 토큰

합법 수가 아닌 이동은 `-1e9`로 마스킹되어 모델이 불가능한 수를 선택하지 않습니다.

## 학습 손실 함수

```
Loss = Loss_policy + Loss_value

Loss_policy = -Σ π(a) * log P(a|s)   # Cross-entropy (MCTS 정책과 비교)
Loss_value  = (v_pred - z)²           # MSE (실제 게임 결과와 비교)
```

- `π`: MCTS 방문 횟수 기반 정책
- `z`: 게임 결과 (+1 승, -1 패, 0 무)
