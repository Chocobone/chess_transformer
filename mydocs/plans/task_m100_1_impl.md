# Task #1: 기사별 기보(PGN) 전처리 및 LoRA 파인튜닝 학습 코드 작성 — 구현 계획서

## 공통: 디렉터리 구조

```
chess_transformer/
├── requirements.txt            ← torch, chess, numpy, pytest (Stage 0)
├── pytest.ini                  ← 테스트 경로 설정 (Stage 0)
├── docker/                     ← 로컬 개발 컨테이너 (Stage 0)
├── .gitignore                  ← data/, runs/, checkpoints/, base_model/*/, __pycache__ (Stage 0)
├── src/
│   ├── players.py              ← 기사별 설정 (PGN 경로, 이름 매칭 규칙)
│   ├── data/
│   │   ├── preprocess.py       ← PGN → 국면 데이터셋 (Stage 1)
│   │   └── dataset.py          ← 학습용 Dataset / collate / 합법수 한도 묶음 분할 (Stage 4)
│   ├── model/
│   │   ├── tokenizer.py        ← FEN 77토큰화 (Stage 2)
│   │   ├── actions.py          ← 1,968 행동 공간 (Stage 2)
│   │   ├── transformer.py      ← 베이스 모델 PyTorch 이식 (Stage 2)
│   │   ├── convert.py          ← Haiku 체크포인트 → state_dict (Stage 2)
│   │   └── lora.py             ← LoRA 주입·저장·로드 (Stage 3)
│   └── train/
│       ├── policy.py           ← 합법수별 q → 정책 (Stage 3)
│       ├── losses.py           ← 모방·기력·스타일 항 (Stage 3)
│       └── train.py            ← 학습 루프 (Stage 4)
├── scripts/
│   ├── export_haiku_params.py  ← orbax 체크포인트 → npz, GPU 서버용 (Stage 2)
│   ├── dump_jax_reference.py   ← 원본 JAX 기준 출력 추출, GPU 서버용 (Stage 2)
│   ├── reference_fens.txt      ← 이식 검증용 국면 53개 (Stage 2)
│   ├── verify_port.py          ← 이식 수치 비교 (Stage 2)
│   └── cache_base_q.py         ← 동결 베이스 q 사전 계산 (Stage 4)
└── tests/                      ← pytest 단위 테스트
```

생성 데이터(`data/`), 체크포인트, 어댑터 가중치는 git에 올리지 않는다.

**로컬 환경**: 로컬 개발 컨테이너(`docker/`, Python 3.13)에서 구현과 단위 테스트를 수행한다. 실제 가중치 검증과 학습은 GPU 서버에서 수행한다.

---

## 0단계: 개발 환경 설정

2026-10-07 작업지시자 지시로 추가한 단계다. 환경 설정을 끝낸 뒤 1단계를 시작한다.

### 목표

이후 단계가 코드 작성에만 집중할 수 있도록 로컬 실행 환경, 의존성 명세, 저장소 골격, 테스트 실행 체계를 먼저 갖춘다.

### 구현

- **로컬 컨테이너**: `docker/Dockerfile`, `docker/compose.yml` (CPU용 `torch`, `chess`, `numpy`, `pytest` 설치), `.gitattributes` (셸 스크립트 LF 고정)
- **`requirements.txt`**: `torch`, `chess`, `numpy`, `pytest`. GPU 서버에서는 CUDA에 맞는 `torch`를 먼저 설치한 뒤 이 파일을 설치한다.
- **`.gitignore`**: `data/`, `runs/`, `checkpoints/`, `base_model/` 하위 체크포인트, 파이썬 캐시
- **패키지 골격**: `src/`, `src/data/`, `src/model/`, `src/train/`의 `__init__.py`. `scripts/`는 파일이 처음 생기는 2단계에서 만든다.
- **`pytest.ini`**: 테스트 경로 `tests/`, 저장소 루트를 import 경로에 추가
- **`tests/test_env.py`**: 환경 점검 테스트 (torch 순전파·역전파, python-chess 합법수 생성, 세 기사 PGN 읽기, `src` 패키지 import)

### 검증

- 컨테이너 안에서 `torch`, `chess`, `numpy`, `pytest` import 및 버전 확인
- `pytest` 통과
- `git status`에서 생성 데이터·캐시가 추적 대상으로 잡히지 않음

---

## 1단계: PGN 전처리 파이프라인

### 목표

세 기사의 PGN을 "대상 기사가 둘 차례인 국면" 단위 데이터셋으로 변환하고, 판 단위로 train/val을 분할한다.

### 구현

**`src/players.py`** — 기사별 설정:

```python
PLAYERS = {
    "tal":     Player(pgn="model_tal/Tal.pgn",         name_prefix="Tal,"),
    "karpov":  Player(pgn="model_karpov/Karpov.pgn",   name_prefix="Karpov,"),
    "smyslov": Player(pgn="model_smyslov/Smyslov.pgn", name_prefix="Smyslov,"),
}
```

`Karpov,` 접두어는 `Karpov, Anatoly` / `Karpov,Ana` / `Karpov,An` 3종을 모두 잡고 `Kasparov, Gary`는 잡지 않는다.

**`src/data/preprocess.py`** — 전처리 CLI:

```
python -m src.data.preprocess --player tal [--include-fast] [--val-ratio 0.05] [--seed 0]
```

처리 규칙:

- **색 판별**: White/Black 태그 중 접두어가 맞는 쪽이 대상 기사. 양쪽 다 맞거나 둘 다 아니면 그 판은 건너뛴다.
- **이벤트 필터 (기본 제외)**: 이벤트명을 대소문자 무시 정규식으로 분류해 `event_type`을 정하고, `classical`이 아니면 제외한다. 분류는 `simul`(`simul`, 단어 `sim`·`sim1`), `blindfold`(`blind`), `blitz`, `rapid`(`rapid`, 단어 `rap`, `g/25`, `30'` 같은 제한 시간 표기), `tiebreak`(`tiebreak`, 단어 `TB`), `exhibition`(`exh`로 시작하는 단어), `media`(단어 `TV`, `internet`), `advanced`(`advanced`, `man+comp`) 순으로 먼저 맞는 것을 쓴다. `--include-fast`로 끌 수 있다. 제외 여부와 무관하게 `event_type` 필드를 남긴다. (1단계에서 실제 이벤트명을 조사해 확정한 규칙이다.)
- **국면 추출**: python-chess로 본선을 따라가며 대상 기사 차례마다 (둔 수 직전의 FEN, 둔 수 UCI)를 기록한다. 상대 차례 국면은 기록하지 않는다.
- **오류 처리**: 비합법 수 등으로 파싱이 실패한 판, 수가 하나도 없는 판은 통째로 건너뛰고 건수를 집계한다. PGN은 `latin-1`로 읽는다 (카르포프 파일이 UTF-8이 아니다).
- **분할**: 판 단위로 섞어 train/val 분할 (기본 95/5, 고정 seed). 같은 판의 국면이 양쪽에 걸치지 않는다.
- **결과 `*` 판**: 수 자체는 실제로 둔 수이므로 포함한다.

출력 (`data/processed/{player}/`):

- `train.jsonl`, `val.jsonl` — 한 줄에 한 국면: `{"fen", "move", "game_id", "ply", "color", "year", "eco", "result", "event_type"}`
- `stats.json` — 전체 판 수, 제외 판 수(사유별), 파싱 실패 수, 사용 판 수, 국면 수, 색별 국면 수

### 검증

- 단위 테스트: 이름 매칭(카르포프 3종 표기, Kasparov 비매칭), 이벤트 필터, 짧은 샘플 PGN에서 추출된 (FEN, 수)가 기대값과 일치, 분할이 판 단위임
- 세 기사 실제 PGN 전처리 실행 후 `stats.json`으로 판 수·국면 수 보고
- 모든 레코드의 `move`가 해당 `fen`에서 합법수인지 전수 확인
- 제외 후 세 기사의 판 수가 비슷한 규모인지 확인하고, 필터에 걸린 이벤트명 목록을 보고서에 첨부해 오탐 여부를 작업지시자가 확인할 수 있게 한다

---

## 2단계: 베이스 모델 PyTorch 이식 및 가중치 로드·검증

### 목표

searchless_chess action-value Transformer를 PyTorch로 재구현하고, 공개 체크포인트를 변환해 실었을 때 원본 JAX 출력과 수치적으로 일치함을 확인한다.

### 구현

**`src/model/tokenizer.py`** — 원본 `tokenizer.py` 이식. FEN → 77토큰 (수순 1 + 판 64 + 캐슬링 4 + 앙파상 2 + 반수 3 + 전수 3), 문자 어휘 31종.

**`src/model/actions.py`** — 원본 `utils._compute_all_possible_actions` 이식. UCI 문자열 ↔ 행동 번호(1,968개). 번호가 원본과 한 칸이라도 어긋나면 임베딩이 뒤섞이므로 생성 순서를 그대로 따른다.

**`src/model/transformer.py`** — 모델 본체:

```python
@dataclass
class TransformerConfig:
    vocab_size: int = 1968
    output_size: int = 128          # 승률 구간 수
    embedding_dim: int = 1024
    num_layers: int = 16            # 136M은 8
    num_heads: int = 8
    max_sequence_length: int = 79   # FEN 77 + 행동 1 + 리턴 1
    widening_factor: int = 4

class ActionValueTransformer(nn.Module):
    def forward(self, tokens: LongTensor[N, 79]) -> Tensor[N, 128]:  # 마지막 위치의 구간 로그확률
    def win_prob(self, tokens) -> Tensor[N]:                         # 구간 중심값과의 기대값
```

원본 구조를 그대로 따른다: 임베딩에 √d 스케일, 학습형 위치 인코딩, 인과 마스크 없음, 블록마다 pre-LN, 어텐션 q/k/v는 bias 없음, MLP는 SwiGLU, 최종 LN 후 선형 출력과 log-softmax. 입력 시퀀스의 shift 처리, 출력 투영의 bias 유무처럼 요약만으로 확정할 수 없는 세부는 구현 시 원본 소스를 줄 단위로 대조해 맞춘다.

**`scripts/export_haiku_params.py`** — GPU 서버의 JAX 환경에서 실행. 원본 체크포인트는 orbax(OCDBT) 형식이라 numpy만으로는 읽을 수 없으므로, orbax로 읽어 평평한 npz(키: `모듈경로/파라미터명`)로 내보낸다.

```
python scripts/export_haiku_params.py --checkpoint base_model/270M --out base_model/270M_params.npz
```

**`src/model/convert.py`** — npz → PyTorch 체크포인트 (numpy와 torch만 사용):

```
python -m src.model.convert --params base_model/270M_params.npz --size 270M --out base_model/270M.pt
```

선형층 가중치는 전치한다. 매핑되지 않은 키, 남는 키, 형상이 다른 텐서가 하나라도 있으면 실패한다. (2단계에서 9M 체크포인트의 실제 구조를 확인해 확정했다.)

**`scripts/dump_jax_reference.py`** — GPU 서버에서 원본 searchless_chess 저장소 안에서 실행. 고정된 국면 목록(`scripts/reference_fens.txt`, 개시·중반·종반·프로모션·캐슬링·앙파상 포함 53개)의 모든 합법수에 대해 원본 토크나이저로 만든 입력 토큰과 구간 로그확률을 `reference_{size}.npz`로 저장한다.

**`scripts/verify_port.py`** — 같은 입력을 PyTorch 모델에 넣어 기준 출력과 비교한다. 입력 토큰·합법수 순서 일치 여부, 구간 로그확률·확률·승률의 최대 절대 오차, 최선수 일치율을 출력한다.

개발 중에는 원본이 함께 공개한 소형 9M 체크포인트로 변환 코드를 먼저 맞춘 뒤 136M, 270M에 적용한다.

### 검증

- 단위 테스트: 토큰화 결과가 원본 규칙의 손계산 예시와 일치, 행동 공간 크기 1,968 및 대표 수(`e2e4`, 프로모션, 캐슬링) 왕복 변환, 무작위 가중치 소형 모델의 입출력 형상
- 변환 시 키 누락·잉여 0건, 파라미터 수가 136M/270M 규모와 일치
- **완료 조건**: GPU 서버에서 `verify_port.py` 실행 결과 구간 로그확률 최대 절대 오차가 float32 수준(1e-4 이하 목표)이고 최선수가 전 국면에서 일치. 결과는 `mydocs/test_result/task_m100_1_test.md`에 기록

(2026-10-07 갱신: 작업지시자 지시로 로컬 컨테이너에서 원본을 직접 실행해 9M·136M·270M 비교를 마쳤다.) 원래 계획은 다음과 같았다. 이 단계의 완료 조건은 GPU 서버 실행이 필요하므로, 작업지시자가 `export_haiku_params.py`, `dump_jax_reference.py`, `convert.py`, `verify_port.py`를 서버에서 실행하고 출력을 `test_result/`에 저장해야 한다.

---

## 3단계: LoRA 어댑터 및 손실함수

### 목표

동결 베이스에 LoRA를 주입하고, 합법수별 q로부터 정책을 만들어 모방 항·기력 항(및 스타일 항 인터페이스)을 계산한다.

### 구현

**`src/model/lora.py`**:

```python
class LoRALinear(nn.Module):        # W x + (alpha / r) * B A x, B는 0으로 초기화
def inject_lora(model, rank=16, alpha=32, dropout=0.0,
                targets=("q", "k", "v", "o")) -> list[nn.Parameter]
def lora_state_dict(model) -> dict   # 어댑터 파라미터만
def load_lora(model, state_dict)
```

- 주입 전에 모델 전체를 `requires_grad=False`로 동결하고, LoRA의 A/B만 학습 대상으로 둔다.
- 기본 대상은 어텐션의 q, k, v, 출력 투영이다. `targets`에 `"mlp"`를 넣으면 MLP 선형층에도 주입한다.
- rank는 제안서 5.3의 16~32 범위에서 기본 16으로 둔다.
- B를 0으로 초기화하므로 주입 직후 출력은 베이스와 완전히 같다.

**`src/train/policy.py`**:

```python
def legal_action_tokens(fens) -> (tokens[M, 79], owner[M], action_ids[M])
def policy_from_q(q[M], owner[M], temperature) -> log_pi[B, A_max], mask[B, A_max]
```

한 배치의 모든 합법수를 한 줄로 펴서 순전파 1회로 처리한 뒤 국면별로 다시 묶는다. 정책은 π(a|s) = softmax(q(s,a)/T).

**`src/train/losses.py`**:

```python
def imitation_loss(log_pi, target_idx, base_log_pi=None, gamma=0.0)
def strength_loss(log_pi, q_base, mask, tau)
def style_loss(log_pi, features, w_p, mask)        # features가 None이면 0
def total_loss(...) = L_imit + alpha * L_style + lam * L_strength
```

- **모방 항**: 기사가 둔 수에 대한 교차엔트로피. `gamma > 0`이면 표본 가중치 (1 − π_base(a*|s))^γ를 곱해, 베이스도 당연히 두는 수는 약하게, 베이스가 예상하지 못한 "그 기사 특유의 수"는 강하게 학습한다. 이는 제안서 4.3의 "베이스 대비 잔차"를 구현 가능한 형태로 옮긴 구현자의 해석이며, 기본값 γ=0(일반 교차엔트로피)으로 두고 옵션으로 제공한다.
- **기력 항**: regret(s,a) = max q_base − q_base(s,a), 손실은 Σ π(a|s)·max(0, regret − τ). 제안서 4.3 수식 그대로다.
- **스타일 항**: −Σ π(a|s)·(w_p·f̃(s,a)). 특성 텐서가 주어지지 않으면 0을 반환한다. 이번 타스크에서는 α=0 기본값으로 비활성이다.

### 검증

- 단위 테스트 (소형 무작위 모델, CPU):
  - LoRA 주입 직후 출력이 베이스와 동일
  - 학습 가능한 파라미터가 LoRA뿐이고, 역전파 한 스텝 후 베이스 가중치가 불변
  - 어댑터 저장 → 새 모델에 로드 → 출력 동일
  - 정책이 국면별로 합 1, 패딩 위치 확률 0
  - 기력 항: 모든 수가 밴드 안이면 0, 밴드 밖 수에 확률이 실리면 양수
  - 스타일 항: 특성 없으면 0, 손계산 예시와 일치
- 소형 모델에서 같은 배치를 반복 학습했을 때 모방 손실이 감소(과적합 가능)함을 확인

---

## 4단계: 학습 스크립트 및 GPU 서버 실행 가이드

### 목표

전처리 데이터와 변환된 베이스로 기사별 LoRA 어댑터를 학습하는 스크립트를 완성하고, GPU 서버에서의 실행 절차를 문서화한다.

### 구현

**`scripts/cache_base_q.py`** — 동결 베이스의 q 사전 계산:

```
python scripts/cache_base_q.py --player tal --base base_model/270M.pt
```

각 국면의 합법수 목록과 베이스 q(s,a)를 `data/processed/{player}/base_q_{size}.npz`에 저장한다. 기울기를 보관하지 않는 추론이라 메모리 부담이 작으므로 `--batch-moves`(기본 1024)로 한 번에 넣는 합법수 수만 조절한다. 베이스는 학습 중 변하지 않으므로 한 번만 계산하면 되고, 학습 시 베이스 순전파가 사라져 비용이 절반으로 준다. 기력 항과 모방 항 가중치가 이 캐시를 쓴다.

**`src/data/dataset.py`** — JSONL과 q 캐시를 읽는 `PositionDataset`, 가변 개수의 합법수를 한 배치로 묶는 `collate_fn`, 한 배치를 합법수 한도에 맞춰 나누는 `split_by_moves`.

```python
def split_by_moves(batch, max_moves) -> list[Batch]
```

- 국면 순서대로 채우다가 합법수 합계가 `max_moves`를 넘기 직전에 끊는다.
- 한 국면의 합법수는 쪼개지 않는다 (정책이 그 국면의 모든 합법수에 대한 softmax이기 때문이다). 합법수가 한도보다 많은 국면은 혼자 한 묶음이 된다.
- 묶음마다 `owner`를 0부터 다시 매긴다 (`pad_by_owner`의 전제).

**`src/train/train.py`** — 학습 CLI:

```
python -m src.train.train --player tal --base base_model/270M.pt \
    --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 \
    --batch-positions 32 --max-moves-per-forward 128 \
    --temperature 0.02 --lam 1.0 --tau 0.02 --alpha 0.0 --gamma 0.0 --out runs/tal
```

- **기울기 누적** (2026-10-07 작업지시자 결정): 갱신 단위는 국면 `--batch-positions`개(기본 32)로 유지하되, 한 번에 순전파·역전파하는 양은 합법수 `--max-moves-per-forward`개(기본 128) 이하의 묶음으로 나눈다. 국면 32개(합법수 610~1,325개)를 한 번에 넣으면 136M·270M에서 GPU 메모리가 부족하기 때문이다 (3단계 보고서 측정).
  - 묶음마다 국면별 손실의 합을 **배치 전체 국면 수**로 나눠 역전파한다. 손실이 국면 평균이므로 누적된 기울기는 한 번에 계산한 것과 같다.
  - gradient clipping, optimizer 갱신, 학습률 스케줄, 로그의 스텝은 묶음이 아니라 갱신 단위로 센다. AMP도 묶음마다 역전파하고 갱신은 한 번만 한다.
  - epoch 끝의 덜 찬 배치는 실제 국면 수로 나눈다.
  - 3단계의 손실함수는 그대로 쓴다. 묶음별 평균에 `묶음 국면 수 / 배치 국면 수`를 곱한다.
  - 한도 기본값 128의 근거: 세 기사 데이터 282,910국면의 합법수는 평균 약 32개, 최대 68개라 한 국면이 한도를 넘는 일은 없다. 로컬 12GB GPU 실측으로 합법수 96~133개 묶음이 136M 4.7 GiB, 270M 9.2 GiB였다. 270M을 12GB에서 돌릴 때는 여유가 적으므로 96으로 낮추고, 메모리가 큰 GPU 서버에서는 올려서 묶음 수를 줄인다.
  - 한도를 바꿔도 갱신 결과는 같고 속도만 달라진다.
- AdamW, 워밍업 후 코사인 감쇠, 혼합 정밀도(AMP), gradient clipping, 고정 seed.
- 주기적으로 val에서 평가: 모방 손실, 기사 수 top-1/top-3 일치율, 평균 regret, 밴드(τ) 이탈률. 같은 지표를 베이스에 대해서도 학습 전에 한 번 측정해 기준선으로 로그에 남긴다.
- 저장물: 어댑터 가중치(`adapter.pt`), 설정(`config.json`), 학습 로그(`log.jsonl`). 베이스 가중치는 저장하지 않는다.
- 하이퍼파라미터 기본값(T, τ, λ, lr)은 출발점이며 첫 학습 결과를 보고 조정한다. 세 기사에는 반드시 같은 값을 쓴다.

**`mydocs/manual/training_guide.md`** — GPU 서버 실행 가이드: 환경 설치, 체크포인트 다운로드·변환, 이식 검증, 전처리, q 캐시, 학습 명령, 결과를 `mydocs/test_result/`에 저장하는 방법.

### 검증

- 단위 테스트: collate가 가변 합법수를 올바르게 묶음, q 캐시의 수 순서가 `policy.py`의 순서와 일치
- 단위 테스트 (묶음 분할): 묶음의 합법수 합계가 한도 이하, 국면이 쪼개지지 않고 순서·총 개수 보존, 한도보다 큰 국면은 단독 묶음
- 단위 테스트 (기울기 누적): 소형 모델에서 한 배치를 한 번에 역전파한 기울기와 여러 묶음으로 누적한 기울기가 float32 오차 범위에서 일치, 한도를 바꿔도 한 스텝 뒤 어댑터 가중치가 일치
- 로컬 스모크 테스트: 소형 무작위 모델과 수백 국면으로 전체 파이프라인(전처리 → 캐시 → 학습 2~3스텝 → 어댑터 저장·재로드)이 CPU에서 오류 없이 완주
- 로컬 GPU(12GB)에서 136M으로 짧은 학습을 돌려 손실 감소, val 지표 출력, 최대 GPU 메모리와 갱신당 시간을 확인
- GPU 서버에서 본 학습 실행 (작업지시자 실행, 결과는 `test_result/`에 저장)

---

## 검증 (전체)

- `pytest tests/` 통과
- 세 기사 전처리 완료 및 `stats.json` 보고
- GPU 서버에서 이식 검증(`verify_port.py`) 통과
- 로컬 스모크 테스트 완주
- 세 기사에 대해 `--player`만 바꿔 동일 명령으로 학습 가능

## 예상 단계: 5단계 (0~4단계)
