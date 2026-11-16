# Task #13: 대국 실행기 (Stockfish, 오프닝 세트, C1·C2·C4) — 구현 계획서

> **상태 (2026-11-16)**: 승인 (해석 10건, 기존 파일 변경 3건 포함)
> 수행 계획서: `mydocs/plans/task_m100_13.md` (승인. 결정 6건 권장안)

## 공통

- **`src/`에는 새 파일만 더한다.** 새 패키지 `src/match/`(대국·Stockfish, torch 사용)와 `src/eval/games.py`(집계·판정, numpy만)다. 기존 함수(`legal_action_tokens`, `policy_from_q`, `ordered_legal_moves`, `load_base`, `inject_lora`, `load_lora`, `src/eval/stats.py` 전체)는 불러 쓴다
- **기존 파일을 고치는 곳은 3개다** (승인 요청). Stockfish 설치 때문이다
  - `gpu_train/setup_env.sh`: conda-forge `stockfish` 설치 한 줄과 버전 확인
  - `docker/Dockerfile`: 로컬 개발 컨테이너에 같은 버전 Stockfish 설치
  - `gpu_train/README.md`: 대국·C4 절 추가
- 결과물은 git 제외 경로 `runs/gpu/match/`(기보), `runs/gpu/quality/`(C4 캐시), `runs/gpu/games/`(판정)에 둔다. 오프닝 200개 파일만 저장소에 커밋한다
- 로컬 테스트는 Task #12와 같이 개발 컨테이너(`chess-transformer-dev`)에서 GPU 없이 돌린다. 모델은 `tests/test_train.py`의 `workspace` fixture(소형 무작위 베이스)를 쓴다. Stockfish가 필요한 테스트는 실행 파일이 없으면 건너뛴다
- 단계마다 `mydocs/working/task_m100_13_stage{N}.md`를 쓰고, 승인 뒤 그 단계 소스와 함께 `local/task13`에 커밋한다

## 모델과 표기

| 모델 ID | 정체 | 쓰는 곳 |
|---|---|---|
| `base` | 베이스 136M (LoRA 가중치 0) | C1 상대, C2·C4 비교 대상, C2 노드 수 보정 |
| `R1`, `R8`, `R2` | 탈·카르포프·스미슬로프 α = 0 (`runs/gpu/`) | α = 0 기준선 (결정 5) |
| `sf` | Stockfish, 노드 수 고정 | C2 상대 |

## 해석 (승인 요청)

기준 문서가 정하지 않은 부분을 결과를 보기 전에 다음과 같이 정한다.

| # | 항목 | 해석 |
|---|---|---|
| 1 | 400 ply 한도 | 오프닝 8 ply를 포함한 전체 ply 수로 센다 |
| 2 | 대국 seed | 오프닝 번호 i, 모델 색 c(백 0, 흑 1)에서 seed = 2i + c. 모든 조합이 같은 seed를 쓴다. C1에서는 한 대국의 두 모델이 같은 생성기를 차례로 쓴다 |
| 3 | Stockfish 결정성 | 매 수 전에 해시를 비운다(`ucinewgame`). 한 수의 결과가 (수순, 노드 수)만으로 정해져, 대국을 몇 개 동시에 돌리든 같은 기보가 나온다. 수순 전체를 넘겨 Stockfish도 반복을 안다 |
| 4 | 점수율과 Elo | 짝 점수(0~2, 모델 쪽)를 짝 단위로 복원 추출한다. s = Σ짝 점수 / (2 × 짝 수). s가 0이나 1이면 Elo가 무한이 되므로 s를 [0.001, 0.999]로 자른다(±1,200 Elo). 원표본에서 잘린 경우는 보고한다 |
| 5 | C2 차이의 쌍대 부트스트랩 | 오프닝 번호를 한 번 뽑아 어댑터 짝과 베이스 짝에 같이 쓴다. 반복마다 Elo(어댑터) − Elo(베이스) |
| 6 | 노드 수 보정 | 예비: 후보 ~~1천·3천·1만·3만·10만~~ **100·300·1천·3천·1만** 노드 (2026-11-16 수정, 2단계 시험에서 1만 노드 대 베이스 0/10. 작업지시자 승인), 오프닝 0~9(20국)씩. 본 보정: 예비 점수율이 0.5를 사이에 두는 후보와 그 근처로 3개를 골라 오프닝 0~49(100국)씩. 0.35~0.65에 드는 것 중 점수율이 0.5에 가장 가까운 노드 수를 고른다(같으면 작은 쪽). 하나도 없으면 이웃 두 후보의 기하 평균으로 100국을 한 번 더 둔다 |
| 7 | 오프닝 선별 | `8moves_v3`의 각 줄을 8 ply까지 두어 FEN으로 중복을 없앤다. C4와 같은 깊이로 평가해 백 기준 \|cp\| ≤ 50(메이트 점수 제외)만 남기고, `default_rng(0)`으로 200개를 뽑는다. 번호는 뽑은 순서다 |
| 8 | C4 수 평가 | 국면마다 최선수 탐색 1회(깊이 고정, MultiPV 1). 모델 수가 최선수와 같으면 ℓ = 0. 다르면 그 수만 허용(`searchmoves`)해 같은 깊이로 1회. W는 둘 차례 쪽 기준이고, 메이트 점수는 이기는 쪽이면 1, 지는 쪽이면 0이다. 탐색 불일치로 ℓ < 0이면 0으로 자르고 그 수를 보고한다. 탐색마다 해시를 비운다 |
| 9 | C4 판정 단위 | S_val 전체, 대국 단위 쌍대 부트스트랩을 기사별 층화로 한다(Task #12 해석 1과 같다). 기사별 값과 ply ≥ 20 값은 보고한다 |
| 10 | 배치와 재현성 | 여러 대국을 동시에 진행하고 같은 모델 차례의 국면을 묶어 계산한다. 같은 장비·같은 묶음 크기면 같은 기보가 나와야 한다. 묶음 크기와 동시 대국 수를 기보 헤더에 기록한다. 묶음 크기에 따라 부동소수점 차이로 드물게 수가 갈릴 수 있음은 테스트로 확인해 보고한다 |

---

## 1단계: Stockfish 연결, 승률 변환, 오프닝 세트

### 목표

Stockfish 버전을 정하고 연결 모듈을 만든다. 오프닝 200개를 뽑아 커밋한다. C4 깊이 제안(18)의 탐색 시간을 잰다.

### 구현

**`src/match/__init__.py`** — 빈 패키지

**`src/match/stockfish.py`** — UCI 연결 (`chess.engine` 사용)

```python
WIN_K = 0.00368208
def win_prob(score: chess.engine.PovScore, pov: chess.Color) -> float   # 해석 8
def find_binary(path: str | None = None) -> Path         # 인자 → $STOCKFISH → PATH
class Stockfish:                                          # Threads=1, Hash 고정
    def __init__(self, binary: Path, hash_mb: int = 16)
    version: str                                          # "Stockfish 17.1" 등 (id name)
    def move(self, board: chess.Board, nodes: int) -> chess.Move      # 해석 3
    def evaluate(self, board, depth: int, only: chess.Move | None = None) -> tuple[chess.Move, PovScore]
    def close(self)
```

**`src/match/openings.py`** — 오프닝 선별

```python
def read_book(path: Path) -> list[tuple[str, list[str]]]   # (FEN, UCI 8개)
def select(lines, engine, depth: int, n: int = 200, limit_cp: int = 50, seed: int = 0) -> list[dict]
# python -m src.match.openings --book 8moves_v3.pgn --depth 18 --out src/match/openings_200.json
```

**`src/match/openings_200.json`** (커밋) — 번호, FEN, 수순(UCI), 평가 cp, 출처·버전·깊이·seed 메타데이터

- `official-stockfish/books` 저장소의 라이선스를 확인해 단계 보고서에 기록한다. 커밋할 수 없는 조건이면 받는 스크립트만 두고 파일은 git에서 뺀다
- Stockfish 버전: 단계 시점 최신 공식 릴리스, conda-forge에 같은 버전이 있는지 확인한다
- 깊이 18 탐색 시간을 로컬 S_val 국면 표본(약 200개)으로 재어, 서버 C4 소요를 추정한다

**`docker/Dockerfile`** — Stockfish 설치 (기존 파일 변경)

**`tests/test_match_stockfish.py`**

### 검증

- `win_prob`: cp 0 → 0.5, 대칭성, 메이트 → 0/1 (손 계산)
- `read_book`: 작은 PGN 예제로 FEN·수순, 중복 제거
- `select`: 가짜 평가 함수로 ±50 필터, seed 고정 시 같은 결과
- (Stockfish 있을 때) 같은 국면 `move` 두 번 → 같은 수, `evaluate(only=…)`가 그 수를 돌려준다
- 오프닝 200개 모두 합법 수순, 8 ply, 백 차례, FEN 중복 없음

---

## 2단계: 대국 실행기

### 목표

모델 대 모델(C1), 모델 대 Stockfish(C2) 대국을 동시에 여러 판 돌려 PGN으로 남긴다. 끊겨도 이어서 돌린다.

### 구현

**`src/match/policy.py`** — 묶음 정책

```python
class PolicyBank:          # 136M 베이스 한 번 로드, LoRA 한 번 주입, 모델별 LoRA 가중치를 바꿔 끼운다
    def __init__(self, base: Path, adapters: dict[str, Path], device)   # "base"는 LoRA 0
    def sample(self, model_id: str, boards: list[chess.Board], generators: list[torch.Generator]) -> list[tuple[chess.Move, float]]
    sha256: dict[str, str]
```

- π 계산은 `legal_action_tokens` → `win_prob` → `policy_from_q`(T = 0.02, float32)로 대국 앱·학습 평가와 같다
- 어댑터 설정이 서로 다르면 거부한다 (`Engine`과 같은 검사)

**`src/match/game.py`** — 대국 하나의 상태와 종료 판정

- 종료 판정은 `src/play/game.py`의 `outcome`과 같은 규칙에 400 ply 한도(해석 1)를 더한다. 기존 파일을 고치지 않으려 새로 쓴다

**`src/match/runner.py`** — 동시 대국과 기록

```python
def schedule(openings: list[int], white_black: ...) -> list[GameSpec]  # 짝: 오프닝마다 색을 바꾼 2국
def run(specs: list[GameSpec], bank: PolicyBank, stockfish_pool, out: Path, concurrent: int = 64) -> None
# python -m src.match.play --model R1 --opponent base --openings 0-199 --out runs/gpu/match/C1_R1
# python -m src.match.play --model base --opponent sf --nodes 10000 --openings 0-199 --out runs/gpu/match/C2_base_10000
```

- 한 바퀴마다 모델 차례 국면을 모델별로 묶어 계산하고, Stockfish 차례는 프로세스 풀(CPU 수만큼)에 나눈다
- 끝난 대국마다 `games.pgn`에 한 판을 덧붙이고 `index.jsonl`에 (오프닝, 색, 결과, 종료 사유, ply 수)를 적는다. 다시 실행하면 `index.jsonl`에 있는 대국을 건너뛴다
- PGN 헤더: `White`·`Black`(모델 ID), `Opening`(번호), `Seed`, `ModelSha256`, `Engine`(Stockfish 버전), `Nodes`, `Batch`, `Concurrent`, `Termination`, `PlyCount`
- 실행 기록 `run.json`: 장비, torch·Stockfish 버전, git HEAD, 시작·끝 시각, 수당 평균 시간

**`tests/test_match_runner.py`**

### 검증

- 같은 seed로 두 번 → 같은 기보. 동시 대국 수 1과 8 → 같은 기보 (CPU, 해석 10)
- `PolicyBank`의 π가 `src/play/engine.py`의 `Engine`과 같다 (소형 모델, 같은 국면)
- 종료 사유 6가지(체크메이트, 스테일메이트, 3회 반복, 50수, 기물 부족, 400 ply)를 만든 국면으로 확인
- 중간에 멈춘 뒤 이어서 실행 → 대국 중복·누락 없음
- 상대가 Stockfish 대신 가짜(첫 합법수)일 때 짝 구성·색 교대·헤더 확인
- (Stockfish 있을 때) 베이스 136M 대 Stockfish 10국 로컬 시험, 수당 시간 측정

---

## 3단계: C4 수 품질 도구, 집계·판정

### 목표

S_val의 C4 손실을 캐시와 함께 계산하는 도구와, 기보·C4 캐시로 C1·C2·C4·C7(반복 비율)을 판정하는 집계 코드를 만든다.

### 구현

**`src/match/quality.py`** — C4 탐색 (서버, CPU)

```python
# python -m src.match.quality --dump runs/gpu/dump --models base R1 R8 R2 --depth 18 --workers 16 --out runs/gpu/quality
```

- `positions.npz`와 각 모델 덤프의 1순위 수(합법수 순서 = `ordered_legal_moves`)를 읽는다. FEN은 val 기록(`data/processed/`)에서 읽고, 국면 수·합법수 수가 덤프와 같은지 확인한다
- 캐시 `best.jsonl`(국면 → 최선수, 점수), `moves.jsonl`(국면, 수 → 점수). 이미 있는 항목은 건너뛴다 (해석 8)
- 결과 `{모델}.npz`: 국면별 ℓ, 잘린 음수 개수

**`src/eval/games.py`** — 집계·판정 (numpy만)

```python
def read_index(dir: Path) -> pairs                       # 오프닝별 짝 점수
def elo(s) -> float                                      # 해석 4
def c1(pairs, n_boot) -> dict                            # 구간 하한 > −50 통과, 상한 < −50 실패
def c2(pairs_model, pairs_base, n_boot) -> dict          # 해석 5
def c4(loss_model, loss_base, positions, n_boot) -> dict # 해석 9, 판정 0.02, 큰 실수(ℓ ≥ 0.20) 보고
def c7_repetition(index) -> float
def calibrate(dirs) -> dict                              # 해석 6, 노드 수 선택
# python -m src.eval.games judge --match runs/gpu/match --quality runs/gpu/quality --out runs/gpu/games
# python -m src.eval.games calibrate runs/gpu/match/C2_base_*
```

- 판정은 `stats.verdict`를 그대로 쓴다. 결과는 `results.json`, `report.md`

**`tests/test_match_quality.py`**, **`tests/test_eval_games.py`**

### 검증

- Elo: s = 0.5 → 0, s = 0.429 → 약 −50, 자르기 (손 계산)
- 쌍대 부트스트랩: 어댑터와 베이스 짝이 같으면 차이 구간이 [0, 0]
- C4: 손으로 만든 손실 배열로 평균 차이, 층화 구간, 판정
- 보정: 예비·본 보정 가짜 결과로 해석 6의 선택 규칙 (범위 안 여러 개, 하나도 없음)
- 캐시: 같은 국면·수를 다시 묻지 않는다 (가짜 엔진의 호출 횟수)

---

## 4단계: 서버 실행 틀과 배포 → 노드 수 보정 (작업지시자)

### 목표

서버에 Stockfish를 설치하고 대국·C4 작업을 제출하는 틀을 만든다. `devel`에 반영해 작업지시자가 노드 수 보정을 돌린다.

### 구현

- **`gpu_train/setup_env.sh`** (기존 파일 변경): conda-forge `stockfish=버전` 설치, `stockfish` 버전 확인
- **`gpu_train/match.sbatch`**, **`submit_match.sh`**: 대국 작업 (`--model`, `--opponent`, `--nodes`, `--openings`). 기존 `submit_dump.sh`와 같은 방식으로 `site.env` 옵션을 붙인다. 끊기면 같은 명령으로 다시 제출하면 이어 간다
- **`gpu_train/quality.sbatch`**, **`submit_quality.sh`**: C4 작업 (CPU만, `--cpus-per-task`)
- **`gpu_train/README.md`** (기존 파일 변경): 대국·C4 절 (설치, 보정 순서, 소요 시간, 회수)
- `local/task13` → `local/devel` → `devel` 반영과 push (Task #10·#11·#12와 같이 서버 `git pull`을 위한 선반영, 작업지시자 승인 후)

### 서버 실행 (작업지시자)

1. `bash gpu_train/setup_env.sh`로 Stockfish 설치 확인
2. 예비 보정: `base` 대 `sf` 후보 5개 × 20국
3. (회수 뒤 내가 후보 3개를 정함) 본 보정: 3개 × 100국

### 검증

- `SBATCH_DRY_RUN=1`로 제출 명령 확인 (로컬)
- 서버 첫 작업 로그에서 Stockfish 버전·실행 파일 SHA-256이 로컬과 같은지 확인

---

## 5단계: 보정 결과와 기준 문서 정정 → α = 0 대국·C4 (작업지시자)

### 목표

노드 수를 정하고, 기준 문서에 고정할 값 네 가지를 정정으로 기록한다. 이후 α = 0 대국과 C4를 서버에서 돌린다.

### 구현

- 보정 기보 회수, `python -m src.eval.games calibrate`로 노드 수 선택 (해석 6)
- **`mydocs/tech/evaluation_criteria.md`** 정정 (2.7절 절차, 0절 정정 이력): 5.3절 오프닝 세트 출처, 5.4절 Stockfish 버전·C2 노드 수, C4 깊이·승률 변환식. 5.6·8.4절 소요 시간을 서버 실측으로 고친다. 판정 조건 값은 바꾸지 않는다
- 서버 실행 안내

### 서버 실행 (작업지시자)

| 작업 | 대국·탐색 |
|---|---|
| C1: `R1`·`R8`·`R2` 대 `base` | 3 × 400국 |
| C2: `base`·`R1`·`R8`·`R2` 대 `sf` (정한 노드 수) | 4 × 400국 |
| C4: `base`·`R1`·`R8`·`R2` | S_val 13,846국면, 캐시 공유 |

### 검증

- 정정 기록이 α > 0 결과 전이다 (α > 0 학습 미착수)
- 보정 선택값을 독립 계산으로 다시 확인

---

## 6단계: 결과 회수, 판정, 결과 정리

### 목표

α = 0 기준선의 C1·C2·C4 값·구간·판정을 내고 test 문서로 정리한다.

### 구현

- 기보·C4 캐시 회수, 무결성(대국 수 400, 오프닝·색 누락 없음, 모델 SHA-256 = Task #11 기록) 확인
- `python -m src.eval.games judge`로 판정
- **`mydocs/test_result/task_m100_13_test.md`**: 실행 기록(장비, 버전, 시간), 노드 수 보정 표, C1·C2·C4 값·구간·판정, 종료 사유 분포, C7 반복 비율, 큰 실수 비율, C3(Task #12)와 C4 비교

### 검증

- 원표본 점수율·평균 손실을 `src.eval`을 쓰지 않는 독립 스크립트로 다시 계산해 `results.json`과 대조
- 문서 표를 `results.json`에서 다시 만들어 대조
- 무작위로 고른 기보 몇 판을 같은 seed로 로컬에서 다시 두어 같은지 확인 (장비가 달라 다르면 차이를 보고)

---

## 검증 (전체)

- `pytest` 전체 통과 (개발 컨테이너, CPU)
- 기존 테스트 회귀 없음 (`src/` 기존 파일 변경 없음)
- 기준 문서 정정 4건이 α > 0 결과 전에 기록됨
- α = 0 어댑터 3개의 C1·C2·C4 판정 (C3는 Task #12)

## 위험 (단계별)

| 단계 | 위험 | 대응 |
|---|---|---|
| 1 | 오프닝 세트 라이선스가 커밋을 허용하지 않는다 | 받는 스크립트와 SHA-256만 커밋 |
| 1 | conda-forge에 최신 Stockfish가 없다 | 공식 바이너리를 받는 설치 스크립트로 바꾼다 (수행 계획서 결정 3 대안) |
| 2 | 서버 대국 속도가 느리다 | 2단계 로컬 측정과 4단계 예비 보정으로 다시 추정. 동시 대국 수를 늘리고 C1·C2 작업을 나눠 제출한다 |
| 3 | C4 깊이 18이 서버 CPU로 감당되지 않는다 | 1단계 측정으로 판단해, 5단계 정정 전에 깊이를 결정 요청한다 |
| 5 | 보정에서 범위 안 노드 수가 없다 | 해석 6의 추가 보정 1회. 그래도 없으면 결정 요청 |

## 예상 단계: 6단계
