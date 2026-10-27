# Task #7: 모델과 사용자가 직접 대국하는 체스 앱 개발 (유저 확인용) — 구현 계획서

## 공통

- 학습·모델 코드(`src/model/`, `src/train/`, `src/data/`)는 수정하지 않는다. 새 코드는 `src/play/`에 두고, 기존 함수는 가져다 쓰기만 한다. 수정이 필요해지면 멈추고 따로 승인받는다.
- 실행 환경은 로컬 개발 컨테이너(WSL2), RTX 4070 Ti 12 GB, Python 3.13이다. GPU는 4단계 실기 확인에서만 쓴다. 1~3단계 테스트는 CPU와 소형 무작위 모델로 돌린다.
- 추론은 float32로 한다 (학습 중 평가 설정 `eval_amp: false`와 같음). 앱의 π가 보고서의 평가값과 같은 계산이 되게 하기 위해서다.
- 기존 테스트 116건은 단계마다 계속 통과해야 한다.

**산출물 위치**

```
src/play/__init__.py
src/play/engine.py                      ← 1단계: 모델 불러오기, 어댑터 전환, 정책, 수 선택, 비교
src/play/game.py                        ← 2단계: 대국 상태 (python-chess 감싸기)
src/play/server.py                      ← 2단계: FastAPI 앱, 실행 진입점
src/play/static/index.html              ← 3단계
src/play/static/style.css               ← 3단계
src/play/static/board.js                ← 3단계: 체스판 그리기·입력
src/play/static/app.js                  ← 3단계: 설정 창, API 호출, 패널
src/play/static/piece/cburnett/*.svg    ← 3단계: 기물 12개 + 출처·라이선스 파일
tests/test_play_engine.py               ← 1단계
tests/test_play_server.py               ← 2단계
requirements.txt                        ← 2단계: fastapi, uvicorn, httpx(TestClient용) 추가
mydocs/test_result/task_m100_7_test.md  ← 4단계: 응답 시간, GPU 메모리 실측
mydocs/manual/play_app.md               ← 4단계: 실행 매뉴얼
README.md                               ← 4단계: 대국 앱 절 추가
```

---

## 1단계: 추론 엔진 모듈과 테스트

### 목표

웹과 무관한 순수 Python 엔진을 만든다. 엔진은 모델 id와 FEN을 받아 학습과 같은 정책 π를 계산하고, 그 정책에서 수를 샘플링한다.

### 구현

**`src/play/engine.py`** (신규)

```python
TEMPERATURE = 0.02  # 학습 설정과 같음

@dataclass(frozen=True)
class ModelSpec:
    id: str              # "base-136M", "tal-136M" 등
    label: str           # 화면 표시용 ("탈 (136M)")
    size: str            # "9M" | "136M" | "270M"
    adapter: Path | None # 베이스면 None

@dataclass
class Candidate:
    uci: str
    san: str
    prob: float          # π(a|s)
    q: float             # q(s,a), 둔 쪽 기준 예상 승률

@dataclass
class Analysis:
    model_id: str
    candidates: list[Candidate]  # π 내림차순, 상위 k개
    win_prob: float              # max_a q(s,a)
    legal_count: int

def default_specs(base_dir="base_model", runs_dir="runs") -> list[ModelSpec]: ...
    # 베이스 9M·136M·270M, 어댑터 tal·karpov·smyslov (136M). 파일이 있는 것만 돌려준다

class Engine:
    def __init__(self, specs, device="cuda" or "cpu", loader=load_base): ...
    def available(self) -> list[ModelSpec]
    def analyze(self, model_id: str, fen: str, top_k: int = 5) -> Analysis
    def choose(self, model_id: str, fen: str, generator: torch.Generator) -> tuple[Candidate, Analysis]
    def compare(self, model_ids: list[str], fen: str, top_k: int = 3) -> list[Analysis]
```

**정책 계산**: `src.train.policy.legal_action_tokens([fen])` → `model.win_prob(tokens)` → `policy_from_q(q, owner, TEMPERATURE)`. 학습과 같은 함수를 그대로 부르므로 π의 정의가 학습과 같다. 수 목록은 `encode_position`이 쓰는 `ordered_legal_moves` 순서이고, SAN은 python-chess로 만든다.

**수 선택 (학습 정책)**: `torch.multinomial(pi, 1, generator=generator)`로 π에서 하나를 뽑는다. 생성기는 서버가 하나 들고 있다가 넘긴다. 생성기의 시드를 고정하면 같은 수열이 재현된다. 최선수(argmax) 모드는 넣지 않는다 (작업지시자 결정: 학습 정책대로).

**모델 관리**:
- 베이스는 크기별로 필요할 때 처음 한 번 불러온다. 다른 크기가 필요하면 이전 크기를 GPU에서 내리고 `torch.cuda.empty_cache()`를 부른다. 그래서 GPU에는 한 크기만 올라 있다.
- 136M에 어댑터를 쓸 때는 베이스에 LoRA를 한 번만 주입한다 (`inject_lora`, 어댑터 파일의 settings 사용). 각 어댑터의 LoRA 가중치는 `torch.load`로 CPU에 들고 있다가 `load_lora`로 바꿔 끼운다.
- "베이스 136M"은 LoRA를 주입한 상태에서 `lora_B`를 모두 0으로 둔 가중치를 끼워 표현한다. B = 0이면 출력이 베이스와 같다 (기존 `test_injection_keeps_output`이 확인하는 성질).
- 세 어댑터의 settings(rank, alpha, targets)가 서로 다르면 시작할 때 오류를 낸다 (지금은 세 어댑터 모두 rank 16, alpha 32, qkvo).
- 지금 어떤 어댑터가 끼워져 있는지 기억한다. 같은 어댑터면 다시 끼우지 않는다.
- `torch.inference_mode()`와 `model.eval()`에서 계산한다.

**오류**: 없는 모델 id, 합법수가 없는 국면(종료 국면), 잘못된 FEN이면 `ValueError`를 낸다. 서버는 이를 400으로 바꾼다.

**`tests/test_play_engine.py`** (신규). 소형 모델 `TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)`을 쓴다. 어댑터 파일은 `save_adapter`로 tmp_path에 만들고, `loader`를 주입해 실제 가중치 없이 돈다.

| 테스트 | 확인 내용 |
|---|---|
| 정책 일치 | `analyze`의 π·q가 같은 모델로 `legal_action_tokens` → `policy_from_q`를 직접 계산한 값과 같다 |
| 합법수만 | 여러 국면(시작, 승진 가능, 앙파상, 캐슬링)에서 `choose`가 고른 수가 합법수다 |
| 샘플링 재현 | 같은 시드의 생성기 두 개로 고르면 같은 수열이 나온다 |
| 샘플링 분포 | 고정 국면에서 2,000회 뽑은 빈도가 π와 맞는다 (가장 큰 확률 수의 빈도 오차 0.05 이내) |
| 어댑터 전환 | A → B → A 순으로 전환한 뒤의 출력이 각 어댑터만 단독으로 실은 모델의 출력과 같다. "베이스 136M"으로 돌아오면 LoRA 없는 베이스 출력과 같다 |
| 후보 정렬 | 후보가 π 내림차순이고, 상위 k개이며, 합법수가 k개보다 적으면 그 수만큼만 나온다 |
| 오류 | 없는 모델 id, 체크메이트 국면, 잘못된 FEN에서 `ValueError`가 난다 |
| 비교 | `compare`가 요청한 모델 순서대로 결과를 내고, 각 결과가 `analyze`의 결과와 같다 |

### 검증

- `pytest` 전체 통과 (기존 116건 + 신규)
- 1단계 보고서에 엔진 API와 테스트 결과를 적는다

---

## 2단계: FastAPI 서버(대국 상태, API)와 테스트

### 목표

대국 상태 관리와 JSON API를 만든다. 브라우저 없이 `TestClient`만으로 한 판을 끝까지 둘 수 있게 한다.

### 구현

**`src/play/game.py`** (신규): `chess.Board`를 감싼 대국 상태.

```python
@dataclass
class MoveRecord:
    uci: str
    san: str
    by: str                  # "user" | "model"
    prob: float | None       # 모델이 둔 수의 π (사용자 수는 None)
    candidates: list[Candidate] | None
    repetition: bool         # 이 수로 같은 국면이 3번째가 되었는지

class Game:
    model_id: str
    user_color: chess.Color
    board: chess.Board
    records: list[MoveRecord]
    resigned: chess.Color | None
    def push_user(self, uci: str) -> None      # 차례·합법성 검사, 불법이면 ValueError
    def push_model(self, candidate, analysis) -> None
    def undo(self) -> None                     # 사용자 차례가 될 때까지 되돌린다 (보통 2수)
    def resign(self) -> None
    def outcome(self) -> dict | None           # board.outcome(claim_draw=True) 기준 + 기권
    def state(self) -> dict                    # API 응답용
    def pgn(self) -> str                       # 헤더: Event, Date, White, Black(모델 label), Result
```

- 종료 판정은 `board.outcome(claim_draw=True)`를 쓴다. 3회 반복과 50수 규칙은 자동으로 무승부 처리한다 (lichess 컴퓨터 대국과 같은 방식).
- 승진: 사용자가 보낸 UCI에 승진 기물이 없는데 승진 수만 합법이면 오류로 돌려보낸다. 승진 기물은 화면에서 고른다.

**`src/play/server.py`** (신규): FastAPI 앱. 한 사용자만 쓰므로 진행 중인 대국은 서버 메모리에 하나만 둔다. 모델 계산은 `threading.Lock`으로 한 번에 하나만 돈다.

| 메서드·경로 | 요청 | 응답 |
|---|---|---|
| `GET /` | | `static/index.html` |
| `GET /api/models` | | 모델 목록 (id, label, size), 비교 패널 기본 모델 4개 |
| `POST /api/game` | `{model, color: "white"\|"black"\|"random"}` | 대국 상태. 사용자가 흑이면 백인 모델의 첫 수는 화면이 따로 요청한다 |
| `GET /api/game` | | 대국 상태 (새로고침 복구용) |
| `POST /api/game/move` | `{uci}` | 사용자 수를 반영한 상태 |
| `POST /api/game/engine` | | 모델 수를 반영한 상태 (모델이 둔 수, π, 후보 상위 5개, 예상 승률) |
| `POST /api/game/undo` | | 상태 |
| `POST /api/game/resign` | | 상태 |
| `GET /api/game/pgn` | | PGN 텍스트 (내려받기) |
| `POST /api/analyze` | `{fen, models: [...]}` | 모델별 후보 상위 3개 (비교 패널용) |

대국 상태 JSON:

```json
{
  "fen": "...", "turn": "white", "user_color": "white",
  "model": {"id": "tal-136M", "label": "탈 (136M)"},
  "legal_moves": ["e2e4", "..."],     // 사용자 차례일 때만
  "last_move": "e7e5", "check": "e1" | null,
  "moves": [{"uci", "san", "by", "prob", "repetition"}],
  "last_analysis": {"candidates": [...], "win_prob": 0.53},
  "outcome": null | {"result": "1-0", "reason": "checkmate" | "stalemate" | "insufficient_material" | "threefold_repetition" | "fifty_moves" | "resign", "winner": "white"}
}
```

- 사용자 수와 모델 수를 나눈 이유: 화면이 사용자 수를 곧바로 그리고, 그다음 모델 수를 기다리게 하기 위해서다.
- 오류: 엔진과 대국의 `ValueError`는 400 `{"detail": "..."}`로, 대국이 없으면 404로, 차례가 아니거나 대국이 끝났으면 409로 돌려준다.
- 실행: `python -m src.play.server [--host 127.0.0.1] [--port 8000] [--device cuda|cpu] [--seed N] [--base-dir base_model] [--runs-dir runs]`. 기본 device는 CUDA가 있으면 cuda다. `--seed`를 주지 않으면 시드는 매번 무작위이고, 쓴 시드를 시작할 때 출력한다.
- 앱은 `create_app(engine, seed)` 함수로 만든다. 테스트는 소형 모델 엔진을 주입한다.

**`requirements.txt`**: `fastapi`, `uvicorn`을 추가한다. `TestClient`가 쓰는 `httpx`도 추가한다 (테스트 전용 의존성. 기존 `pytest`처럼 한 파일에 둔다).

**`tests/test_play_server.py`** (신규)

| 테스트 | 확인 내용 |
|---|---|
| 모델 목록 | 주입한 모델이 모두 나온다 |
| 한 판 진행 | 사용자 백으로 수를 두고 모델 수를 받는다. 이를 반복하다 종료되거나 40수에 이르면 멈춘다. 매 상태의 FEN이 기록을 다시 둔 국면과 같다 |
| 사용자 흑 | 새 대국 직후에는 모델 차례다. 이때 사용자 수는 409로 거부되고, 모델 수를 받은 뒤에는 사용자 차례가 된다 |
| 불법수·차례 | 불법수는 400, 모델 차례의 사용자 수는 409, 종료 뒤의 수는 409로 거부된다 |
| 승진 | 승진 기물이 없는 UCI는 400이고, `a7a8q`는 받아들인다 |
| 무르기 | 두 수를 되돌려 다시 사용자 차례가 된다. 첫 수 전에는 아무 일도 하지 않는다 |
| 종료 판정 | 바보의 메이트(체크메이트), 기권, 3회 반복 무승부의 outcome이 맞다 |
| PGN | 내려받은 PGN을 python-chess로 읽으면 수 기록과 결과가 같다 |
| 비교 | `/api/analyze`가 요청한 모델 수만큼 결과를 내고, 종료 국면 FEN은 400이다 |
| 시드 재현 | 같은 시드로 만든 앱 두 개에서 같은 사용자 수를 두면 모델의 응수가 같다 |

### 검증

- `pytest` 전체 통과
- 서버를 소형 모델로 띄워 `curl`로 새 대국과 수 두기를 확인한다

---

## 3단계: 브라우저 화면 (lichess 참고 디자인, 비교 패널 토글)

### 목표

lichess 대국 화면의 배치와 색감을 참고한 화면을 프레임워크 없이 만든다. 사용자는 마우스로 수를 두고, 기보·후보·비교 패널을 본다.

### 구현

**기물**: cburnett 세트의 SVG 12개(`wK.svg` … `bP.svg`)를 `src/play/static/piece/cburnett/`에 넣는다. 원본은 Wikimedia Commons의 Colin M.L. Burnett 기물이다. 같은 폴더에 `README.md`를 두어 출처, 저자, 라이선스(CC BY-SA 3.0)를 적는다. 받아 온 SVG는 스크립트가 없는 순수 도형인지 확인한 뒤 넣는다.

**배치** (`index.html`, `style.css`):

```
┌───────────────────────────────────────────────┐
│ 상단 바: 앱 이름 · [새 대국] · [비교 패널 ON/OFF] │
├────────────┬──────────────────┬───────────────┤
│ 왼쪽 패널   │                  │ 모델 이름·승률  │
│ 대국 정보   │     체스판        │ ─────────────  │
│ (모델, 색,  │   (정사각, 갈색)  │ 기보 2열 표     │
│  결과)     │                  │ ─────────────  │
│ 비교 패널   │                  │ ⟲무르기 ⚑기권   │
│ (ON일 때)   │                  │ ↓PGN  ◀ ▶      │
│            │                  │ ─────────────  │
│            │                  │ 사용자          │
├────────────┴──────────────────┴───────────────┤
│ 후보 수 패널: 모델이 마지막에 고려한 상위 5수     │
│ (SAN, π 막대, q)                               │
└───────────────────────────────────────────────┘
```

- 색: 어두운 배경(#161512 계열)에 패널은 조금 밝게(#262421 계열) 둔다. 판은 갈색 두 톤(#f0d9b5 / #b58863), 강조는 마지막 수 노란 녹색 반투명, 선택 칸 녹색, 합법수는 점(빈칸)과 테두리 원(잡는 수)으로 표시한다. 체크는 왕 칸에 붉은 방사형 그라데이션을 쓴다. 이 값들은 CSS 변수로 모은다.
- 반응형: 폭 1000px 미만에서는 판을 위에, 패널을 아래로 쌓는다. 판은 화면 폭에 맞춘 정사각형이다.
- 글꼴: 시스템 글꼴을 쓴다 (외부 요청 없음).

**체스판** (`board.js`): CSS 그리드 8×8 칸에 기물 `<img>`를 놓는다.
- 입력: 클릭 두 번(기물 → 칸)과 끌어 놓기 둘 다 지원한다. 서버가 준 `legal_moves`에 있는 수만 받는다.
- 승진: 승진 수면 판 위에 퀸·룩·비숍·나이트 선택 창을 띄운다.
- 사용자가 흑이면 판을 뒤집는다. 칸 좌표(a~h, 1~8)를 판 가장자리에 표시한다.
- 되돌려 보기: 기보의 수를 누르거나 ◀ ▶ 키(키보드 ←/→)로 지난 국면을 본다. 지난 국면을 보는 동안에는 입력을 막는다. FEN은 기록을 python-chess가 만든 값으로 받는다. 이를 위해 2단계 상태의 `moves`에 각 수 뒤의 FEN을 넣는다 (화면에서 규칙을 다시 구현하지 않기 위해서다).
- 기물이 움직일 때 짧은 이동 애니메이션(약 150 ms)을 넣는다.

**흐름** (`app.js`):
1. 처음 열면 `GET /api/game`으로 진행 중인 대국을 확인한다. 없으면 대국 설정 창을 띄운다. 설정 창에서는 모델(목록에서 선택)과 색(백·무작위·흑 버튼 세 개, lichess 형식)을 고른다.
2. 사용자가 수를 두면 `POST /api/game/move`를 부른 뒤 판을 그린다. 대국이 끝나지 않았으면 이어서 `POST /api/game/engine`을 부른다. 기다리는 동안 모델 쪽에 "생각 중" 표시를 띄운다.
3. 모델 수를 받으면 판, 기보, 후보 패널, 승률을 갱신한다. 비교 패널이 켜져 있으면 새 국면으로 `POST /api/analyze`를 부른다.
4. 대국이 끝나면 결과와 이유를 판 위 창과 기보 끝에 보여 준다 ("백 승 · 체크메이트" 등).

**기보**: 수 번호와 백·흑 SAN의 2열 표로 보여 준다. 모델이 둔 수에는 π를 작게 붙인다 (예: `Nf3 42%`). π가 10% 미만이면 색을 달리해 "확률 낮은 수"를 알린다. 3회 반복을 만든 수에는 표시를 붙인다.

**비교 패널** (작업지시자 결정: ON/OFF):
- 상단 바의 토글로 켜고 끈다. 상태는 `localStorage`에 저장한다. 저장소를 쓸 수 없는 환경에서는 기본값(OFF)으로 동작한다.
- 켜면 현재 보고 있는 국면(되돌려 보기 중이면 그 국면)에서 네 모델(베이스 136M·탈·카르포프·스미슬로프)의 상위 3수를 열 네 개로 보여 준다. 각 수에는 SAN, π, q를 적고, 실제로 둔 수와 같은 수는 강조한다.
- 끄면 `/api/analyze`를 부르지 않는다.
- 종료 국면에서는 "종료 국면" 문구를 띄운다.

### 검증

- `pytest` 전체 통과 (화면 코드는 자동 테스트하지 않는다)
- 소형 모델 서버로 브라우저에서 확인한다
  - 백·흑 각각으로 대국 시작
  - 클릭·끌어 놓기로 수 두기, 승진 선택 (승진 직전 국면은 FEN으로 시작할 수 없으므로 테스트용 국면 진행으로 확인)
  - 무르기, 되돌려 보기, 기권, PGN 저장
  - 비교 패널 ON/OFF와 새로고침 뒤 유지
  - 폭 1000px 미만 배치
- 화면 캡처를 3단계 보고서에 넣는다

---

## 4단계: 실기 확인, 실행 매뉴얼, README

### 목표

실제 가중치(136M 베이스, 어댑터 세 개, 9M·270M 베이스)로 앱을 돌려 응답 시간과 GPU 메모리를 잰다. 실행 방법을 문서로 남긴다.

### 구현

1. **실측** (GPU): 서버를 띄우고 스크래치패드 스크립트로 API를 호출한다.
   - 모델별 첫 응답 시간(베이스 불러오기 포함)과 이후 모델 수 응답 시간(중앙값, 최대, 국면 20개 이상)
   - 어댑터 전환 시간 (탈 → 카르포프 등)
   - 비교 패널 ON일 때 `/api/analyze`(모델 4개) 응답 시간
   - GPU 사용량: 크기별로 올렸을 때 `torch.cuda.max_memory_allocated`와 `nvidia-smi` 전체 사용량 (MiB 단위로 적고, GiB 환산은 1024로 나눈다)
   - 9M → 136M → 270M로 바꿀 때 이전 크기가 내려가는지
2. **실기 대국**: 작업지시자가 브라우저로 한 판 이상 둔다. 확인한 점과 PGN은 테스트 결과 문서에 남긴다. 이상 동작이 나오면 멈추고 보고한다.
3. **`mydocs/test_result/task_m100_7_test.md`**: 실측 명령, 환경, 원 출력, 표.
4. **`mydocs/manual/play_app.md`** (신규): 설치(`pip install -r requirements.txt`), 필요한 파일(`base_model/*.pt`, `runs/*_136M/adapter.pt`), 실행 명령과 옵션, WSL2에서 Windows 브라우저로 접속하는 방법(localhost가 안 되면 `--host 0.0.0.0`과 WSL IP), 화면 사용법, 수 선택 방식(π에서 T = 0.02로 샘플링), 비교 패널, 알려진 한계(3회 반복, FEN만 봄), 자주 나는 오류.
5. **`README.md`**: "대국 앱" 절(실행 명령 두 줄과 매뉴얼 링크), 저장소 구조에 `src/play/` 추가, 현재 상태 표에 Task #7, 문서 표에 매뉴얼 추가.

### 검증

- `pytest` 전체 통과
- 136M 어댑터 세 개와 베이스 세 크기 모두로 대국이 시작되고 수를 둔다
- 매뉴얼 절차대로 새 셸에서 실행된다

---

## 검증 (전체)

- `pytest` 통과 (기존 116건 + 신규 엔진·서버 테스트)
- `python -m src.play.server` 실행 후 `http://localhost:8000`에서 사용자와 모델의 대국이 끝까지 진행된다
- 앱의 π가 학습 코드의 π와 같다 (1단계 테스트)
- 비교 패널을 켜면 네 모델의 후보가 나오고, 끄면 계산하지 않는다

## 예상 단계: 4단계
