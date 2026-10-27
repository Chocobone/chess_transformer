# Task #7 2단계 완료 보고서: FastAPI 서버(대국 상태, API)와 테스트

## 요약

- 대국 상태 `src/play/game.py`와 FastAPI 서버 `src/play/server.py`를 만들었다. 브라우저 없이 API만으로 한 판을 끝까지 둘 수 있다.
- 서버 테스트 15건을 추가했다 (`tests/test_play_server.py`). 전체 142건(기존 116 + 엔진 11 + 서버 15)이 통과한다.
- 실제 가중치로 서버를 띄워 `curl`로 확인했다 (CPU). 6개 모델을 모두 찾았고, 새 대국 → 사용자 수 → 모델 수 → PGN이 정상 동작한다.
- `requirements.txt`에 `fastapi>=0.162`, `uvicorn>=0.54`, `httpx>=0.28`(테스트 전용)을 추가했다. 설치 버전은 fastapi 0.162.0, uvicorn 0.54.0, httpx 0.28.1이다.
- 계획과 다른 점이 두 가지 있다 (5절). 3회 반복 판정 방식과, `httpx`의 폐기 예정 경고다.

## 1. 대국 상태 (`src/play/game.py`)

| 이름 | 내용 |
|---|---|
| `MoveRecord(uci, san, by, fen, prob, analysis, repetition)` | 수 하나. `fen`은 수를 둔 뒤의 국면(되돌려 보기용), `prob`·`analysis`는 모델 수에만 있다. `repetition`은 이 수로 같은 국면이 세 번째가 되었는지 |
| `Game(model_id, model_label, user_color)` | 시작 국면에서 시작한다 |
| `require_turn(user)` | 대국이 진행 중이고 그쪽 차례인지 확인한다. 아니면 `TurnError` |
| `push_user(uci)` | 표기 오류, 불법수, 승진 기물이 빠진 승진 수는 `ValueError` |
| `push_model(candidate, analysis)` | 모델 수를 기록한다 |
| `undo()` | 사용자가 마지막으로 둔 수까지 되돌린다. 모델 수를 받은 뒤면 2수, 받기 전이면 1수다. 사용자 수가 없으면 그대로 둔다. 기권도 풀린다 |
| `resign()` | 사용자 기권 |
| `outcome()` | 기권, 체크메이트, 스테일메이트, 기물 부족, 75수·5회 반복(python-chess 자동), **3회 반복**, **50수 규칙**. `{"result", "reason", "winner"}` |
| `state()` | API 응답 JSON (아래) |
| `pgn()` | Event, Site, Date, White, Black(모델 이름), Result, Termination 헤더를 붙인 PGN |

대국 상태 JSON (실제 응답에서 발췌):

```json
{"fen": "...", "initial_fen": "...", "turn": "white", "user_color": "white",
 "model": {"id": "tal-136M", "label": "탈 (136M)"},
 "legal_moves": ["g1h3", "..."],
 "last_move": "c7c5", "check": null,
 "moves": [{"uci": "c7c5", "san": "c5", "by": "model", "fen": "...", "prob": 0.7247, "repetition": false}],
 "last_analysis": {"model_id": "tal-136M", "candidates": [{"uci": "c7c5", "san": "c5", "prob": 0.7247, "q": 0.5155}, "..."],
                   "win_prob": 0.5155, "legal_count": 20},
 "outcome": null}
```

`legal_moves`는 사용자 차례이고 대국이 진행 중일 때만 채운다. 화면은 이 목록에 있는 수만 받는다.

## 2. 서버 (`src/play/server.py`)

| 메서드·경로 | 동작 | 오류 |
|---|---|---|
| `GET /` | `static/index.html` | 파일이 없으면 404 (3단계 전) |
| `GET /api/models` | 모델 목록과 비교 패널 기본 모델(있는 것만) | |
| `POST /api/game` | `{model, color}`로 새 대국. `random`은 서버 생성기로 정한다 | 없는 모델·잘못된 색 400 |
| `GET /api/game` | 현재 상태 (새로고침 복구용) | 대국 없음 404 |
| `POST /api/game/move` | `{uci}` 사용자 수 | 불법수 400, 모델 차례·종료 409 |
| `POST /api/game/engine` | 모델 수 (π에서 샘플링, 후보 상위 5개) | 사용자 차례·종료 409 |
| `POST /api/game/undo` | 무르기 | |
| `POST /api/game/resign` | 기권 | 종료 뒤 409 |
| `GET /api/game/pgn` | PGN 내려받기 (`game_20261027_tal-136M.pgn`) | |
| `POST /api/analyze` | `{fen, models?}` 모델별 후보 상위 3개와 이름. `models`가 없으면 비교 패널 기본 모델 | 종료 국면·잘못된 FEN·없는 모델 400 |

- `create_app(engine, seed)`가 앱을 만든다. 진행 중인 대국은 메모리에 하나만 둔다. 엔진과 대국 상태는 `threading.Lock` 하나로 보호한다 (FastAPI는 동기 함수를 스레드 풀에서 돌린다).
- 모델 수 샘플링과 무작위 색은 서버가 가진 `torch.Generator` 하나를 쓴다. 시드가 같고 요청 순서가 같으면 같은 대국이 재현된다.
- 실행 옵션: `--host`(기본 127.0.0.1), `--port`(8000), `--device`(기본: CUDA가 있으면 cuda), `--seed`(기본: 무작위, 시작할 때 출력), `--base-dir`, `--runs-dir`. 모델 파일이 하나도 없으면 안내 문구를 내고 끝난다.

## 3. 테스트 (`tests/test_play_server.py`)

소형 무작위 모델 엔진(베이스 9M·136M, 탈 어댑터)을 `create_app`에 넣고 FastAPI `TestClient`로 돌린다. 모델이 무작위라 원하는 수를 두게 할 수 없는 종료 판정은 `Game`에 정해 둔 수를 넣어 확인한다.

| 테스트 | 확인 내용 | 결과 |
|---|---|---|
| `test_models` | 모델 목록과, 없는 모델이 빠진 비교 목록 | 통과 |
| `test_play_a_game` | 사용자 백으로 무작위 합법수를 두고 모델 수를 받는다. 이를 종료 또는 40수까지 반복한다. 매번 FEN = 기록을 다시 둔 국면 = 마지막 수의 `fen`, `legal_moves` = python-chess 합법수, 0 < π ≤ 1, 후보 5개 이하. 끝에 `GET /api/game` = 마지막 응답 | 통과 |
| `test_user_plays_black` | 시작 직후 모델 차례이고, 사용자 수는 409다. 모델 수를 받은 뒤 사용자 차례가 된다. 모델 수를 다시 요청하면 409다 | 통과 |
| `test_random_color` | 시드 10개에서 백·흑이 모두 나온다 | 통과 |
| `test_rejects_bad_requests` | 대국 없음 404, 없는 모델·잘못된 색·불법수·표기 오류 400, 차례 아님 409 | 통과 |
| `test_undo` | 첫 수 전엔 그대로, 모델 수 뒤엔 2수, 모델 수 전엔 1수를 되돌린다 | 통과 |
| `test_undo_when_user_is_black` | 모델의 첫 수만 있으면 그대로 두고, 그 뒤로는 사용자 차례로 돌아간다 | 통과 |
| `test_resign` | 결과 1-0(사용자 흑 기권), 종료 뒤 모델 수·재기권 409, 무르기로 풀린다 | 통과 |
| `test_pgn` | 내려받은 PGN을 python-chess로 읽으면 수 기록·결과·이름이 같다 | 통과 |
| `test_analyze` | 기본 모델 순서, 지정 모델, 종료 국면 400, 없는 모델 400 | 통과 |
| `test_same_seed_same_replies` | 같은 시드로 만든 앱 두 개에서 모델 응수가 같다 | 통과 |
| `test_checkmate` | 바보의 메이트: 0-1 checkmate, `check` = e1, `legal_moves` 비어 있음, 종료 뒤 수 거부 | 통과 |
| `test_threefold_repetition` | 나이트 왕복 8수로 3회 반복 무승부. 마지막 수에만 `repetition` 표시. 무르면 다시 진행 | 통과 |
| `test_repetition_is_not_claimed_early` | 다음 수로 3회 반복이 될 수 있을 뿐인 국면(7수째)은 끝나지 않는다 | 통과 |
| `test_promotion` | 승진 기물이 없는 `a7a8`은 400이고, `a7a8q`는 `a8=Q+`다 | 통과 |

```
$ python -m pytest -q
142 passed, 57 warnings in 41.82s
```

경고는 기존 56건에 `httpx` 폐기 예정 경고 1건이 더해진 것이다 (5절).

처음 실행에서 `test_checkmate`가 실패했다. 테스트가 이미 끝난 국면에서 불법수의 SAN을 만들려 한 테스트 쪽 결함이었고, 후보를 직접 만들도록 고쳤다.

## 4. 실제 가중치 확인 (CPU)

```
$ python -m src.play.server --device cpu --port 8765 --seed 0
모델 6개: base-9M, base-136M, base-270M, tal-136M, karpov-136M, smyslov-136M
장치 cpu, 샘플링 시드 0
```

- 탈(136M)로 새 대국을 열고 사용자 백 1.e4를 두었다. 모델은 1...c5(시실리안)를 π 0.725로 골랐다. 2위는 e5 0.180, 3위 e6 0.030이다. 첫 모델 수 응답은 베이스 불러오기를 포함해 2.8 s(CPU)였다.
- PGN 내려받기가 정상이다 (`[White "사용자"] [Black "탈 (136M)"] ... 1. e4 c5 *`).
- `GET /`는 `index.html`이 없어 처음에 500을 냈다. 파일이 없으면 404를 내도록 고쳤다.

## 5. 계획과 다른 점

1. **3회 반복 판정**: 계획서에는 `board.outcome(claim_draw=True)`를 쓴다고 적었다. 하지만 이 함수는 **다음 한 수로 3회 반복이 될 수 있기만 해도** 무승부로 본다. lichess처럼 같은 국면이 실제로 세 번 나왔을 때만 끝내기 위해 `board.is_repetition(3)`과 `board.is_fifty_moves()`를 직접 검사한다. `test_repetition_is_not_claimed_early`로 확인했다.
2. **`httpx` 폐기 예정 경고**: 설치된 starlette 1.10.0이 `TestClient`에서 `httpx`를 쓰면 `StarletteDeprecationWarning`을 내고 후속 패키지 `httpx2`를 권한다. 테스트는 정상 통과한다. `httpx2` 설치는 승인받은 의존성이 아니어서 하지 않았다 (설치 시도가 권한에서 거부됨). **작업지시자 결정**: (가) `httpx`로 유지하고 경고를 감수한다.
3. 차례 검사를 `Game.require_turn`으로 공개해, 서버가 모델을 부르기 전에 차례를 확인하게 했다. 계획에 없던 작은 추가다.

## 6. 다음 단계

3단계: `src/play/static/`의 화면 (lichess 참고 배치, 체스판, 기보, 후보·비교 패널). cburnett 기물 SVG 12개를 받아 넣는다.
