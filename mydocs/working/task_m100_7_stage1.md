# Task #7 1단계 완료 보고서: 추론 엔진 모듈과 테스트

## 요약

- 웹과 무관한 추론 엔진 `src/play/engine.py`를 만들었다. 모델 id와 FEN을 받아 학습과 같은 정책 π(a|s) = softmax(q / 0.02)를 계산하고, 그 π에서 수를 샘플링한다.
- 테스트 11건을 추가했다 (`tests/test_play_engine.py`). 전체 127건(기존 116 + 신규 11)이 통과한다. 새 테스트는 `-W error`로 돌려도 통과한다. 경고 56건은 기존 테스트에서 나오던 것이다 (추가 전에도 56건).
- 학습·모델 코드는 수정하지 않았다.
- 실제 가중치로 CPU에서 간단히 확인했다. 같은 국면에서 네 모델의 후보가 서로 다르게 나온다 (3절). 정식 실측은 4단계에서 GPU로 한다.

## 1. 엔진 API (`src/play/engine.py`)

| 이름 | 내용 |
|---|---|
| `TEMPERATURE = 0.02` | 학습 설정과 같은 온도 |
| `ModelSpec(id, label, size, base, adapter)` | 모델 하나. 베이스만 쓰면 `adapter=None` |
| `Candidate(uci, san, prob, q)` | 후보 수. `prob` = π(a\|s), `q` = 둘 차례인 쪽의 예상 승률 |
| `Analysis(model_id, candidates, win_prob, legal_count)` | 한 국면의 평가. `candidates`는 π 내림차순 상위 k개, `win_prob` = max q |
| `default_specs(base_dir, runs_dir)` | `base-9M`·`base-136M`·`base-270M`, `tal-136M`·`karpov-136M`·`smyslov-136M` 가운데 파일이 있는 것 |
| `parse_board(fen)` | 잘못된 FEN, 규칙에 맞지 않는 국면, 종료 국면이면 `ValueError` |
| `Engine(specs, device, loader)` | `device`를 주지 않으면 CUDA가 있을 때 cuda를 쓴다. `loader`는 테스트에서 소형 모델을 넣는 자리다 |
| `Engine.analyze(model_id, fen, top_k=5)` | 합법수 전체를 평가해 상위 k개 후보를 돌려준다 |
| `Engine.choose(model_id, fen, generator, top_k=5)` | π에서 하나를 샘플링한다 (`torch.multinomial`). 고른 수와 `Analysis`를 돌려준다 |
| `Engine.compare(model_ids, fen, top_k=3)` | 같은 국면을 여러 모델로 평가한다. 결과는 요청한 순서다. 비교 패널용 |
| `COMPARE_IDS` | 비교 패널 기본 모델 4개 (`base-136M`, `tal-136M`, `karpov-136M`, `smyslov-136M`) |

**정책 계산**: `legal_action_tokens([board.fen()])` → `model.win_prob` → `policy_from_q(q, owner, 0.02)`. 학습·평가 코드의 함수를 그대로 부른다. 수 목록은 `ordered_legal_moves` 순서다. 토큰의 행동 번호가 이 순서와 같은지 assert로 확인한다. 계산은 float32와 `torch.inference_mode()`로 한다.

**모델 관리**
- 어댑터 파일은 시작할 때 모두 CPU로 읽는다. 136M 어댑터 하나가 약 4 MB다. 세 어댑터의 settings(rank, alpha, dropout, targets)가 다르거나, 136M 베이스 경로가 둘 이상이거나, 어댑터가 136M이 아닌 베이스에 붙어 있으면 시작할 때 `ValueError`를 낸다.
- 베이스는 처음 쓸 때 불러온다. 다른 크기가 필요하면 지금 크기를 내리고(`torch.cuda.empty_cache()`) 새로 불러온다. 메모리에는 늘 한 크기만 있다.
- 136M을 불러올 때 LoRA를 한 번 주입한다. 그 뒤로는 모델 id에 맞는 LoRA 가중치를 `load_lora`로 끼운다. `base-136M`에는 모두 0인 가중치를 끼운다. 지금 끼워진 id를 기억해 두고, 같은 id면 다시 끼우지 않는다.
- 엔진은 스레드 안전하지 않다. 서버(2단계)가 잠금으로 동시 호출을 막는다.

## 2. 테스트 (`tests/test_play_engine.py`)

실제 가중치 없이 돈다. 크기별 베이스는 seed가 다른 소형 모델 `TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)`이고, `loader`로 엔진에 넣는다. 어댑터는 소형 136M 베이스에 LoRA(rank 4)를 넣고 `lora_B`를 무작위로 채워 `save_adapter`로 만든다 (탈·카르포프 두 개).

| 테스트 | 확인 내용 | 결과 |
|---|---|---|
| `test_policy_matches_training_code` | 시작 국면과 중반 국면에서, 엔진의 수별 (π, q)가 학습 코드 경로로 직접 계산한 값과 **정확히 같다**. 비교한 모델은 베이스 270M과, 단독으로 `load_adapter`한 탈 모델이다. π의 합은 1이다 | 통과 |
| `test_choose_returns_legal_moves` | 시작·승진·앙파상·캐슬링·중반 국면에서 고른 수가 합법수이고, SAN이 python-chess와 같다 | 통과 |
| `test_sampling_is_reproducible` | 같은 시드면 30수가 같고, 시드가 다르면 다르다 | 통과 |
| `test_sampling_follows_policy` | 2,000회 뽑은 빈도와 π의 차이가 모든 수에서 0.05 미만이다 | 통과 |
| `test_adapter_switching` | 탈 → 카르포프 → 탈 → 베이스 136M → 카르포프 → 베이스 9M → 탈 → 베이스 136M 순으로 바꾼다. 매번 출력이 그 어댑터를 단독으로 실은 모델과 정확히 같다. 베이스 136M은 LoRA 없는 베이스와 같다 | 통과 |
| `test_only_one_size_is_loaded` | 9M → 136M(탈) → 270M으로 바꿀 때 엔진에 한 크기만 남는다 | 통과 |
| `test_candidates_are_sorted_and_truncated` | 후보가 π 내림차순 상위 k개다. 합법수가 k보다 적으면 합법수만큼 나온다. `win_prob` = max q | 통과 |
| `test_errors` | 없는 모델, 체크메이트 국면, 잘못된 FEN, 왕 없는 국면, `compare`의 없는 모델에서 `ValueError`가 난다 | 통과 |
| `test_compare` | 요청한 순서로 결과가 나오고, 각 결과가 `analyze`와 같다 | 통과 |
| `test_mismatched_adapter_settings` | rank가 다른 어댑터를 섞으면 시작할 때 `ValueError`가 난다 | 통과 |
| `test_default_specs` | 있는 파일만 목록에 들어간다. 136M 베이스가 없으면 어댑터도 빠진다 | 통과 |

```
$ python -m pytest -q
127 passed, 56 warnings in 40.43s
$ python -m pytest -q tests/test_play_engine.py -W error
11 passed in 10.09s
```

## 3. 실제 가중치 확인 (CPU, 참고용)

`default_specs()`는 로컬의 6개 모델을 모두 찾았다. 백 차례인 이탈리안 게임 국면(`r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4`)에서 상위 3수는 다음과 같다.

| 모델 | 1수 (π, q) | 2수 | 3수 | 예상 승률 |
|---|---|---|---|---|
| 베이스 136M | Ng5 (0.265, 0.527) | d3 (0.223, 0.524) | Nc3 (0.169, 0.518) | 0.527 |
| 탈 | **d4** (0.734, 0.587) | Ng5 (0.177, 0.558) | Nc3 (0.047, 0.532) | 0.587 |
| 카르포프 | Ng5 (0.419, 0.517) | d3 (0.357, 0.514) | Qe2 (0.128, 0.493) | 0.517 |
| 스미슬로프 | **d3** (0.437, 0.547) | Ng5 (0.303, 0.539) | Nc3 (0.195, 0.530) | 0.547 |
| 베이스 9M | Nc3 (0.246, 0.523) | d3 (0.199, 0.519) | O-O (0.160, 0.515) | 0.523 |

- 시간은 CPU 기준으로 136M 첫 호출 3.79 s(불러오기 포함), 어댑터 전환 뒤 1.2~1.3 s, 9M 0.31 s다. GPU 실측은 4단계에서 한다.
- 탈은 중앙을 여는 d4(개방 전개)에 확률을 몰았고, 스미슬로프는 조용한 d3을 1순위로 뒀다. 국면 하나의 관찰일 뿐이므로 기풍의 근거로 쓰지는 않는다. 다만 비교 패널로 이런 차이를 볼 수 있음은 확인했다.
- 어댑터는 q 자체도 바꾼다 (탈의 d4 q 0.587, 베이스의 최대 q 0.527). LoRA가 가치 머리를 공유하므로 생기는 현상이다. 화면의 "예상 승률"은 그 모델이 본 값이라고 표시한다 (3단계).

## 4. 계획 대비

- 구현 계획서대로 진행했다. 바꾼 점은 없다.
- `Engine.choose`에도 `top_k` 인자를 두었다. 서버가 모델 수와 함께 후보 상위 5개를 보내기 위해서다.

## 5. 다음 단계

2단계: `src/play/game.py`(대국 상태), `src/play/server.py`(FastAPI), `tests/test_play_server.py`. `requirements.txt`에 `fastapi`, `uvicorn`, `httpx`를 추가하고 설치한다.
