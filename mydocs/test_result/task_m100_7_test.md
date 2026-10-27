# Task #7 테스트 결과: 대국 앱 실기 확인 (응답 시간, GPU 메모리)

## 환경

| 항목 | 값 |
|---|---|
| 장소 | 로컬 개발 컨테이너 `chess-dev` (WSL2) |
| GPU | NVIDIA GeForce RTX 4070 Ti 12282 MiB, 드라이버 617.42 |
| 소프트웨어 | Python 3.13.16, torch 2.14.1+cu130, fastapi 0.162.0, uvicorn 0.54.0, python-chess 1.11.2 |
| 정밀도 | float32 (학습 평가 설정 `eval_amp: false`와 같음) |
| 모델 | `base_model/{9M,136M,270M}.pt`, `runs/{tal,karpov,smyslov}_136M/adapter.pt` |
| 측정 전 GPU 사용량 | 1346~1348 MiB (`nvidia-smi`. 앱과 무관한 다른 프로그램 사용분) |

측정 스크립트는 저장소에 넣지 않았다 (스크래치패드). 날짜: 2026-10-27.

## 1. API 응답 시간 (서버 경유)

```bash
python -u -m src.play.server --host 0.0.0.0 --seed 0
python -I measure_api.py          # 클라이언트: urllib, 시간은 요청~응답 전체
```

방법: 모델마다 새 대국(사용자 백)을 열고, 사용자 쪽은 무작위 합법수(`random.Random(0)`)를 둔다. 모델 수 24개(`POST /api/game/engine`)의 시간을 잰다. 대국이 끝나면 같은 모델로 새 대국을 열어 이어 간다. 모델 순서는 9M → 136M 베이스 → 탈 → 카르포프 → 스미슬로프 → 270M → 탈이다. 크기가 바뀌는 첫 수에는 베이스 불러오기 시간이 들어간다. `nvidia-smi`는 각 모델 측정이 끝난 직후의 전체 사용량이다.

원 출력:

```
시작 nvidia-smi 1347 MiB
base-9M       첫 수   516.2 ms | 이후 중앙값 8.7 ms, 최대 12.0 ms, n=23 | nvidia-smi 1761 MiB
base-136M     첫 수  2301.1 ms | 이후 중앙값 54.7 ms, 최대 60.3 ms, n=23 | nvidia-smi 3287 MiB
tal-136M      첫 수    37.0 ms | 이후 중앙값 41.5 ms, 최대 50.9 ms, n=23 | nvidia-smi 3287 MiB
karpov-136M   첫 수    33.6 ms | 이후 중앙값 47.8 ms, 최대 81.2 ms, n=23 | nvidia-smi 3689 MiB
smyslov-136M  첫 수    33.1 ms | 이후 중앙값 56.9 ms, 최대 70.8 ms, n=23 | nvidia-smi 3689 MiB
base-270M     첫 수  9130.6 ms | 이후 중앙값 90.9 ms, 최대 110.8 ms, n=23 | nvidia-smi 4604 MiB
tal-136M      첫 수  5497.6 ms | 이후 중앙값 54.9 ms, 최대 115.7 ms, n=23 | nvidia-smi 3173 MiB
/api/analyze (모델 4개) 중앙값 180.4 ms, 최대 226.9 ms, n=20 | nvidia-smi 3174 MiB
/api/analyze 270M → 136M 전환 포함 1935.8 ms | nvidia-smi 2729 MiB
```

| 항목 | 결과 |
|---|---|
| 모델 수 (크기 전환 없음) | 9M 중앙값 8.7 ms · 136M 41.5~56.9 ms · 270M 90.9 ms. 최대 115.7 ms |
| 크기 전환 뒤 첫 수 | 9M 0.5 s, 136M 2.3 s(처음) / 5.5 s(270M 다음), 270M 9.1 s |
| 어댑터 전환 (136M 안에서 탈 → 카르포프 → 스미슬로프) | 첫 수 33~37 ms. 이후 수의 중앙값보다 짧다. 어댑터 바꿔 끼우기 비용은 측정 잡음 안에 든다 (첫 수는 시작 국면 직후라 합법수가 적다) |
| 비교 패널 `/api/analyze` (모델 4개, 136M 국면 20개) | 중앙값 180.4 ms, 최대 226.9 ms |
| 270M 사용 중 비교 패널 첫 계산 | 1.9 s (136M 다시 올리기 포함) |

- 136M을 두 번째로 올릴 때(5.5 s)가 처음(2.3 s)보다 느렸다. 원인은 따로 재지 않았다. 직전에 내린 270M의 메모리 정리와 디스크 캐시 상태가 영향을 줬을 수 있다. 대화형 사용에는 지장 없는 수준이다.

## 2. GPU 메모리 (엔진 직접 호출)

```bash
python measure_memory.py          # 서버를 내린 뒤 실행. Engine(default_specs(), "cuda")
```

방법: 모델마다 `torch.cuda.reset_peak_memory_stats()` 뒤 국면 3개(시작 국면, 중반 국면 2개)에서 `choose`를 부른다. 그다음 최대 할당량, 현재 할당량, 캐시 예약량, `nvidia-smi`를 잰다. 크기 순서는 9M → 136M(베이스·어댑터 3개) → 270M → 136M → 9M이다.

원 출력:

```
시작: allocated 0 MiB, nvidia-smi 1346 MiB
base-9M       max_allocated    87.1 MiB | allocated    42.3 MiB | reserved   142.0 MiB | nvidia-smi 1723 MiB | 국면 3개 499 ms
base-136M     max_allocated  1051.0 MiB | allocated   533.1 MiB | reserved  1286.0 MiB | nvidia-smi 2867 MiB | 국면 3개 2079 ms
tal-136M      max_allocated   705.9 MiB | allocated   533.1 MiB | reserved  1286.0 MiB | nvidia-smi 2867 MiB | 국면 3개 124 ms
karpov-136M   max_allocated   705.9 MiB | allocated   533.1 MiB | reserved  1286.0 MiB | nvidia-smi 2867 MiB | 국면 3개 123 ms
smyslov-136M  max_allocated   705.9 MiB | allocated   533.1 MiB | reserved  1286.0 MiB | nvidia-smi 2867 MiB | 국면 3개 123 ms
base-270M     max_allocated  2075.2 MiB | allocated  1041.2 MiB | reserved  2308.0 MiB | nvidia-smi 3889 MiB | 국면 3개 4434 ms
tal-136M      max_allocated  1051.0 MiB | allocated   533.1 MiB | reserved  1286.0 MiB | nvidia-smi 2867 MiB | 국면 3개 2054 ms
base-9M       max_allocated   533.1 MiB | allocated    42.3 MiB | reserved   150.0 MiB | nvidia-smi 1731 MiB | 국면 3개 195 ms
```

| 크기 | 상주 (`allocated`) | 최대 (`max_allocated`) | 앱 몫 `nvidia-smi` (측정값 − 1346) |
|---|---|---|---|
| 9M | 42.3 MiB | 87.1 MiB | 377 MiB (0.37 GiB) |
| 136M (LoRA 포함) | 533.1 MiB | 1051.0 MiB (불러오는 중), 추론만 705.9 MiB | 1521 MiB (1.49 GiB) |
| 270M | 1041.2 MiB | 2075.2 MiB (불러오는 중) | 2543 MiB (2.48 GiB) |

- **이전 크기가 내려가는지**: 내려간다. 270M → 136M에서 `allocated`가 1041.2 → 533.1 MiB, 다시 9M로 가면 42.3 MiB가 된다. `reserved`도 2308 → 1286 → 150 MiB로 줄어 `nvidia-smi` 사용량이 9M 처음 값(1723 MiB)과 거의 같은 1731 MiB로 돌아온다.
- 최대 할당량은 불러올 때 상주량의 약 2배다. 불러오는 동안 가중치가 잠시 두 벌 있기 때문이다. 크기를 바꿀 때의 마지막 행(9M 최대 533.1 MiB)은 이전 136M을 내리기 전 값이다.
- 서버 측정(1절)에서는 앱 몫이 136M 최대 2342 MiB, 270M 3257 MiB로 더 컸다. 합법수가 많은 국면을 여러 번 지나며 PyTorch 캐시(`reserved`)가 커진 것이다. 12 GB GPU에서 270M까지 여유가 크다.

## 3. 모델별 대국 시작 확인

1절 측정에서 6개 모델(베이스 9M·136M·270M, 탈·카르포프·스미슬로프 136M) 모두 새 대국이 열리고 모델 수 24개를 두었다. 오류 응답은 없었다.

## 4. 매뉴얼 절차 확인

환경 변수를 비운 새 로그인 셸에서 매뉴얼 3절 명령을 그대로 실행했다.

```
$ env -i HOME=$HOME PATH=$PATH bash -lc 'cd /workspace && python -u -m src.play.server --host 0.0.0.0'
모델 6개: base-9M, base-136M, base-270M, tal-136M, karpov-136M, smyslov-136M
장치 cuda, 샘플링 시드 54486295
http://localhost:4321
INFO:     Uvicorn running on http://0.0.0.0:4321 (Press CTRL+C to quit)
```

`GET /` 200, `GET /api/game` 404(대국 없음, 정상).

## 5. 실기 대국 (작업지시자)

작업지시자가 Windows 브라우저(`http://localhost:4321`)로 탈(136M)과 한 판 두었다. 사용자 백, 2026-10-27.

| 항목 | 결과 |
|---|---|
| 결과 | 1-0, 57수(113플라이) 만에 백의 체크메이트 |
| 종료 판정 | 체크메이트를 판정해 대국이 끝났다. PGN 헤더 `Result "1-0"`, `Termination "checkmate"` |
| 승진 | 51. a8=Q (화면 승진 창으로 퀸 선택) |
| SAN 표기 | 퀸 세 개가 있는 국면의 구분 표기(`Qac8+`, `Qbd7+`, `Qce8+`, `Qdf7+`, `Qeg8+`, `Qfh7#`)가 맞게 나왔다 |
| PGN 검증 | 내려받은 PGN을 python-chess로 다시 읽어 113플라이를 모두 두었다. 파싱 오류 없음, 마지막 국면 체크메이트 확인 |
| 이상 동작 | 작업지시자 보고 없음 |

- 모델은 14. Nxa8, 20. Rxh8로 룩 두 개를 잃는 등 초반부터 기물을 크게 잃었다. 탐색 없이 π(T = 0.02)에서 한 수씩 샘플링하는 방식이라 수읽기가 필요한 전술에 약한 것으로 보인다. 앱 결함이 아니라 모델 기력의 문제이며, 이 타스크(확인용 앱)의 범위 밖이다.

PGN (앱에서 내려받은 그대로):

```
[Event "모델과 대국"]
[Site "chess_transformer 대국 앱"]
[Date "2026.10.27"]
[Round "?"]
[White "사용자"]
[Black "탈 (136M)"]
[Result "1-0"]
[Termination "checkmate"]

1. e4 e5 2. Nf3 d5 3. exd5 Qxd5 4. Nc3 Qd7 5. Nxe5 Qe7 6. d4 Nc6 7. Bb5 f6 8. Nd5 Qd6 9. Nxc6 bxc6 10. Qf3 cxb5 11. Bf4 Qe6+ 12. Kd2 Kf7 13. Nxc7 Qc4 14. Nxa8 Qxd4+ 15. Kc1 Bd6 16. Bxd6 Qxd6 17. a4 b4 18. Rd1 Qxh2 19. Rd8 Ne7 20. Rxh8 Qh1+ 21. Kd2 Qxa1 22. Qh5+ Ke6 23. Re8 Qxb2 24. Nc7+ Kd7 25. Qb5+ Nc6 26. Qf5+ Kxc7 27. Rxc8+ Kd6 28. Qd3+ Nd4 29. Rd8+ Kc5 30. Rc8+ Kb6 31. Rc4 b3 32. Rxd4 bxc2 33. Rd6+ Kc7 34. Rd7+ Kc8 35. Rd8+ Kb7 36. Qd5+ Ka6 37. Rd6+ Qb6 38. Kxc2 h5 39. Rxb6+ Kxb6 40. Qf7 g6 41. Qxg6 Kc7 42. Qxf6 h4 43. Qxh4 Kb6 44. Qb4+ Kc6 45. a5 Kc7 46. a6 Kc6 47. Qb7+ Kc5 48. Qxa7+ Kd6 49. Qb7 Ke6 50. a7 Kd6 51. a8=Q Kc5 52. Qac8+ Kd6 53. Qbd7+ Ke5 54. Qce8+ Kf6 55. Qdf7+ Kg5 56. Qeg8+ Kh6 57. Qfh7# 1-0
```

## 6. pytest

```
$ python -m pytest -q
142 passed, 57 warnings in 44.14s
```
