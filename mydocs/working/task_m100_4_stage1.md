# Task #4 1단계 완료 보고서: 사전 확인, 사전 측정, 대책 비교

## 요약

사전 확인은 모두 통과했다. 기본 설정의 사전 측정에서 평가 시간(195초)과 평가 사이 증가(없음)는 정상이었지만, GPU 전체 사용량이 최대 11.29 GiB로 정상 기준(11.0 GiB)을 넘었다. 대책 (가) `--eval-max-moves 128`과 (나) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 같은 조건으로 재 보니 둘 다 기준을 통과했다. **(나)를 권고한다.** GPU 최대가 5.57 GiB로 가장 낮고, 지표·어댑터가 기본 설정과 완전히 같으며, `config.json`과 실행 스크립트를 바꾸지 않는다. 소스는 수정하지 않았다.

## 사전 확인 (03:11)

| 항목 | 결과 |
|------|------|
| `python -m pytest -q` | 111 passed |
| 캐시 SHA-256 | `b0904be63c28c1f4318e816f99c5ec6f398b1f4aab51915fa8a61e6b1b82f87e` — Task #2 1단계 보고서 값과 일치 |
| `git diff 06b57fa HEAD -- src scripts tests` | 비어 있음 |
| `bash -n runs/logs/train_player.sh` | 통과 |
| GPU | 학습·측정 프로세스 없음, 791 MiB 사용 |
| `runs/smyslov_136M/`, `runs/logs/*smyslov_136M*` | 없음 |

## 사전 측정과 대책 비교

상세는 `mydocs/test_result/task_m100_4_test.md`의 "사전 측정".

| 항목 | 기본 (`base`) | (가) `emm128` | (나) `expand` | 정상 기준 |
|------|---------------|---------------|---------------|-----------|
| 평가 1회 시간 | 194.6~195.0초 | 195.1~195.3초 | 194.1초 | 260초 이하 |
| GPU 전체 최대 | **11.29 GiB** | 9.95 GiB | 5.57 GiB | 11.0 GiB 이하 |
| 데스크톱 제외 최대 | 10.51 GiB | 9.30 GiB | 4.91 GiB | |
| 평가 사이 증가 | −5 MiB | 0 | 0 | 0.3 GiB 이하 |
| 학습 처리량 | 24.4~24.9 국면/초 | 24.6~25.3 | 24.6~25.3 | |
| val 지표 차이 (기본 대비) | | 최대 8.2e-08 | 0 | 1e-6 / 0 |
| step 60 `adapter.pt` | `1bbe9922…` | 같음 | 같음 | |
| `config.json` 변화 | | `eval_max_moves` 256 → 128 | 없음 | |
| 판정 | 기준 초과 | 통과 | 통과 | |

- 기본 측정의 기준선은 어제 단독 측정의 스미슬로프 기준선과 같다.
- 기본 설정에서는 학습만 돌 때 7.86 GiB(8,046 MiB)였다가 첫 평가가 시작되자 30초 만에 11.29 GiB가 되었고, 이후 그대로였다.
- (가)는 평가가 더하는 양만 줄인다 (데스크톱 제외 −1.22 GiB).
- (나)는 학습이 잡는 양도 줄인다. 학습만 돌 때 5.57 GiB였고 평가가 시작돼도 늘지 않았다.
- 계획서는 (가)가 통과하면 (나)를 재지 않아도 되는 순서였다. 그러나 (가)는 `config.json`이 탈·카르포프와 달라지고 (나)는 달라지지 않으므로, 비교 자료를 위해 (나)도 쟀다 (11분).

## 권고와 결정이 필요한 사항

1. **대책 선택**
   - 안 (나) (권장): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
     - GPU 여유가 6.4 GiB로 가장 크다. 데스크톱 사용량이 늘거나 평가 뒤에 누적이 생겨도 넘칠 가능성이 가장 낮다.
     - 60 갱신의 어댑터와 지표가 기본 설정과 비트 단위로 같다. `config.json`, 실행 스크립트, 소스 모두 그대로다.
     - 걸리는 점: 환경변수는 `config.json`·`log.jsonl`에 남지 않는다. 결과 문서와 보고서에 명시하고, 실행 로그 첫 줄에 남기는 방법을 아래 실행 명령에 넣었다.
   - 안 (가): `--eval-max-moves 128`. GPU 최대 9.95 GiB로 여유가 2.0 GiB다. `config.json`의 `eval_max_moves`가 탈·카르포프와 다르고, 지표에 1e-7 수준 차이가 있다. `train_player.sh`에 인자 전달을 넣어야 한다.
   - 안 (가)+(나): 둘 다 쓴다. 이득이 거의 없다고 보아 권하지 않는다.
2. **학습 시작 승인**: 승인 후 바로 아래 명령으로 시작한다. 스크립트는 수정하지 않고 환경변수를 넘긴다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup bash -c 'echo "=== PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"; exec runs/logs/train_player.sh smyslov' \
    > runs/logs/train_smyslov_136M.log 2>&1 < /dev/null &
```

## 2단계 소요 시간 재추정

사전 측정의 처리량(약 24.9 국면/초, 갱신당 1.29초)과 평가 시간(195초)으로 계산했다.

| 구간 | 추정 |
|------|------|
| 학습 8,736 갱신 | 약 3.1시간 |
| val 평가 9회 + 기준선 | 약 29분 |
| **합계** | **약 3.6시간** (중단이 없고 평가가 정상일 때) |

## 산출물

- `mydocs/test_result/task_m100_4_test.md` (신규, "사전 측정" 절)
- `runs/smyslov_probe_{base,emm128,expand}/`, `runs/logs/probe_smyslov_*.log` (git 제외, 본 학습에는 쓰지 않는다)

## 계획 대비 차이

- (가)가 통과한 뒤에도 비교를 위해 (나)를 쟀다 (위 설명).
- 사전 측정 결과를 보고 승인받은 뒤 학습을 시작하는 것은 계획대로다. 학습은 아직 시작하지 않았다.

## 승인 (2026-10-14 작업지시자)

1단계 보고를 승인받았다. 대책은 안 (나) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 정했다.
