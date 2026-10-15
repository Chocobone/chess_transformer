# Task #4 2단계 완료 보고서: 스미슬로프 어댑터 학습

## 요약

스미슬로프의 어댑터를 3 epoch(8,736 갱신) 끝까지 학습했다. 1단계에서 정한 대책(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)으로 돌렸고, 평가 9회가 모두 약 194초로 정상이었다. GPU 전체 사용량은 최대 6.07 GiB [정정 (2026-10-15, Task #6): ÷1024로 5.92 GiB (6,066 MiB)]에 머물렀다. 학습 구간 처리량도 끝까지 정상이었다. 중간에 CUDA 오류(device-side assert)로 한 번 멈췄고, 스크립트가 체크포인트에서 자동 재개했다. 전체 소요는 3시간 37분으로 추정(3.6시간)과 맞았다. 멈출 기준에는 한 번도 닿지 않았다. 소스는 수정하지 않았다.

## 실행

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup bash -c 'echo "=== PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"; exec runs/logs/train_player.sh smyslov' \
    > runs/logs/train_smyslov_136M.log 2>&1 < /dev/null &
```

스크립트가 실행한 명령은 `python -u -m src.train.train --player smyslov --base base_model/136M.pt --out runs/smyslov_136M --eval-every 1000 --resume`이다. 두 시도 모두 학습 프로세스의 환경에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 있음을 `/proc/{pid}/environ`으로 확인했다.

## 실행 경과

| 시도 | 시작 | 종료 | 결과 |
|------|------|------|------|
| 1 | 04:08:30 | 06:21:49 | step 5,400~5,420 사이에서 CUDA 오류로 종료 (종료 코드 1). 마지막 체크포인트 step 5400 [정정 (2026-10-15, Task #5): 중단은 step 5,580~5,600 사이 (실패한 시도의 마지막 train 줄이 step 5580), 잃은 갱신 180~199] |
| 2 | 06:22:11 | 07:45:06 | step 5400에서 재개해 8,736 갱신까지 완료 |

- 전체 3시간 36분 36초 (04:08:30~07:45:06). 로그의 누적 `elapsed`는 마지막 갱신까지 12,531초다 (평가 8회 포함, 마지막 평가 제외).
- 중단으로 잃은 시간은 약 4분이다. step 5400 체크포인트(약 06:18) 뒤 시도 1이 멈출 때까지의 갱신과, 재시작 22초를 합한 값이다. [정정 (2026-10-15, Task #5): 체크포인트 뒤 시도 1이 기록한 구간이 225초(step 5400→5580)이고, 기록되지 않은 최대 19 갱신과 재시작 22초를 더해 약 4~4.5분. 잃은 갱신은 180~199개]

### 중단 (시도 1)

```
vectorized_gather_kernel: ... Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
  File "/workspace/src/train/train.py", line 342, in train
    totals = accumulate_gradients(model, batch, objective, args.max_moves_per_forward, amp)
  File "/workspace/src/train/train.py", line 77, in accumulate_gradients
    loss, parts = chunk_loss(q, chunk, objective)
  ...
  File "/workspace/src/train/policy.py", line 32, in pad_by_owner
    counts = torch.bincount(owner)
torch.AcceleratorError: CUDA error: device-side assert triggered
```

- 오류는 `bincount`에서 보고됐지만, CUDA 오류는 비동기로 보고되므로 실제로 실패한 커널은 그 앞의 gather다. 어느 연산인지는 이 정보만으로는 모른다.
- 탈 학습(Task #2)의 1차·3차 중단과 같은 메시지다. 탈에서는 같은 구간을 다시 돌려도 재현되지 않았고, 실행 환경의 불안정으로 판단했다.
- 직전 GPU 전체 사용량은 5,824 MiB로 메모리 초과는 아니다. 처리량도 직전까지 정상이었다.
- 재개 뒤 첫 구간(step 5400→5420) 소요 26.4초는 평소(25.1초)와 같다. 그 뒤로 추가 오류는 없었다.

## 학습 중 지표

| 항목 | 결과 | 멈출 기준 |
|------|------|-----------|
| 평가 1회 소요 (step 1000~8000) | 193.3~194.5초 | 400초 초과 |
| 마지막 평가 (step 8736) | 약 190초 (벽시계로 어림) | |
| 학습 처리량 | 최저 23.9, 중앙값 25.5, 최고 26.5 국면/초 | 18 미만이 10분 넘게 지속 |
| GPU 전체 사용량 (`nvidia-smi` 60초) | 시도 1: 5,739~6,066 MiB, 시도 2: 5,764~5,795 MiB | 11.8 GiB(12,083 MiB) 이상 |
| `max_memory_gib` | 4.57 GiB | |
| 온도 | 최고 72℃ | |
| 손실·기울기 노름 | NaN·무한대 없음, 기울기 노름 최대 1.811 (clipping 1.0) | |

평가 1회의 소요는 평가가 낀 20 갱신 구간의 `elapsed` 간격에서 평소 간격(중앙값 25.1초)을 뺀 값이다. 마지막 평가는 뒤에 train 줄이 없어서 벽시계로 어림했다: 시도 2 시작 + 재개 뒤 학습 시간 → 07:42 전후, `adapter.pt` 저장 07:45:04.

| 평가 step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 |
|---|---|---|---|---|---|---|---|---|
| 소요 (초) | 193.9 | 193.8 | 194.1 | 193.3 | 194.5 | 193.7 | 193.7 | 193.8 |

카르포프의 같은 표(8.6~38.6분)와 달리 평가 시간이 처음부터 끝까지 일정했다. 카르포프에서 처리량이 떨어진 구간(step 4,260~6,200)에서도 처리량은 정상이었다.

## val 지표 (4,753국면 전체, float32)

| step | imitation | top1 | top3 | regret_top1 |
|---|---|---|---|---|
| 기준선 | 1.4497 | 0.5639 | 0.8515 | 0 |
| 1000 | 1.2052 | 0.6122 | 0.8723 | 0.00170 |
| 2000 | 1.1420 | 0.6215 | 0.8818 | 0.00254 |
| 3000 | 1.1201 | 0.6228 | 0.8860 | 0.00277 |
| 4000 | 1.1083 | 0.6242 | 0.8900 | 0.00288 |
| 5000 | 1.1015 | 0.6276 | 0.8900 | 0.00288 |
| 6000 | 1.0954 | 0.6318 | 0.8906 | 0.00288 |
| 7000 | 1.0956 | 0.6335 | 0.8906 | 0.00288 |
| 8000 | 1.0936 | 0.6327 | 0.8908 | 0.00291 |
| 8736 | 1.0936 | 0.6335 | 0.8908 | 0.00293 |

나머지 지표(regret, band_violation, strength)와 로그 원문은 3단계에서 `mydocs/test_result/task_m100_4_test.md`에 스크립트로 정리한다. epoch별 train 줄 평균 imitation은 1.2287 / 1.0851 / 1.0674다.

## 산출물 (`runs/smyslov_136M/`, git 제외)

| 파일 | 크기 | 비고 |
|------|------|------|
| `adapter.pt` | 4,213,603 | SHA-256 `b0ae0ea5e2bd247b04b7e9b8957173156cc5a6a7b8b764758a7c4a17fd9ef791` |
| `checkpoint.pt` | 12,680,115 | 재개용, step 8736 |
| `config.json` | 1,029 | |
| `log.jsonl` | 143,520 | |

로그 `runs/logs/train_smyslov_136M.log`, `runs/logs/gpu_mem_smyslov_136M.log`.

## 검증

- 로그에 `=== smyslov 완료 2026-10-14 07:45:06 (시도 2회, step 8736)` 줄이 있다 (종료 코드 0일 때만 찍힌다)
- `runs/smyslov_136M/`에 네 파일이 모두 있다
- `log.jsonl`의 마지막 `eval` step(8736)이 `config.json`의 `total_steps`(8736)와 같다
- train 줄 437개의 step이 20, 40, …, 8720, 8736으로 간격이 20(마지막만 16)이다. 재개 지점(5400)에서 빠짐·겹침이 없다
- 학습·감시 프로세스가 모두 끝났다

## 다음 단계 (3단계)

어댑터 재평가, 로그 연속성·NaN 검사(위의 결과를 스크립트로 다시 확인), 탈·카르포프와의 `config.json` 대조, 결과 문서 작성. 결과 문서에는 환경변수(`expandable_segments:True`)가 `config.json`에 남지 않는다는 점을 명시한다.

## 확인하지 못한 것

- CUDA 오류의 원인. 범위 외이며, 탈에서와 같이 자동 재개로 대응했다.
- 재개 뒤 구간(step 5400 이후)이 끊기지 않은 학습과 같은지. 비교할 실행이 없고, Task #2의 테스트와 60 갱신 GPU 확인으로만 뒷받침된다.
  - **정정 (2026-10-15, Task #5)**: 같지 않았다. 수정 전 코드는 `--num-workers` > 0에서 epoch 1 이후에 재개하면 데이터 순서가 끊기지 않은 학습과 달라졌다. 스미슬로프는 step 5400(epoch 1)에서 재개해, epoch 1에서 국면 11,597개(12.45%)가 두 번, 같은 수가 한 번도 학습되지 않았다. epoch 2는 빠짐없이 학습했지만 순서가 다르다 (`mydocs/troubleshootings/resume_data_order.md`).
