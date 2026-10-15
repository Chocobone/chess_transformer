# Task #3 단계1 완료 보고서: 사전 확인

## 수행 내용

카르포프 학습을 시작해도 되는 상태인지 확인했다 (2026-10-13 01:41). 모든 항목이 통과해, 작업지시자 지시에 따라 구현 계획서 승인 전인 01:42:04에 학습(2단계)을 시작했다. 이 보고서에는 사전 확인 결과와 학습 초반 기록을 적는다.

```bash
python -m pytest -q
sha256sum data/processed/karpov/base_q_136M.npz
nvidia-smi
bash -n runs/logs/train_player.sh
git diff --stat 06b57fa HEAD -- src scripts tests
```

학습 시작 명령:

```bash
setsid nohup runs/logs/train_player.sh karpov > runs/logs/train_karpov_136M.log 2>&1 < /dev/null &
```

## 변경 파일

소스 변경 없음.

| 파일 | 변경 내용 |
|------|----------|
| `mydocs/working/task_m100_3_stage1.md` | 이 보고서 |
| `mydocs/orders/20261013.md` | 작업일지 갱신 |

git에 올라가지 않는 산출물: `runs/karpov_136M/`(학습 진행 중), `runs/logs/train_karpov_136M.log`, `runs/logs/gpu_mem_karpov_136M.log`.

## 발견 사항

1. **학습 초반은 탈과 같은 수준으로 돈다.** step 120까지(01:45 확인) 처리량 24.3~25.2 국면/초(중앙값 25.0), 갱신당 약 1.28초, `max_memory_gib` 4.61 GiB다. 탈은 23.1~25.6 국면/초, 4.61 GiB였다. 120 갱신만 본 값이므로 전체 학습의 처리량은 2단계 보고서에 적는다.
2. **GPU 전체 사용량은 학습 초반에 약 8.2 GiB다** (`nvidia-smi`, 데스크톱 약 1.0 GiB 포함). 탈에서는 학습 중 7.5 GiB에서 9.98 GiB까지 늘었다. [정정 (2026-10-15, Task #6): ÷1024로 카르포프 초반 8.01 GiB (8,200 MiB), 탈 7.37 → 9.75 GiB (7,544 → 9,983 MiB)] 카르포프에서도 늘어나는지는 `runs/logs/gpu_mem_karpov_136M.log`로 본다.
3. **베이스 기준선** (val 5,213국면, 학습 전):

   | 지표 | 값 |
   |---|---|
   | imitation | 1.4941 |
   | top1 | 0.5586 |
   | top3 | 0.8565 |
   | regret | 0.01341 |
   | regret_top1 | 0 |
   | band_violation | 0.2658 |
   | strength | 0.00550 |

4. **실행 스크립트 머리말에 "Task #2"라고 적혀 있다.** `runs/logs/train_player.sh`는 git 제외 파일이고 동작에는 영향이 없어 그대로 두었다.

## 검증

| 항목 | 결과 |
|------|------|
| `pytest` | 111 passed |
| 캐시 SHA-256 | `4606d36323b43a79e43fe3c0ac9fc1bde7170bb840afceb8ea3ec1440c7ed8a3` — Task #2 1단계 보고서의 값과 같다 |
| GPU의 다른 작업 | 없음. 학습 전 사용량 1,034 MiB (데스크톱) |
| 이전 실행의 잔여물 | `runs/karpov_136M/`, `runs/logs/*karpov*` 없음 |
| 소스 동일성 | `git diff 06b57fa HEAD -- src scripts tests`가 비어 있다 (탈 학습 때와 같은 코드) |
| 실행 스크립트 문법 | `bash -n` 통과 |
| 학습 시작 | `config.json`의 `total_steps` 9,039. `시도 1 시작 2026-10-13 01:42:04 (체크포인트 step 0)`, 01:45 현재 중단 없음 |

실행 환경: RTX 4070 Ti 12GB, 드라이버 617.42, torch 2.14.1+cu130.

확인하지 않은 것:

- 캐시의 내용(국면 수, 합법수 순서, q 재계산)은 다시 검증하지 않았다. SHA-256이 Task #2 1단계에서 검증한 파일과 같다는 것으로 갈음한다.
- 호스트가 절전에 들어가지 않는지 (작업지시자 확인 필요).

## 다음 단계

2단계: 카르포프 학습이 끝나기를 기다린다 (중단이 없으면 05:35 전후 종료 추정). 끝나면 실행 경과·소요 시간·처리량·GPU 사용량을 `mydocs/working/task_m100_3_stage2.md`에 적는다. 재시도 한도 안에 끝나지 못하면 멈춘 상태로 두고 보고한다.

## 커밋

`Task #3 단계1: 사전 확인, 카르포프 학습 시작`
