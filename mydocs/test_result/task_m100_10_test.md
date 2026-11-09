# Task #10 테스트 결과: 서버 smoke (Slurm + Miniconda)

`gpu_train/README.md` 2~6절을 따라 작업지시자가 GPU 서버에서 실행했다. 이 문서는 작업지시자가 로컬로 옮긴 결과 묶음 `smoke_20261109_115519.tar.gz`(`collect.sh smoke` 출력)를 정리한 것이다 (2026-11-09).

## 환경

| 항목 | 값 |
|---|---|
| 서버 | moana (로그인 노드 `moana-master`, 계산 노드 `moana-u1`) |
| 저장소 경로 | `/data/yho7374/repos/chess_transformer` |
| 코드 커밋 | `f39318a57d43f087be8181dd8c1ed371869a5975` (Task #10 단계5) |
| 미커밋 변경 | `M gpu_train/site.env.example` (아래 "짚어 둘 점" 2) |
| GPU | NVIDIA RTX A2000, 6138 MiB, 드라이버 535.216.03 |
| torch | 2.5.1+cu121, CUDA 12.1, `GPU 보임 True` |
| Slurm 작업 | 153212 (사슬 1번 / 10), 예약된 다음 작업 153213 |
| 실행 시각 | 2026-11-05 15:58:46 시작 ~ 16:05:03 완료. 묶은 시각 2026-11-09 11:55:19 |

## 받지 못한 자료

구현 계획서가 이 문서에 넣기로 한 것 중 아래 세 가지는 **서버 권한 문제로 출력을 받지 못했다** (작업지시자 확인, 2026-11-09). 각각 다른 자료로 대신 확인한 범위를 적는다.

| 자료 | 대신 확인한 것 | 확인하지 못한 것 |
|---|---|---|
| `setup_env.sh` 출력 끝부분 | 환경이 만들어져 학습이 돌았고, torch·CUDA 버전과 GPU 인식은 작업 기록에 있다 | 서버의 pytest 결과, Python 버전, `pip freeze` (묶음에 `env_*.txt`가 없다) |
| `fetch_inputs.sh` 출력 끝부분 | 작업 시작 시 `check_inputs.sh`가 13개를 통과했다. 매니페스트는 git(`f39318a`)의 것이므로 입력 내용은 로컬과 같다 | 받은 리비전 문자열 |
| `sacct --name=smoke` 출력 | 작업 기록에 다음 작업 153213 예약과 취소가 남아 있다 | Slurm 회계 기록에서 153213의 최종 상태 `CANCELLED` |

## 1. 작업 출력 (`runs/gpu/logs/smoke_153212.out`)

```
입력 확인 통과: 13개
sbatch: MOANA: Job submitted
제출: smoke 작업 153213 (사슬 2번, 로그 runs/gpu/logs/smoke_153213.out)
=== 2026-11-05 15:59:09 다음 작업 153213 예약 (사슬 2번, 이 작업이 끝난 뒤 시작)
=== 2026-11-05 15:59:09 smoke 시도 1 시작 (체크포인트 step 0)
{"type": "baseline", "imitation": 1.356114046804607, "top1": 0.656, "top3": 0.896, "regret": 0.011584872612496838, "regret_top1": 0.0, "band_violation": 0.2269073176048696, "strength": 0.005139145421679132}
{"type": "train", "step": 20, ..., "loss": 1.2371781526017003, "positions_per_sec": 10.35995453954618, "elapsed": 61.77633357048035, "max_memory_gib": 4.571650505065918}
{"type": "eval", "step": 25, "imitation": 1.35495922617428, "top1": 0.654, "top3": 0.896, "regret": 0.011566279602469877, ...}
{"type": "train", "step": 40, ..., "loss": 1.8872258067131042, "positions_per_sec": 13.995855938556172, "elapsed": 210.92756366729736, "max_memory_gib": 4.571650505065918}
{"type": "train", "step": 50, ..., "loss": 1.358107779175043, "positions_per_sec": 10.82672652125282, "elapsed": 240.48405051231384, "max_memory_gib": 4.571650505065918}
{"type": "eval", "step": 50, "imitation": 1.3544610713003202, "top1": 0.654, "top3": 0.896, "regret": 0.011559966105152853, ...}
=== 2026-11-05 16:05:03 smoke 완료 (시도 1회, step 50)
=== 2026-11-05 16:05:03 다음 작업 153213 취소
```

(`...`는 지면상 줄였다. 전문은 묶음과 아래 `log.jsonl`에 있다.)

- `sbatch: MOANA: Job submitted`는 이 서버의 sbatch가 표준 오류로 내는 안내다. `submit.sh`는 그래도 작업 ID 153213을 바르게 읽었다.
- 첫 줄 앞에 `inputs.sha256` 주석 줄에 대한 "형식이 잘못된 줄" 경고가 없다. 단계1 발견 사항 4의 우려(오래된 coreutils)는 이 서버에서는 해당하지 않는다.

## 2. 작업 기록 (`runs/gpu/meta/smoke_153212.txt`)

```
학습 ID: smoke
시작: 2026-11-05 15:58:46
호스트: moana-u1
작업 ID: 153212  사슬 번호: 1 / 10  노드: moana-u1
앞 작업 시작 step: 
git HEAD: f39318a57d43f087be8181dd8c1ed371869a5975
git status --porcelain (비어 있으면 미커밋 변경 없음):
   M gpu_train/site.env.example
GPU: NVIDIA RTX A2000, 6138 MiB, 535.216.03
torch: 2.5.1+cu121 | CUDA 12.1 | GPU 보임 True
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
명령: python -u -m src.train.train --player tal --out runs/gpu/smoke --base base_model/136M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --max-steps 50 --eval-every 25 --eval-positions 500 --resume
시작 체크포인트 step: 0
=== 2026-11-05 15:59:09 다음 작업 153213 예약 (사슬 2번, 이 작업이 끝난 뒤 시작)
=== 2026-11-05 15:59:09 smoke 시도 1 시작 (체크포인트 step 0)
=== 2026-11-05 16:05:03 smoke 완료 (시도 1회, step 50)
=== 2026-11-05 16:05:03 다음 작업 153213 취소
```

## 3. 결과 요약 (`SUMMARY.txt`)

```
학습 ID: smoke
묶은 시각: 2026-11-09 11:55:19  호스트: moana-master
완료: 2026-11-05 16:05:03 작업 153212 (시도 1회, step 50)
adapter.pt: 4213206 bytes  SHA-256 9314b4fc5f7d097e4a2b281646b6a208bd189cc479f94d364e41fa798575f550
재개: 0회 []
```

baseline, 마지막 eval·train 줄은 1절과 같아서 생략한다. `adapter.pt`의 SHA-256은 로컬에서 다시 계산해도 같았다.

## 4. GPU 사용량 (`runs/gpu/logs/gpu_smoke.log`)

```
2026-11-05 15:59:09, 1 MiB, 0 %, 35
2026-11-05 16:00:09, 4958 MiB, 98 %, 55
2026-11-05 16:01:09, 4958 MiB, 100 %, 59
2026-11-05 16:02:09, 4958 MiB, 100 %, 61
2026-11-05 16:03:09, 4958 MiB, 100 %, 61
2026-11-05 16:04:09, 4958 MiB, 100 %, 62
```

## 5. 설정 (`runs/gpu/smoke/config.json` 중 주요 값)

`player tal`, `rank 16`, `lora_alpha 32.0`, `lr 0.0001`, `batch_positions 32`, `max_moves_per_forward 128`, `temperature 0.02`, `tau 0.02`, `alpha 0.0`, `lam 1.0`, `seed 1`, `max_steps 50`, `eval_every 25`, `eval_positions 500`, `amp true`, `eval_amp false`, `train_positions 79493`, `trainable_parameters 1048576`. 단계1에서 테스트로 고정한 smoke 설정과 같다.

## 6. 판정 (README 6절 기준)

| 항목 | 기대 | 서버 결과 | 판정 |
|---|---|---|---|
| 입력 확인 | `입력 확인 통과: 13개` | 같음 | ✅ |
| GPU | GPU 이름, `GPU 보임 True` | RTX A2000, True | ✅ |
| 완료 | `smoke 완료 (시도 1회, step 50)`, `DONE` | 같음 | ✅ |
| 다음 작업 | 예약 뒤 취소, sacct에 `CANCELLED` | 기록상 예약·취소 확인. sacct는 받지 못함 | ✅ (sacct 미확인) |
| baseline top1·top3 | 0.656, 0.896 정확히 | 0.656, 0.896 | ✅ |
| baseline 소수 값 | 소수 넷째 자리까지 같음 | 아래 표 | ✅ |
| 메모리 | 로컬 `max_memory_gib` 4.57, GPU 전체 6,270 MiB | 4.57 GiB, GPU 전체 4,958 MiB | ✅ |

baseline 비교:

| 항목 | 로컬 (RTX 4070 Ti, torch 2.14.1+cu130) | 서버 (RTX A2000, torch 2.5.1+cu121) | 차이 |
|---|---|---|---|
| imitation | 1.3561140506193041 | 1.356114046804607 | 3.8 × 10⁻⁹ |
| top1 | 0.656 | 0.656 | 0 |
| top3 | 0.896 | 0.896 | 0 |
| regret | 0.011584872650681064 | 0.011584872612496838 | 3.8 × 10⁻¹¹ |
| band_violation | 0.2269073169492185 | 0.2269073176048696 | 6.6 × 10⁻¹⁰ |
| strength | 0.005139145441353321 | 0.005139145421679132 | 2.0 × 10⁻¹¹ |

차이는 모두 10⁻⁸ 이하로, README의 허용 기준(소수 넷째 자리)보다 훨씬 작다. 서버의 입력 파일과 평가 코드가 로컬과 같다고 판단한다.

## 7. 처리량과 시간

| 항목 | 로컬 smoke | 서버 smoke |
|---|---|---|
| `positions_per_sec` | 약 25 | 10.4 (step 20), 14.0 (step 40), 10.8 (step 50) |
| 50 갱신 학습 시간 (`elapsed`) | — | 240 s |
| 작업 전체 (시도 시작 ~ 완료) | 2분 1초 | 5분 54초 |
| GPU 전체 최대 | 6,270 MiB (12 GB 카드) | 4,958 MiB (6 GB 카드, 여유 약 1.2 GB) |

- 서버 처리량은 로컬의 약 0.4~0.6배다. smoke는 50 갱신뿐이라 처음 몇 갱신의 준비 시간이 섞여 있다. 그래서 본 학습 시간은 이 값만으로 정확히 정하지 않는다.
- 단순 비율로 추정하면 학습 1회는 약 7~9시간이다(로컬 3.1~3.6시간, 추정). 실제 시간은 R1·R2·R8 결과에서 확인한다(후속 이슈).
- GPU 전체 사용량이 로컬(6,270 MiB)보다 작은 것은 GPU·드라이버·torch 버전에 따라 캐시 할당이 다르기 때문으로 본다. torch가 실제로 쓴 최대(`max_memory_gib` 4.57)는 로컬과 같다.

## 짚어 둘 점

1. **본 학습은 6 GB GPU에서 돌아간다.** torch 최대 4.57 GiB, GPU 전체 4,958 MiB로 6,138 MiB 안에 든다. 단, smoke는 val 500국면만 평가했다. 본 학습의 평가(val 전체)에서 메모리가 더 필요한지는 R1 결과에서 확인한다. README 10절의 `CUDA out of memory` 항목이 대응 방법이다.
2. **서버 저장소의 `gpu_train/site.env.example`이 수정된 상태다.** 학습 결과에는 영향이 없다. 하지만 예시 파일이 바뀌면 이후 `git pull`에서 충돌할 수 있고, 작업 기록에 미커밋 변경으로 계속 남는다. 서버 값은 `site.env`(git 제외)에 두고 예시 파일은 되돌리기를 권한다: `cp gpu_train/site.env.example gpu_train/site.env`(아직 없으면) → `git checkout gpu_train/site.env.example`.
3. 어댑터 SHA-256(`9314b4fc…f550`)은 로컬(`806fc7ff…633f`)과 다르다. GPU·torch가 달라 예상된 일이며 판정 대상이 아니다 (README 6절).
