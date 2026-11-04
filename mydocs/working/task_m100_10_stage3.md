# Task #10 단계3 완료 보고서: 작업 스크립트

## 수행 내용

- **`gpu_train/submit.sh RUN`**: 제출 스크립트.
  - `site.env`의 Slurm 값으로 `sbatch` 옵션을 만든다. 빈 값은 옵션을 붙이지 않는다. `SBATCH_EXTRA`는 공백으로 나눠 붙인다.
  - 공통 옵션: `--job-name=RUN`, `--output=runs/gpu/logs/%x_%j.out`, `--chdir=$REPO_DIR`, `--parsable`, `--export=ALL,CHAIN_INDEX=…,PREV_START=…`
  - 거부하는 경우(종료 코드 1):
    - `runs.py args`가 실패하는 ID(없는 ID, α 미정)
    - `REPO_DIR`가 비었거나 이 저장소와 다른 경로
    - 같은 이름의 작업이 이미 대기 중이거나 실행 중(`squeue`)
  - `SBATCH_DRY_RUN=1`이면 실행할 `sbatch` 명령만 출력한다.
  - `train.sbatch`가 다음 작업을 이어서 제출할 때도 이 스크립트를 쓴다. 이때는 `SUBMIT_AFTER`(앞 작업 ID), `CHAIN_INDEX`, `PREV_START`가 붙는다. `SUBMIT_AFTER`가 있으면 `--dependency=afterany:…`를 붙이고 같은 이름 확인은 건너뛴다(앞 작업이 같은 이름으로 실행 중이므로).
  - 제출한 작업 ID를 표준 출력에 한 줄로 낸다. 안내 문구는 표준 오류로 낸다.
- **`gpu_train/train.sbatch RUN`**: 작업 본문. 구현 계획서 3단계의 1~7을 그대로 구현했다.
  1. 준비: `site.env`(Slurm 안에서는 필수), conda 활성화, `check_inputs.sh`(실패하면 종료 코드 2)
  2. `DONE`이 있으면 바로 끝낸다
  3. Slurm 안에서만:
     - 전진 확인: 사슬 2번 이후에는 지금 체크포인트 step이 `PREV_START`보다 커야 한다. 아니면 다음 작업을 예약하지 않고 끝낸다.
     - 그다음 `CHAIN_MAX` 미만이면 다음 작업을 예약한다.
  4. 기록 파일 `runs/gpu/meta/{RUN}_{작업ID}.txt`
  5. GPU 사용량 60초 간격, 기존 `train_player.sh`와 같은 형식
  6. 재시도: 최대 10회, 체크포인트가 전진하지 않은 채 연속 3회 실패하면 멈춘다, 20초 간격. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  7. 완료하면 `DONE`을 쓰고 다음 작업을 `scancel`한다
  - **계획에 더한 것: 실패로 멈출 때도 다음 작업을 취소한다.** 연속 3회 실패나 10회 소진으로 멈추면 다음 작업을 이어 가도 같은 실패를 되풀이할 뿐이기 때문이다.
  - **계획에 더한 것: 종료 신호 처리.** 시간 제한에 걸리면 Slurm이 작업의 모든 프로세스에 SIGTERM을 보낸다. 이 신호를 받으면 재시도하지 않고 끝내고, 예약한 다음 작업은 그대로 둔다. 재시도하면 곧 SIGKILL로 끊길 시도를 하나 더 시작하게 된다.
  - Slurm 밖(`SLURM_JOB_ID` 없음)에서는 3과 `scancel`을 건너뛰고, 작업 ID 자리에 `local_{시각}`을 쓴다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/submit.sh` | 신규 |
| `gpu_train/train.sbatch` | 신규 |
| `mydocs/working/task_m100_10_stage3.md` | 이 보고서 |

`src/` 변경은 없다. 두 파일 모두 LF로 고정됨을 `git check-attr eol`로 확인했다.

## 검증

로컬 RTX 4070 Ti에서 Slurm 밖 실행으로 확인했다. Slurm 고유 동작은 가짜 `sbatch`·`squeue`·`scancel`(호출 인자만 기록)과 `SLURM_JOB_ID`를 붙여 확인했다.

### 제출 스크립트

| 경우 | 결과 |
|---|---|
| `SBATCH_DRY_RUN=1 … R1` (계정·QoS 빈칸, `SBATCH_EXTRA="--constraint=a100 --nice=10"`) | `--account`·`--qos`는 빠지고 나머지 옵션과 추가 옵션 2개가 붙음. 종료 코드 0 |
| 이어서 제출 형태 (`SUBMIT_AFTER=12345 CHAIN_INDEX=2 PREV_START=1400`) | `--dependency=afterany:12345`, `CHAIN_INDEX=2`, `PREV_START=1400` |
| R3 (α 미정), R99 (없는 ID), 인자 없음, `REPO_DIR` 빈칸 | 모두 종료 코드 1, 이유 출력 |
| `REPO_DIR`가 다른 경로 | 종료 코드 1 |
| 가짜 `sbatch`로 실제 제출 | `777;cluster` 응답에서 `777`만 출력. 종료 코드 0 |
| 같은 이름 작업 실행 중 (가짜 `squeue`) | 제출하지 않음, 종료 코드 1 |
| 같은 상황에서 이어서 제출 | 이름 확인을 건너뛰고 제출 |

### 로컬 smoke (`bash gpu_train/train.sbatch smoke`)

- 걸린 시간 2분 1초. 과정: 입력 확인 13개 통과 → baseline → 50 갱신(평가는 step 25와 50) → `DONE`. 종료 코드 0
- 만들어진 것:
  - `runs/gpu/smoke/{adapter.pt, checkpoint.pt, config.json, log.jsonl, DONE}`
  - 기록 파일: 호스트, git HEAD와 미커밋 파일 2개, GPU `RTX 4070 Ti, 12282 MiB, 621.01`, torch `2.14.1+cu130`, 명령, 시작 step
  - GPU 로그: 최대 6,270 MiB, 사용률 98%
- 같은 명령을 다시 실행하면 "이미 완료"를 출력하고 종료 코드 0으로 끝난다.
- **기준값 (서버 smoke와 비교한다)**. `log.jsonl`의 baseline 줄은 베이스 모델로 val 앞쪽 500국면을 평가한 값이다:

  | 항목 | 값 |
  |---|---|
  | imitation | 1.3561140506193041 |
  | top1 | 0.656 |
  | top3 | 0.896 |
  | regret | 0.011584872650681064 |
  | band_violation | 0.2269073169492185 |
  | strength | 0.005139145441353321 |

  - top1·top3는 개수 비율이라 같아야 한다.
  - 소수 값은 GPU 종류에 따라 부동소수점 끝자리가 다를 수 있다. README에는 "소수 넷째 자리까지 같으면 정상"으로 적는다.
  - 어댑터 SHA-256 `806fc7ff…633f`는 같은 GPU에서만 같다고 본다. 서버에서는 참고로만 적는다.

### 강제 종료 뒤 재시도

- smoke 도중 step 40 기록이 나온 뒤 학습 프로세스를 SIGKILL로 끝냈다. 그 시점 체크포인트는 step 25였다.
  - 시도 1: 실패(종료 코드 137, 체크포인트 step 25)
  - 20초 뒤 시도 2: step 25에서 이어 감(`log.jsonl`에 `resume` 25) → 완료(시도 2회, step 50)
- **끊김 없이 돈 첫 smoke와 어댑터 SHA-256이 같았다** (`806fc7ff6923b7a5fa590252100898af16c731d133e3be2b8df2a0a7b1fe633f`). 시간 항목과 `resume` 줄을 뺀 `log.jsonl`도 바이트 단위로 같았다.

### Slurm 동작 (가짜 명령)

| 경우 | 결과 |
|---|---|
| (a) `DONE` 있음 | 바로 끝냄, 종료 코드 0, 제출 없음 |
| (b) 사슬 2번, `PREV_START=50`, 지금 step 50 | "체크포인트가 전진하지 않았다", 다음 작업 예약 없이 종료 코드 1 |
| (c) 사슬 10번 (= `CHAIN_MAX`) | "사슬 상한" 기록, 예약 없이 학습 후 완료 |
| (d) 사슬 1번 | 다음 작업 777 예약(`--dependency=afterany:900`, `CHAIN_INDEX=2`, `PREV_START=50`) → 완료 → `scancel 777` |
| (e) 학습 중 작업 프로세스 그룹 전체에 SIGTERM (시간 제한 흉내) | "종료 신호를 받았다 (종료 코드 143)", 재시도 없음, `scancel` 호출 없음(다음 작업 777이 이어 감), GPU 감시 프로세스까지 모두 종료 |

- 전체 pytest: **159 passed** (기존과 같음)

## 발견 사항

1. **이 컨테이너에는 `USER` 변수가 없어서 `submit.sh`가 `set -u`에 걸렸다.** `${USER:-$(id -un)}`로 고쳤다. 서버 로그인 셸에는 보통 있지만, 없는 경우에도 돌게 했다.
2. **`REPO_DIR` 불일치 확인을 더했다.** `submit.sh`는 실행한 위치에 로그 폴더를 만들고, 작업은 `REPO_DIR`에서 돈다. 둘이 다르면 로그와 결과가 엉뚱한 곳에 쌓인다. 실제 제출할 때만 확인하고, 미리 보기(`SBATCH_DRY_RUN=1`)에서는 확인하지 않는다.
3. **site.env의 `SBATCH_*` 이름은 sbatch가 읽는 환경변수 이름과 같다** (`SBATCH_PARTITION` 등). 하지만 site.env를 `source`만 하고 `export`하지 않으므로 sbatch에 직접 전달되지 않는다. submit.sh가 옵션으로 바꿔 붙인 값만 쓰인다. README에 "site.env 값을 export하지 않는다"고 적는다.
4. **검증 방법의 실수 두 번** (스크립트와 무관):
   - 강제 종료 시험에서 `pkill -f`의 패턴이 시험을 돌린 셸의 명령줄에도 들어 있어서 그 셸까지 끝났다. 작업은 계속 돌았다. 남은 과정은 프로세스 번호로 기다려 확인했다.
   - SIGTERM 시험에서 처음에 프로세스 그룹 번호를 잘못 잡아 신호가 가지 않았다. `ps`로 실제 그룹을 찾아 다시 보냈다.
5. 시험 때 만든 `gpu_train/site.env`는 지웠다. git 제외인 `runs/gpu/meta/`에는 시험 기록 파일 4개가 남아 있다(`smoke_local_*` 2개, 가짜 Slurm `smoke_900`·`smoke_901`).

## 서버에서 확인할 것 (서버 smoke)

- `#SBATCH` 헤더와 submit.sh 옵션이 실제 sbatch에서 받아들여지는지
- 계산 노드에서 conda 활성화가 되고 GPU가 보이는지
- 작업 안에서 다음 작업을 제출할 수 있는지(일부 서버는 막는다), `scancel`이 되는지
- baseline 값이 위 기준값과 소수 넷째 자리까지 같은지

## 다음 단계

4단계: 결과 회수와 문서.
- `collect.sh`: 결과 묶음과 `SUMMARY.txt`
- `gpu_train/README.md`
- `evaluation_criteria.md` 8.2절 정정 (Docker → Miniconda·Slurm)
- `training_guide.md`와 `README.md`에 링크 추가

## 커밋

`Task #10 단계3: 작업 스크립트` (승인 후 커밋)
