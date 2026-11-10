# Task #11: α = 0 학습 R1·R2·R8 (GPU 서버) — 구현 계획서

> **상태 (2026-11-10)**: 재수정 승인 (seed 0 대조군을 GPU 서버에서 새로 학습, 5단계. 결정 5건 권장안)
> 수행 계획서: `mydocs/plans/task_m100_11.md` (승인. 범위를 이번 재수정에 맞춰 고침)

## 변경 이력

| 날짜 | 내용 | 이유 |
|---|---|---|
| 2026-11-09 | 최초 작성 (3단계, 로컬 재평가) | |
| 2026-11-10 | 승인. 결정 1 권장안(로컬 Docker + `fetch_inputs.sh`) | 작업지시자 승인 |
| 2026-11-10 | 재평가를 GPU 서버에서 sbatch로 하도록 수정 (4단계) | 이 작업 사본의 PC(Ryzen 7 7840U, Radeon 780M)에 NVIDIA GPU가 없다. RTX 4070 Ti PC에서도 재평가하지 않기로 함 (작업지시자 결정) |
| 2026-11-10 | 평가 대상을 R1·R2·R8로 한정 | 작업지시자 결정 |
| 2026-11-10 | **seed 0 대조군을 GPU 서버에서 현재 코드로 새로 학습**하도록 재수정 (5단계). 학습 ID `R1s0`·`R2s0`·`R8s0`을 추가하고, 기준 문서의 잡음 바닥 쌍을 이 학습으로 바꾸는 정정을 넣음 | 작업지시자 결정. 기존 seed 0 어댑터는 Task #5 수정 전 코드로 로컬에서 학습한 것이라, seed 쌍에 코드 차이와 장비 차이가 섞인다 |
| 2026-11-10 | 재수정 승인, 결정 5건 권장안 | 작업지시자 승인 |

## 왜 바꾸나

기준 문서 4.3절 B3는 같은 기사 쌍을 (기존 seed 0, R seed 1)로 둔다. 기존 seed 0 어댑터 `runs/{player}_136M/`에는 두 가지 차이가 섞여 있다.

- Task #5 결함 차이: 데이터 순서가 다르고, 국면 0.83~12.45%의 학습 횟수가 다르다 (기준 문서 4.3절 B3 주석, 6절 D8 주석)
- 장비 차이: 로컬 RTX 4070 Ti, torch 2.14에서 학습했다. R1·R2·R8은 서버 RTX A2000, torch 2.5.1이다

서버에서 현재 코드(`devel`)로 seed 0을 다시 학습하면, 쌍 사이의 차이가 **seed 하나뿐**이 된다. 잡음 바닥이 과대 추정되지 않는다.

## 공통

- `src/`는 바꾸지 않는다. 새 코드는 `gpu_train/` 아래의 학습 정의 추가, 평가 도구, 그 테스트뿐이다.
- 어댑터와 기록은 git 제외 경로(`runs/`)에 둔다. 기존 seed 0 폴더 `runs/{tal,smyslov,karpov}_136M/`은 읽기만 한다. 대국 앱(`src/play`)은 이 폴더를 계속 쓴다.
- 단계마다 `mydocs/working/task_m100_11_stage{N}.md`를 쓰고, 승인 뒤 그 단계 소스와 함께 `local/task11`에 커밋한다.
- 로컬 테스트는 개발 컨테이너 이미지(`chess-transformer-dev`)를 **GPU 없이** `docker run --rm`으로 돌린다. 이 PC에는 NVIDIA GPU가 없다.

## 서버에서 할 일 (전체)

| 작업 | 대상 | 예상 시간 | 비고 |
|---|---|---|---|
| seed 0 학습 | `R1s0`(탈), `R2s0`(스미슬로프), `R8s0`(카르포프) | 약 27시간 (R1·R2·R8 실측 7.9 + 9.3 + 10.1시간) | 차례로 또는 동시에 제출. GPU 1개씩 |
| 재평가 | `runs/gpu/R1`, `R2`, `R8` | 약 45분 | 받은 어댑터가 학습 끝의 가중치 그대로인지 확인 |

## 1단계: 배치와 무결성 확인 — 완료 (승인, `3ada703`)

변경 없음.

## 2단계: 학습 정의 추가와 서버 평가 도구

### 구현

1. **`gpu_train/runs.py`: 학습 ID 3개 추가 (결정 1)**
   ```python
   Run("R1s0", "tal", 0.0, seed=0, role="α = 0, seed 0. 잡음 바닥 쌍 (R1과 seed만 다름)"),
   Run("R2s0", "smyslov", 0.0, seed=0, role="…(R2와 seed만 다름)"),
   Run("R8s0", "karpov", 0.0, seed=0, role="…(R8과 seed만 다름)"),
   ```
   - 출력 폴더는 `runs/gpu/R1s0` 등이다. 다른 설정은 R1·R2·R8과 같다
   - `submit.sh`, `train.sbatch`, `collect.sh`는 ID만 받으므로 바꾸지 않는다
2. **`gpu_train/eval_adapter.py`** (작성함, 미커밋). 학습 폴더 하나를 다시 평가해 마지막 eval·baseline과 비교한다
3. **`gpu_train/eval.sbatch`** (신규). 여러 학습 폴더를 차례로 평가하는 작업 본문
   ```
   bash gpu_train/eval.sbatch runs/gpu/R1 runs/gpu/R2 runs/gpu/R8
   ```
   - `train.sbatch`와 같은 준비를 한다: `site.env`, conda 활성화, `check_inputs.sh`
   - 기록 파일 `runs/gpu/eval/meta_{작업ID}.txt`에 남길 것: 시작 시각, 호스트, 작업 ID, git HEAD와 미커밋 변경, GPU, torch, 대상 목록
   - 폴더마다 `eval_adapter --baseline --json runs/gpu/eval/{이름}.json`을 실행하고, 출력을 `runs/gpu/eval/{이름}.txt`에 남긴다
   - 하나가 실패해도 나머지는 평가한다. 끝에 폴더별 판정 표를 기록 파일에 남긴다
   - 종료 코드: 0 모두 통과, 1 하나라도 실패·오류, 2 입력 확인 실패
   - GPU 사용량을 60초마다 `runs/gpu/eval/gpu_{작업ID}.log`에 남긴다
   - 이어서 제출과 재시도는 하지 않는다
4. **`gpu_train/submit_eval.sh`** (신규, 결정 4). `site.env`의 Slurm 값으로 `eval.sbatch`를 제출한다
   - 옵션 방식은 `submit.sh`와 같고 `SBATCH_DRY_RUN=1`을 지원한다
   - 작업 이름은 `eval`, 출력은 `runs/gpu/logs/eval_%j.out`이다
   - 제출 전에 각 폴더에 `config.json`·`log.jsonl`·`adapter.pt`가 있는지 확인한다
5. **`gpu_train/README.md`**
   - 7절 본 학습에 `R1s0`·`R2s0`·`R8s0`을 추가한다
   - 새 절 "학습 결과 재평가"에 제출, 상태 확인, 결과 묶기, 판정 읽는 법을 적는다

### 테스트

- `tests/test_gpu_train.py`
  - `R1s0`·`R2s0`·`R8s0`의 명령이 대응하는 R1·R2·R8과 `seed`(0)·`out`만 다르다
  - 출력 폴더가 서로 겹치지 않는다(기존 테스트가 새 ID도 포함)
  - `list`에 "가능"으로 나온다
- `tests/test_eval_adapter.py` (작성함, 27개 통과)
- `eval.sbatch`, `submit_eval.sh` (pytest로 만들어 컨테이너에서 돌린다)
  - `bash -n` 구문 확인
  - `SBATCH_DRY_RUN=1`로 sbatch 명령 확인 (임시 `site.env`)
  - 소형 학습 폴더에 대해 `eval.sbatch`를 Slurm 밖에서 실행한다. `check_inputs.sh`는 테스트 전용 `EVAL_SKIP_INPUT_CHECK=1`로 건너뛴다. 통과하면 0, 바꿔치기한 어댑터가 섞이면 1이 나와야 한다
- 전체 pytest 통과, `git diff --stat -- src/` 비어 있음

## 3단계: 배포와 서버 실행

### 구현

1. **배포 (결정 3)**: 2단계 커밋을 `local/task11 → local/devel → devel`로 merge하고 `origin/devel`에 push한다. push는 작업지시자 승인 뒤에 한다
2. **실행 (작업지시자, 서버)**. 옮길 파일은 없다
   ```
   git pull                                   # devel
   python -m gpu_train.runs list              # R1s0·R2s0·R8s0이 "가능"
   bash gpu_train/submit.sh R1s0              # R2s0, R8s0도 같은 방식
   bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8
   # 끝나면
   bash gpu_train/collect.sh R1s0             # R2s0, R8s0도
   tar -czf runs/gpu/collect/eval_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/eval runs/gpu/logs/eval_*.out
   ```
   - 재평가는 학습과 GPU를 다투지 않게, seed 0 학습과 동시에 돌려도 되는지 서버 GPU 할당에 따라 작업지시자가 정한다
3. 이 단계의 보고서에는 배포 커밋과 제출 안내만 적는다. 학습이 끝나기를 기다리는 동안 다른 단계는 진행하지 않는다

## 4단계: 결과 회수와 확인

### 구현

1. 받은 묶음을 `runs/gpu/collect/`에 두고, 1단계와 같은 방식으로 `runs/gpu/{R1s0,R2s0,R8s0}/`과 `runs/gpu/eval/`에 배치한다
2. **seed 0 학습 확인** (1단계 스크립트 확장)
   - SHA-256 = `SUMMARY.txt`, `DONE` 있음, 마지막 eval step = `total_steps`
   - **`config.json`이 대응하는 R1·R2·R8과 `seed`, `out`만 다르다**
   - baseline이 R1·R2·R8과 같다 (top1·top3 정확히, 소수 값 10⁻⁶ 미만)
   - 작업 기록의 git HEAD가 push한 커밋이다. R1·R2·R8(`f39318a`)과 다른 커밋이므로 `src/`에 차이가 없는지 `git diff f39318a <배포 커밋> -- src/`로 확인한다
3. **재평가 대조**: R1·R2·R8의 판정이 통과인지, 결과 JSON의 SHA-256이 1단계 값과 같은지 본다

### 검증

- seed 0 세 학습이 위 확인을 모두 통과한다
- R1·R2·R8 재평가가 마지막 eval과 baseline 모두 통과한다
- 실패하면 차이를 기록하고 멈춰 보고한다

## 5단계: 결과 정리와 기준 문서 정정

### 구현

1. **`mydocs/test_result/task_m100_11_test.md`**
   - 1~7절(R1·R2·R8)은 작성함
   - seed 0 학습 3회를 같은 항목으로 추가한다
   - 8절에 재평가 결과를 넣는다
   - seed 쌍의 마지막 val 지표를 기술 통계로 나란히 둔다 (판정은 후속 이슈)
2. **기준 문서 `evaluation_criteria.md` 정정** (2.7절 절차, 결정 2). α > 0 결과가 나오기 전이므로 2.7절과 충돌하지 않는다
   - **8.1절**: 학습 표에 `R1s0`·`R2s0`·`R8s0` 3행을 추가한다. 머리말의 "추가 학습은 10회"를 13회로 고친다
   - **8.1·8.4절 시간**: 서버 실측 기준으로 고친다. 13회 약 117시간이고, 그중 남은 7회(R3~R7, R9, R10)가 약 63시간이다(스타일 항 비용 제외). 미리 작성해 둔 "10회 약 90시간" 정정을 이 값으로 바꾼다
   - **4.3절 B3 같은 기사 쌍**: "(기존 seed 0, R1·R8·R2 seed 1)"을 "(R1s0, R1)·(R2s0, R2)·(R8s0, R8)"로 고친다. Task #5 결함 차이가 섞인다는 주석에는 정정 표시를 붙인다
   - **6절 D8**: seed 차이 "(기존 seed 0, 새 seed 1)"을 같은 쌍으로 고치고, 결함 차이 주석에 정정 표시를 붙인다
   - **5절 비교군 문장** "기존 어댑터는 B3의 잡음 바닥 쌍에만 쓴다": 기존 어댑터는 판정에 쓰지 않는다로 고친다
   - **1절 용어 "어댑터"**: seed 0 학습 ID를 덧붙인다
   - 0절 정정 이력에 위 내용을 한 줄로 적는다. 판정 조건 값은 바꾸지 않는다
3. **`gpu_train/README.md` 7절 서버 실측 한 줄**: 작성함

### 검증

- test 문서의 수치가 `SUMMARY.txt`·`log.jsonl`·평가 JSON과 같다 (스크립트 출력 대조)
- 기준 문서 정정 표시와 0절 이력이 2.7절 형식을 따른다. "기존 seed 0"을 가리키는 판정 문장이 남지 않는다(grep 확인)

## 예상 단계: 5단계

| 단계 | 산출물 | 커밋 |
|---|---|---|
| 1 | 1단계 보고서 | 완료 (`3ada703`) |
| 2 | `runs.py` 학습 ID, `eval_adapter.py`, `eval.sbatch`, `submit_eval.sh`, 테스트, README, 2단계 보고서 | 소스 + 보고서 |
| 3 | devel push, 서버 실행 안내, 3단계 보고서 | 보고서 |
| 4 | seed 0 학습과 재평가 결과 확인, 4단계 보고서 | 보고서 |
| 5 | `task_m100_11_test.md`, 기준 문서 정정, README 한 줄, 5단계 보고서 | 문서 |

## 결정 요청

1. ~~학습 ID~~: `R1s0`·`R2s0`·`R8s0` (권장안 승인, 2026-11-10). **학습 ID** (권장: `R1s0`·`R2s0`·`R8s0`)
   - 대응하는 R와 seed만 다르다는 것이 이름에 드러난다. R1~R10의 번호 체계(기준 문서 8.1절)는 건드리지 않는다
   - 다른 안: `R11`·`R12`·`R13`으로 이어 붙인다
2. ~~잡음 바닥 쌍 정정~~: 정정한다 (권장안 승인, 2026-11-10). **기준 문서의 잡음 바닥 쌍 정정** (권장: 정정한다)
   - B3와 D8의 같은 기사 쌍을 서버 seed 0 학습으로 바꾸고, 기존 seed 0 어댑터는 판정에 쓰지 않는다
   - 다른 안: 기준은 그대로 두고, 서버 seed 0 쌍을 보조 결과로만 보고한다. 이 경우 서버 학습 약 27시간을 쓰고도 판정에는 반영되지 않는다
3. ~~배포 방식~~: devel merge와 push (권장안 승인, 2026-11-10). **배포 방식** (권장: 2단계 승인 뒤 `local/devel → devel` merge와 push)
   - 서버는 git으로 코드를 받는다. 타스크 중간에 push하는 것은 Task #10과 같다
   - seed 0 학습의 작업 기록이 git HEAD로 코드를 특정하려면 커밋된 코드여야 한다
4. ~~재평가 제출 스크립트~~: `submit_eval.sh` 신규 (권장안 승인, 2026-11-10). **재평가 제출 스크립트** (권장: `submit_eval.sh`를 새로 만든다)
   - 서버에서 검증한 `submit.sh`를 건드리지 않는다. 대가로 Slurm 옵션을 붙이는 약 10줄이 겹친다
   - 다른 안: 공통 부분을 빼고 `submit.sh`도 그것을 쓰게 고친다
5. ~~이슈 범위~~: #11 안에서 한다 (권장안 승인, 2026-11-10). **이슈 범위** (권장: #11 안에서 한다)
   - α = 0 학습이고 절차와 도구가 같다. #11 본문에 seed 0 학습 추가를 적는다(GitHub 이슈 수정은 승인 뒤)
   - 다른 안: 별도 이슈로 뺀다. 이 경우 #11은 재평가까지만 하고 닫는다
