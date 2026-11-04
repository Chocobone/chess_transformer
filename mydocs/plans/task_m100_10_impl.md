# Task #10: GPU 서버 학습 디렉토리 `gpu_train/` 작성 — 구현 계획서

## 공통

- `src/`는 수정하지 않는다. 필요해지면 멈추고 따로 승인받는다.
- 새 파일은 `gpu_train/`과 `tests/test_gpu_train.py`에만 만든다. `.gitignore`에 서버 설정 파일 한 줄을 추가한다.
- 셸 스크립트는 bash로 쓰고 `set -euo pipefail`을 쓴다. 로컬에 shellcheck가 없으므로 `bash -n`으로 문법을 검사한다.
- 학습 산출물은 모두 `runs/gpu/` 아래에 둔다 (`runs/`는 git 제외).

**디렉토리 구성**

```
gpu_train/
├── __init__.py
├── runs.py              ← 학습 정의 R1~R10, smoke. 학습 명령 생성 (1단계)
├── inputs.sha256        ← 입력 파일 SHA-256 목록 (1단계, 커밋)
├── make_manifest.sh     ← 로컬에서 inputs.sha256을 만든다 (1단계)
├── check_inputs.sh      ← 서버에서 입력을 확인한다 (1단계)
├── environment.yml      ← conda 환경 (Python 3.13, torch 제외) (2단계)
├── setup_env.sh         ← env 생성, torch 설치, pytest (2단계)
├── site.env.example     ← 서버 설정 빈칸과 설명 (2단계, 커밋)
├── train.sbatch         ← 작업 스크립트. Slurm 밖에서도 실행 가능 (3단계)
├── submit.sh            ← site.env 값으로 sbatch 옵션을 붙여 제출 (3단계)
├── collect.sh           ← 결과 묶음 생성 (4단계)
├── README.md            ← 매뉴얼 (4단계, 5단계에서 3절 갱신)
├── inputs.hf            ← 입력을 받을 Hugging Face 저장소와 고정 리비전 (5단계, 커밋)
├── upload_inputs.sh     ← 로컬에서 입력을 Hugging Face 비공개 저장소에 올린다 (5단계)
└── fetch_inputs.sh      ← 서버에서 고정 리비전의 입력을 받고 확인한다 (5단계)

runs/gpu/                ← 서버·로컬 산출물 (git 제외)
├── {RUN}/               ← train.py --out (adapter.pt, checkpoint.pt, config.json, log.jsonl)
├── logs/                ← Slurm 출력 {RUN}_{작업ID}.out, GPU 사용량 gpu_{RUN}.log
├── meta/                ← 작업별 기록 {RUN}_{작업ID}.txt
└── collect/             ← 결과 묶음 {RUN}_{날짜}.tar.gz
```

---

## 1단계: 학습 정의와 입력 매니페스트

### 목표

R1~R10과 smoke의 학습 명령을 한곳에서 만들고 테스트한다. 서버로 옮길 입력의 SHA-256 목록과 확인 스크립트를 만든다.

### 구현

**`gpu_train/runs.py`** — 학습 정의와 명령 생성:

```python
@dataclass(frozen=True)
class Run:
    run_id: str
    player: str              # tal | karpov | smyslov
    alpha: float | None      # None이면 특성 구현 전이라 실행할 수 없다
    lam: float = 1.0
    seed: int = 1
    extra: tuple[str, ...] = ()   # smoke 전용 인자

COMMON = {  # 기준 문서 8.1절. train.py 기본값에 기대지 않고 모두 명시한다
    "--base": "base_model/136M.pt", "--rank": "16", "--lora-alpha": "32",
    "--lr": "1e-4", "--epochs": "3", "--batch-positions": "32",
    "--temperature": "0.02", "--tau": "0.02", "--gamma": "0",
    "--eval-every": "1000", "--checkpoint-every": "200",
}
RUNS = {
    "R1": Run("R1", "tal", 0.0), "R2": Run("R2", "smyslov", 0.0), "R8": Run("R8", "karpov", 0.0),
    "R3": Run("R3", "tal", None), "R4": Run("R4", "tal", None), ...   # a₁, a₂
    "R7": Run("R7", "tal", None, lam=0.0),
    "smoke": Run("smoke", "tal", 0.0, extra=("--max-steps", "50", "--eval-every", "25", "--eval-positions", "500")),
}

def out_dir(run: Run) -> str: ...            # runs/gpu/{run_id}
def train_args(run_id: str) -> list[str]: ... # python -u -m src.train.train ... --resume. alpha가 None이면 오류
```

명령줄 사용:

```bash
python -m gpu_train.runs list            # ID, 기사, α, λ, seed, 실행 가능 여부
python -m gpu_train.runs args R1         # 학습 명령 (작업 스크립트가 쓴다)
python -m gpu_train.runs out R1          # 출력 폴더
```

- R3~R7, R9, R10은 α가 정해지지 않아 `args`에서 거부된다. 오류 메시지에 "특성 구현 이슈에서 α를 정한 뒤 이 파일에 적는다"를 넣는다.
- `--resume`은 항상 붙인다. 처음 실행이면 처음부터 시작하고, 끝난 학습이면 갱신 없이 끝난다.

**`tests/test_gpu_train.py`** — 테스트:

- R1·R2·R8의 인자를 `src.train.train.build_parser()`로 파싱했을 때 값이 8.1절과 같다 (rank 16, lora_alpha 32, lr 1e-4, epochs 3, batch_positions 32, temperature 0.02, tau 0.02, lam 1, alpha 0, gamma 0, seed 1, resume True, base `136M.pt`)
- R1·R2·R8의 player가 tal, smyslov, karpov다
- α가 없는 ID는 `train_args`가 거부한다. R7은 lam 0이다
- 출력 폴더가 ID마다 다르다
- **기존 어댑터와의 설정 대조**: `runs/{player}_136M/config.json`이 있으면 seed, out, 그리고 데이터에 따라 정해지는 값을 뺀 학습 설정이 같은지 본다. 없으면 건너뛴다 (`runs/`는 git 제외)

**`gpu_train/make_manifest.sh`**, **`gpu_train/inputs.sha256`** — 입력 목록:

- 대상: `base_model/136M.pt`, `data/processed/{tal,karpov,smyslov}/{train.jsonl,val.jsonl,stats.json,base_q_136M.npz}` (13개)
- 형식: `sha256sum` 출력 그대로 (저장소 루트 기준 상대 경로). 크기는 주석 줄로 남긴다
- 로컬에서 한 번 만들어 커밋한다

**`gpu_train/check_inputs.sh`** — 입력 확인:

- 저장소 루트에서 `sha256sum -c gpu_train/inputs.sha256`을 실행한다. 하나라도 없거나 다르면 종료 코드 1이다
- 작업 스크립트(3단계)가 학습 전에 부른다

### 검증

- `pytest tests/test_gpu_train.py` 통과, 전체 pytest 통과
- 로컬에서 `check_inputs.sh` 통과. 파일 하나를 임시 사본에서 1바이트 바꾸면 실패하는지 스크래치패드에서 확인한다 (원본은 건드리지 않는다)
- `bash -n` 통과

---

## 2단계: 환경 구성

### 목표

서버 로그인 노드에서 명령 하나로 conda 환경을 만들고 확인할 수 있게 한다. 서버 고유 값을 채울 설정 파일 예시를 만든다.

### 구현

**`gpu_train/environment.yml`**:

```yaml
name: chess-train          # site.env의 CONDA_ENV로 바꿀 수 있다
channels: [conda-forge]
dependencies:
  - python=3.13
  - pip
  - pip:
      - -r ../requirements.txt     # torch는 setup_env.sh가 따로 설치한다
```

- `requirements.txt`의 torch 줄은 setup_env.sh가 먼저 설치한 torch로 만족된다. 대국 앱 의존성도 함께 설치한다. pytest 전체(대국 앱 테스트 포함)가 돌아야 하기 때문이다.

**`gpu_train/site.env.example`** (커밋), **`gpu_train/site.env`** (git 제외, 작업지시자가 채운다):

```bash
# ── 경로 ──
REPO_DIR=          # 서버의 저장소 경로. 계산 노드에서 보여야 한다
CONDA_SH=          # conda.sh 경로 (예: ~/miniconda3/etc/profile.d/conda.sh)
CONDA_ENV=chess-train
# ── torch ──
TORCH_INDEX_URL=   # 비우면 PyPI 기본 휠. CUDA 버전을 맞출 때 https://download.pytorch.org/whl/cuXXX
# ── Slurm (비우면 그 옵션을 붙이지 않는다) ──
SBATCH_PARTITION=
SBATCH_ACCOUNT=
SBATCH_QOS=
SBATCH_GRES=gpu:1
SBATCH_TIME=       # 작업 최대 시간 (예: 04:00:00)
SBATCH_CPUS=4
SBATCH_MEM=32G
# ── 이어서 제출 ──
CHAIN_MAX=10       # 한 학습에서 이어서 제출하는 작업 수의 상한
```

- 항목마다 확인하는 명령을 주석으로 단다 (`sinfo -s`, `sacctmgr show assoc user=$USER`, `scontrol show partition`, `nvidia-smi`).

**`gpu_train/setup_env.sh`**:

1. `site.env`를 읽고, `CONDA_SH`를 source한다
2. 환경이 없으면 `conda env create -f gpu_train/environment.yml -n $CONDA_ENV`를 실행한다. torch를 pip 단계에서 먼저 깔아야 하므로 순서는 다음과 같다: python·pip만 있는 env를 만든다 → torch를 설치한다(`TORCH_INDEX_URL`이 있으면 `--index-url`) → `pip install -r requirements.txt`
3. 확인: `python -c "import torch, chess; print(torch.__version__, torch.version.cuda)"`, `pytest -q`
4. 설치한 패키지 목록(`pip freeze`)을 `runs/gpu/meta/env_{날짜}.txt`에 남긴다

- 로그인 노드에 GPU가 없을 수 있으므로 `torch.cuda.is_available()`은 여기서 요구하지 않는다. GPU 확인은 smoke 작업(3단계)이 한다.

### 검증

- `bash -n` 통과
- 로컬에 conda가 없으므로 setup_env.sh 실행은 로컬에서 확인하지 못한다. 대신 같은 순서를 로컬 Python 3.13 venv(스크래치패드)에서 따라 해 `requirements.txt`가 설치되고 pytest가 도는지 확인한다
- `site.env`가 `.gitignore`에 들어갔는지 `git check-ignore`로 확인한다

---

## 3단계: 작업 스크립트

### 목표

학습 ID 하나로 제출하면, 입력 확인 → 학습 → 실패하면 재시도 → 시간 제한에 걸리면 다음 작업으로 이어서 학습 → 기록까지 사람 손 없이 돌게 한다.

### 구현

**`gpu_train/submit.sh RUN`** — 제출:

- `site.env`의 값으로 `sbatch` 옵션을 만든다. 빈 값은 옵션을 붙이지 않는다
- 공통 옵션: `--job-name=$RUN`, `--output=runs/gpu/logs/%x_%j.out`, `--chdir=$REPO_DIR`
- `python -m gpu_train.runs args $RUN`로 먼저 확인해, 실행할 수 없는 ID는 제출 전에 거부한다
- 이미 같은 이름의 작업이 대기 중이거나 실행 중이면 제출하지 않는다 (`squeue -n $RUN -u $USER`)

**`gpu_train/train.sbatch RUN`** — 작업 본문:

1. **준비**: `site.env`를 읽고 conda 환경을 활성화한다(`CONDA_SH`가 비어 있으면 건너뛴다). 저장소 루트로 이동하고 `check_inputs.sh`를 실행한다. 실패하면 종료 코드 2로 끝낸다
2. **완료 확인**: `runs/gpu/{RUN}/DONE`이 있으면 할 일 없이 끝낸다
3. **이어서 제출 (Slurm 안에서만)**: 학습 전에 같은 스크립트를 `--dependency=afterany:$SLURM_JOB_ID`로 하나 더 제출하고, 그 작업 ID를 기억한다. 사슬 번호(`CHAIN_INDEX`)가 `CHAIN_MAX`에 이르면 제출하지 않는다
   - 이 작업이 학습을 끝내면 대기 중인 다음 작업을 `scancel`한다
   - 이 작업이 시간 제한으로 끊기면 다음 작업이 체크포인트에서 이어 간다
   - **전진 확인**: 다음 작업은 시작할 때 체크포인트 step이 이전 작업 시작 때보다 커졌는지 본다. 같으면 더 제출하지 않고 끝낸다 (기록 파일에 남긴 step과 비교한다)
4. **기록**: `runs/gpu/meta/{RUN}_{작업ID}.txt`에 다음을 남긴다
   - 시각, 호스트, Slurm 작업 ID·사슬 번호·노드 목록
   - `git rev-parse HEAD`, `git status --porcelain` (미커밋 변경 여부)
   - `nvidia-smi --query-gpu=name,memory.total,driver_version`, torch·CUDA 버전
   - 학습 명령 전체, 환경변수 `PYTORCH_CUDA_ALLOC_CONF`
   - 시작 시 체크포인트 step
5. **GPU 사용량**: 60초마다 `nvidia-smi`의 사용 메모리·사용률·온도를 `runs/gpu/logs/gpu_{RUN}.log`에 덧붙인다 (기존 `train_player.sh`와 같은 형식)
6. **학습과 재시도**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 `runs.py args`의 명령을 실행한다. 실패하면 20초 뒤 다시 실행한다. 한 작업 안에서 최대 10회까지 다시 실행하고, 체크포인트가 전진하지 못한 채 연속 3회 실패하면 멈춘다 (기존 방식과 같다). 시도마다 시작·종료 시각, 종료 코드, 체크포인트 step을 기록 파일에 덧붙인다
7. **완료**: 종료 코드 0이면 `DONE`(완료 시각, 작업 ID)을 쓰고, 다음 작업을 취소한 뒤 끝낸다

- **Slurm 밖 실행**: `SLURM_JOB_ID`가 없으면 3(이어서 제출)과 `scancel`을 건너뛰고 나머지를 그대로 실행한다. 기록 파일의 작업 ID 자리에는 `local_{시각}`을 쓴다. 이 모드로 로컬에서 본문을 확인한다
- `#SBATCH` 헤더에는 서버와 무관한 값만 둔다 (`--signal` 없음, 출력 경로만). 나머지는 submit.sh가 붙인다

### 검증 (로컬)

- `bash -n` 통과
- **로컬 smoke**: `bash gpu_train/train.sbatch smoke` (Slurm 밖 실행, 로컬 RTX 4070 Ti)
  - 입력 확인 통과, 50 갱신 학습, `DONE` 생성, 기록 파일과 GPU 로그 생성
  - `log.jsonl`의 baseline 줄(앞쪽 500국면, 캐시된 q로 계산)을 기준값으로 README에 적는다. 장비와 관계없이 같아야 하므로 서버 smoke에서 이 값과 비교한다
  - 같은 명령을 한 번 더 실행하면 `DONE`을 보고 바로 끝난다
- **재시도 확인**: smoke 도중 학습 프로세스를 강제로 끝내면 다음 시도가 체크포인트에서 이어지는지 확인한다 (기록 파일의 시도 2회, 재개 step)
- **제출 스크립트**: `sbatch`가 없으므로, `SBATCH_DRY_RUN=1`이면 실행할 `sbatch` 명령만 출력하는 모드를 둬서 옵션 조합(빈 값 생략)을 확인한다
- Slurm 고유 동작(헤더, 이어서 제출, `scancel`)은 서버 smoke에서 확인한다

---

## 4단계: 결과 회수와 문서

### 목표

결과를 하나로 묶어 가져올 수 있게 하고, 매뉴얼과 기준 문서 정정을 마친다.

### 구현

**`gpu_train/collect.sh RUN`** — 결과 묶음:

- 대상: `runs/gpu/{RUN}/`의 `adapter.pt`, `config.json`, `log.jsonl`, `DONE`, 그리고 `runs/gpu/logs/{RUN}_*.out`, `gpu_{RUN}.log`, `runs/gpu/meta/{RUN}_*.txt`
- `checkpoint.pt`는 넣지 않는다 (재개용, 약 12.7 MB)
- `adapter.pt`의 크기와 SHA-256을 `SUMMARY.txt`에 넣는다. 함께 넣는 것: `log.jsonl`의 baseline 줄, 마지막 eval 줄, 시도·재개 요약
- `runs/gpu/collect/{RUN}_{날짜}.tar.gz`로 묶는다. `DONE`이 없으면 경고하고 묶는다 (중간 회수용)

**`gpu_train/README.md`** — 매뉴얼 (한국어):

1. 개요: 무엇을 어디서 돌리는가, 디렉토리 구성
2. 저장소 받기: `git clone`, `devel` 브랜치, 커밋 확인
3. 입력 옮기기: `rsync`/`scp` 명령 예시(13개 파일, 약 660 MB), `check_inputs.sh`
4. 서버 설정: `site.env.example` 복사, 빈칸마다 확인 명령
5. 환경 구성: `setup_env.sh`
6. smoke: `submit.sh smoke`, 확인할 것(기록 파일의 GPU 이름, baseline 값, `DONE`)
7. 본 학습: `submit.sh R1` 등, 동시 제출과 하나씩 제출, 예상 시간(약 3~3.6시간, 로컬 RTX 4070 Ti 기준)
8. 상태 확인: `squeue`, `tail` 로그, 이어서 제출 사슬 확인, 취소(`scancel`)
9. 결과 회수: `collect.sh`, 로컬로 가져와 `mydocs/test_result/`에 정리할 항목 (학습 가이드 7절과 같은 목록)
10. 문제 해결: 입력 확인 실패, CUDA 메모리 부족(`--max-moves-per-forward`는 runs.py에서만 바꾼다), 사슬이 멈춤, `수정 전 데이터 순서 방식의 체크포인트` 오류
11. R3~R10 추가 방법: α가 정해지면 `runs.py`에 적고 테스트를 갱신한다

**기준 문서 정정** — `mydocs/tech/evaluation_criteria.md`:

- 0절 정정 이력에 한 줄 추가: 2026-11-04, 8.2절, Docker → Miniconda·Slurm, 이유(서버 환경, 작업지시자 결정, Task #10)
- 8.2절 표의 "실행 환경", "실행", "평가 확장" 행을 고치고, 고친 자리에 "정정 (2026-11-04, Task #10)" 표시를 붙인다. 원래 내용은 지우지 않고 정정 표시와 함께 남긴다 (문서 규칙)

**링크**:

- `mydocs/manual/training_guide.md` 1절에 "Slurm·conda 서버는 `gpu_train/README.md`" 한 줄
- `README.md`의 문서 표에 `gpu_train/README.md` 한 줄, 디렉토리 구조에 `gpu_train/` 한 줄

### 검증

- 로컬 smoke 결과로 `collect.sh smoke`를 실행해 묶음 안의 파일 목록과 `SUMMARY.txt`를 확인한다
- README의 명령이 실제 스크립트 이름·인자와 맞는다
- 전체 pytest 통과, `git diff --stat`에 `src/`가 없다

---

## 5단계: 입력 배포 (Hugging Face Hub 비공개 저장소)

추가 단계 (작업지시자 결정, 2026-11-04). 4단계 뒤에 조사해 보니 로컬 개발 컨테이너에 `rsync`, `scp`, `ssh`가 없어서, README 3절의 직접 전송을 여기서 실행할 수 없었다. 입력(13개, 660,418,156 bytes)을 Hugging Face Hub의 비공개 저장소에 올리고, 서버는 고정한 리비전을 받아 체크섬으로 확인한다. 입력이 바뀔 때도 같은 절차로 갱신한다.

### 목표

- 로컬: 명령 하나로 입력을 올리고, 올린 리비전을 저장소 파일에 고정한다.
- 서버: 명령 하나로 그 리비전을 받고 `check_inputs.sh`까지 통과시킨다.
- 갱신: 새로 올리고 → 리비전 파일을 커밋하고 → 서버에서 `git pull` 후 다시 받는다. 바뀐 파일만 받는다.

### 구현

**`gpu_train/inputs.hf`** (커밋. 비밀 값은 없다):

```bash
HF_REPO=           # 예: 계정명/chess-transformer-inputs (비공개 model 저장소)
HF_REVISION=       # 올린 커밋의 해시 (40자). upload_inputs.sh가 채운다
HF_TAG=            # 같은 커밋의 태그 (예: inputs-20261104). 사람이 읽기 위한 값
```

- 서버는 태그가 아니라 커밋 해시로 받는다. 태그는 옮겨 붙일 수 있지만 커밋 해시는 바뀌지 않기 때문이다.

**`gpu_train/upload_inputs.sh 저장소ID`** (로컬, 쓰기 토큰 필요):

1. `check_inputs.sh`로 로컬 입력이 매니페스트와 같은지 먼저 확인한다
2. 저장소가 없으면 **비공개**로 만든다. 이미 있으면 공개 여부를 확인하고, 공개 저장소면 멈춘다
3. `inputs.sha256`에 적힌 13개 파일과 `inputs.sha256` 자신을 같은 상대 경로로 커밋 하나에 올린다. 커밋 메시지에 날짜와 저장소 커밋을 적는다
4. 그 커밋에 태그 `inputs-{날짜}`를 붙인다 (같은 날 두 번이면 `-2`)
5. `gpu_train/inputs.hf`에 저장소 ID, 커밋 해시, 태그를 쓴다. 커밋은 사람이 한다

- `huggingface_hub`의 Python API(`HfApi.create_repo`, `create_commit`, `create_tag`)를 쓴다. CLI는 버전마다 이름이 달라졌기 때문이다(`huggingface-cli` → `hf`).

**`gpu_train/fetch_inputs.sh`** (서버 로그인 노드, 읽기 토큰 필요):

1. `inputs.hf`를 읽는다. 값이 비었으면 멈춘다
2. 고정 리비전에서 `inputs.sha256`에 적힌 파일만 저장소 루트로 받는다 (`snapshot_download(…, revision=HF_REVISION, allow_patterns=…, local_dir=.)`). 이미 받은 파일 중 같은 것은 다시 받지 않는다
3. 받은 `inputs.sha256`이 git의 `inputs.sha256`과 같은지 본다. 다르면 리비전과 매니페스트가 어긋난 것이므로 멈춘다
4. `check_inputs.sh`를 실행한다

- 받기 중 생기는 `.cache/huggingface/`(저장소 루트)는 `.gitignore`에 넣는다.

**토큰** — 저장소와 설정 파일에 두지 않는다:

- 로컬(올리기): 쓰기 권한 토큰으로 `hf auth login`. 작업지시자가 직접 한다 (`! hf auth login`)
- 서버(받기): **그 저장소만 읽을 수 있는 fine-grained 토큰**으로 `hf auth login`, 또는 `HF_TOKEN` 환경변수. 토큰은 `~/.cache/huggingface/token`에 저장된다

**의존성**: `requirements.txt`에 `huggingface_hub`를 한 줄 더한다. 서버는 `setup_env.sh`가 함께 설치하고, 로컬은 `pip install -r requirements.txt`로 설치한다. `src/` 코드는 이 패키지를 쓰지 않는다.

**문서**:

- `gpu_train/README.md`
  - 3절: Hugging Face로 받기를 기본으로 하고, 직접 전송(rsync)은 대안으로 남긴다. 서버 토큰을 만드는 방법을 적는다
  - 1절 흐름도·파일 표, 10절 문제 해결(401·403·404, 리비전 불일치), 11절 입력 갱신 절차(`make_manifest.sh` → `upload_inputs.sh` → `inputs.hf` 커밋)
  - 순서도 바꾼다: 환경 구성(5절)이 `huggingface_hub`를 설치하므로, 입력 받기가 환경 구성 뒤로 온다
- `mydocs/manual/training_guide.md` 2절 방법 A에 한 줄

### 검증

- `bash -n` 통과, 전체 pytest 통과, `src/` 변경 없음
- 토큰이 없을 때, `inputs.hf`가 비었을 때 이유를 출력하고 멈추는지
- **실제 업로드** (작업지시자가 로컬에서 `hf auth login`을 한 뒤. 외부 서비스에 올리는 일이므로 실행 직전에 저장소 ID와 함께 한 번 더 확인받는다):
  - 비공개 저장소가 만들어지고 13개 + 매니페스트가 올라가는지
  - 태그와 `inputs.hf`가 채워지는지
  - 로그인하지 않은 상태로 받으려 하면 거부되는지 (비공개 확인)
- **받기 확인** (로컬 스크래치패드에 저장소 사본을 만들어, 서버 대신):
  - 빈 사본에서 `fetch_inputs.sh` → `check_inputs.sh` 통과
  - 파일 하나를 지우고 다시 받으면 그 파일만 받는지
  - 파일 하나를 1바이트 바꾸면 다시 받거나 `check_inputs.sh`에서 걸리는지

---

## 서버 smoke (5단계 승인 뒤, 작업지시자 실행)

`gpu_train/README.md` 2~6절을 따라 서버에서 실행한다. 다음을 `mydocs/test_result/task_m100_10_test.md`에 저장해 주시면 최종 보고서에서 확인한다.

- `setup_env.sh` 출력 끝부분 (torch·CUDA 버전, pytest 결과)
- `runs/gpu/meta/smoke_*.txt`
- `collect.sh smoke`의 `SUMMARY.txt`
- 이어서 제출 사슬이 생겼다가 취소됐는지 (`sacct --name=smoke` 출력. 처음 계획의 `sacct -n smoke`는 `-n`이 머리글 없음 옵션이라 틀렸다)
- `fetch_inputs.sh` 출력 끝부분 (받은 리비전, `check_inputs.sh` 결과)

## 검증 (전체)

- `pytest` 전체 통과 (기존 + `tests/test_gpu_train.py`)
- `src/` 변경 없음
- 로컬 smoke(Slurm 밖 실행)와 재시도 확인 통과
- 서버 smoke 결과 확인 (최종 보고서)

## 예상 단계: 5단계

- 1~4단계는 처음 계획이고, 5단계(입력 배포)는 4단계 뒤에 더했다 (작업지시자 결정, 2026-11-04).
