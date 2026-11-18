# Task #14: 270M 베이스 기사별 LoRA 어댑터 학습 준비 — 구현 계획서

> **상태 (2026-11-18)**: 승인 (해석 8건, 6단계, `hf-tools` 환경. 작업지시자 결정)
> 수행 계획서: `mydocs/plans/task_m100_14.md` (승인. 결정 4건 권장안)

## 공통

- **`src/`는 바꾸지 않는다.** 바꿀 파일은 `gpu_train/`, `tests/`, 문서다. 학습(`src/train/train.py`)과 캐시(`scripts/cache_base_q.py`)는 지금 코드 그대로 쓴다. 둘 다 이미 `--base`로 크기를 받는다.
- **136M 경로는 그대로 둔다.**
  - 기존 학습 ID(R1~R10, R1s0·R2s0·R8s0, smoke)가 만드는 명령은 한 글자도 바뀌지 않아야 한다. 고치기 전 출력을 테스트에 그대로 박아 고정한다.
  - 기존 입력 13개의 줄과 SHA-256은 매니페스트에서 순서까지 같게 둔다. 270M 파일은 그 뒤에 붙인다.
- **결정 3(서버에서 캐시 생성)에 따라 HF 업로드를 두 번 한다.**
  - 업로드 ①: `270M.pt`를 더한 14개
  - 업로드 ②: 캐시 3개를 더한 17개
  - 업로드할 때마다 `inputs.hf`를 커밋하고 `devel`에 push해야 서버가 받을 수 있다.
- **이 PC에는 `data/processed/`가 없다.** `upload_inputs.sh`는 매니페스트의 파일이 모두 로컬에 있어야 돈다(`check_inputs.sh`). 그래서 3단계에서 기존 입력 13개를 HF에서 이 worktree로 받는다 (아래 해석 6).
- 단계마다 `mydocs/working/task_m100_14_stage{N}.md`를 쓰고, 승인 뒤 그 단계 소스와 함께 `local/task14`에 커밋한다.
- 테스트는 지금과 같이 개발 컨테이너(`chess-transformer-dev`, CPU)에서 돌린다.

## 학습 정의

| ID | 기사 | α | seed | 베이스 | 합법수 한도 | 그 밖 |
|---|---|---|---|---|---|---|
| `R1_270M` | tal | 0 | 1 | `base_model/270M.pt` | 48 (해석 3) | R1과 같다 |
| `R2_270M` | smyslov | 0 | 1 | 〃 | 〃 | R2와 같다 |
| `R8_270M` | karpov | 0 | 1 | 〃 | 〃 | R8과 같다 |
| `smoke_270M` | tal | 0 | 1 | 〃 | 〃 | smoke와 같다 (50 갱신, 25마다 평가, val 500국면) |

출력 폴더는 `runs/gpu/{ID}`다. 136M 결과 폴더와 겹치지 않는다.

## 해석 (승인 요청)

| # | 항목 | 해석 |
|---|---|---|
| 1 | `Run`에 베이스 넣기 | `Run`에 `base`(기본 `base_model/136M.pt`)와 `max_moves`(기본 없음) 필드를 더한다. `train_args`는 COMMON의 `--base` 값을 `run.base`로 바꿔 쓴다. `max_moves`가 있으면 COMMON 바로 뒤에 `--max-moves-per-forward`를 붙인다. 136M 정의는 두 필드가 기본값이라 명령이 같다 |
| 2 | 136M 무변경 고정 | 고치기 전 `python -m gpu_train.runs args <ID>` 출력 7개(실행 가능 ID 전부: R1·R2·R8·R1s0·R2s0·R8s0·smoke)를 `tests/test_gpu_train.py`에 문자열로 넣어 비교한다 |
| 3 | 합법수 한도 | 처음 값은 48이다. 계산 근거는 가중치 약 1.1 GB + 65 MiB × 48 ≈ 4.2 GiB이고, A2000은 6 GB다. smoke에서 메모리가 부족하면 32로 고친다. 고치는 일은 5단계 보고서에서 승인받는다 (결정 4). 한도는 기울기 누적 단위만 바꾸고 학습 결과는 같다 (학습 가이드 6절) |
| 4 | 매니페스트 순서와 모드 | `make_manifest.sh`의 기본 목록은 17개다: 기존 13개(같은 순서) → `base_model/270M.pt` → 기사별 `base_q_270M.npz` 3개(tal, karpov, smyslov). `--base-only`를 주면 앞 14개만 쓴다 (업로드 ①용) |
| 5 | 캐시 작업 단위 | 기사 하나가 sbatch 작업 하나다. `submit_cache.sh tal karpov smyslov`는 작업 3개를 낸다. 기사당 약 5시간(추정)이라 작업 하나의 시간 한도 안에 든다. 이미 검증을 통과한 캐시가 있으면 건너뛴다 |
| 6 | 로컬 입력 준비 | 이 worktree에 기존 입력 13개를 `fetch_inputs.sh`로 받는다. 지금 리비전 `5f48d15f`이고 약 0.7 GB다. 실행 환경은 호스트 miniconda에 만든 별도 conda 환경 `hf-tools`(`huggingface_hub`만 설치)다. 읽기·쓰기 토큰 로그인(`hf auth login`)은 작업지시자가 직접 한다. 토큰은 내가 다루지 않는다 |
| 7 | 캐시 검증 | 다음 두 가지를 본다. ① 국면 수와 합법수 수가 데이터와 같다 (학습 시작 시 검사와 같은 내용을 작업 안에서 먼저 확인). ② 기사마다 `default_rng(0)`으로 train·val 국면 100개씩을 골라 로컬 CPU에서 270M q를 다시 계산하고 서버 캐시와 비교한다. ②의 통과 기준은 최대 \|Δq\| ≤ 1e-3이다 (GPU·CPU float32 차이를 넉넉히 잡은 값). 실제 최대값을 보고한다 |
| 8 | smoke 측정 항목 | 학습·평가 메모리(`log.jsonl`의 `max_memory_gib`, `gpu_smoke_270M.log`의 GPU 전체 사용량), 갱신당 시간, 평가 1회 시간. 학습 1회 시간은 136M R1·R2·R8의 갱신 수(7,455·8,736·9,039)에 smoke의 갱신당 시간을 곱하고 평가 시간을 더해 추정한다 |

## 1단계: 학습 정의와 입력 목록

### 구현

**`gpu_train/runs.py`** (기존 파일 변경)

```python
BASE_136M = "base_model/136M.pt"
BASE_270M = "base_model/270M.pt"

@dataclass(frozen=True)
class Run:
    ...
    base: str = BASE_136M
    max_moves: int | None = None   # --max-moves-per-forward. None이면 train.py 기본값(128)

RUNS += R1_270M, R2_270M, R8_270M, smoke_270M   # 표 "학습 정의"
```

- `list` 출력에 베이스 열을 더한다. ID 열 너비도 270M ID가 들어가게 넓힌다.

**`gpu_train/make_manifest.sh`** (기존 파일 변경): 해석 4대로 고친다.

**`tests/test_gpu_train.py`** (기존 파일에 추가)

- 136M 명령 7개가 고치기 전과 같다 (해석 2).
- 270M 4개: `--base base_model/270M.pt`, `--max-moves-per-forward 48`. 그 밖의 인자는 짝인 136M 정의와 같다.
- `make_manifest.sh`: 임시 저장소에 가짜 파일을 두고 실행한다. 기본 17줄, `--base-only` 14줄, 앞 13줄의 경로 순서가 지금 `inputs.sha256`과 같은지 본다.

### 검증

- pytest 전체 (313 + 새 테스트)
- `python -m gpu_train.runs list`, `args R1_270M` 출력을 보고서에 싣는다

## 2단계: q 캐시 작업 도구

### 구현

**`gpu_train/cache.sbatch`** (새 파일): 기사 하나

1. conda 활성화, `270M.pt`와 그 기사 train·val의 SHA-256을 매니페스트와 대조
2. 캐시가 이미 있고 검증을 통과하면 건너뛴다
3. 작업 기록: git HEAD, GPU, torch, 입력 SHA-256
4. GPU 사용량 기록 (60초마다)
5. `python scripts/cache_base_q.py --player <기사> --base base_model/270M.pt` (float32, `--batch-moves` 기본 256)
6. 검증 (해석 7 ①). 실패하면 캐시를 지우고 종료 코드 1
7. 캐시 SHA-256과 크기를 기록

**`gpu_train/submit_cache.sh`** (새 파일)

- `submit_dump.sh`와 같은 방식이다: `site.env` 옵션, `SBATCH_DRY_RUN=1`, `SITE_ENV`.
- 기사 이름을 확인하고(tal·karpov·smyslov, 중복 거부) 기사마다 작업을 낸다.

**`tests/test_gpu_train.py`** (추가): 문법(`bash -n`), LF 줄 끝, dry run 옵션, 잘못된·겹친 기사 거부

### 검증

- 개발 컨테이너에서 `cache.sbatch`를 Slurm 밖에서 돌린다. 데이터는 작은 가짜 데이터, 베이스는 테스트용 소형 모델이다. 끝까지 돌고 검증이 통과하는지 본다 (가능한 범위. 270M 실물은 서버에서만 돈다)

## 3단계: 업로드 ①과 서버 캐시 생성

### 진행

1. (나) 호스트에 `hf-tools` 환경을 만든다 (해석 6). (작업지시자) `hf auth login`
2. (나) `bash gpu_train/fetch_inputs.sh` → 이 worktree에 기존 입력 13개. `check_inputs.sh` 통과를 확인한다
3. (나) `make_manifest.sh --base-only` → `inputs.sha256` 14개
4. (작업지시자) `bash gpu_train/upload_inputs.sh chocobone/player_chessengine` → `inputs.hf` 갱신. `270M.pt` 1.08 GB를 올린다
5. (나) 1·2단계 소스와 `inputs.sha256`·`inputs.hf`를 `local/devel` → `devel`에 반영하고 push한다 (작업지시자 승인 뒤)
6. (작업지시자, 서버) 대기 중인 학습 작업이 없는 때에 `git pull` → 바로 `bash gpu_train/fetch_inputs.sh` (14개 통과) → `bash gpu_train/submit_cache.sh tal karpov smyslov`

### 검증

- 로컬·서버 모두 `check_inputs.sh`에서 14개 통과
- 업로드한 리비전의 매니페스트가 git의 매니페스트와 같다 (`fetch_inputs.sh`가 확인)

## 4단계: 캐시 회수·검증과 업로드 ②

### 진행

1. (작업지시자, 서버) 캐시 3개와 작업 기록을 묶어 회수한다

   ```bash
   tar -czf runs/gpu/collect/cache270_$(date +%Y%m%d_%H%M%S).tar.gz \
     data/processed/*/base_q_270M.npz runs/gpu/logs/cache_*.out runs/gpu/meta/cache_*.txt
   ```

2. (나) worktree에 풀고 검증한다 (해석 7 ①②). 작업 기록의 SHA-256과도 대조한다
3. (나) `make_manifest.sh` → 17개
4. (작업지시자) 업로드 ② → `inputs.hf`
5. (나) `devel` 반영과 push → (작업지시자, 서버) `git pull` → `fetch_inputs.sh` (17개 통과)

### 검증

- 해석 7 ①②, 서버 `check_inputs.sh` 17개 통과
- 기존 13개의 SHA-256이 업로드 ①·② 모두에서 그대로다

## 5단계: 270M smoke와 실측

### 진행

1. (작업지시자, 서버) `bash gpu_train/submit.sh smoke_270M`. A2000 노드에 고정한다 (`SBATCH_EXTRA=--nodelist=moana-u1`)
2. (나) 회수한 `log.jsonl`, `gpu_smoke_270M.log`, 작업 기록을 분석한다 (해석 8)
3. 메모리가 부족하면 한도를 32로 고치고(`runs.py`, 승인 요청) smoke를 다시 낸다
4. 학습 1회 시간 추정, `SBATCH_TIME`·`CHAIN_MAX`가 충분한지 판단한다

### 검증

- `DONE`이 있고, 학습·평가 모두 메모리 부족이 없다
- smoke의 평가 지표가 같은 val 500국면에서 베이스 기준선(`baseline` 행)과 일관된다 (50 갱신이라 큰 변화는 없어야 한다)

## 6단계: 문서 정리

- **`gpu_train/README.md`**: 270M 절. 입력 17개, 캐시 작업, 270M 학습 ID, 실측 메모리·시간, A2000 노드 고정
- **`mydocs/manual/training_guide.md`**: 6절 메모리 표에 A2000 270M 실측 행
- **`mydocs/test_result/task_m100_14_test.md`**: 캐시 생성 기록·검증, 업로드 리비전, smoke 실측, 학습 1회 시간 추정
- 최종 결과보고서는 이 단계 승인 뒤에 따로 쓴다

## 검증 (전체)

| 항목 | 기준 |
|---|---|
| 136M 무변경 | 실행 가능 ID 7개의 명령 문자열이 고치기 전과 같다. 기존 13개 입력의 SHA-256이 같다 |
| 캐시 | 해석 7 ①② 통과 |
| 서버 입력 | 서버 `check_inputs.sh` 17개 통과 |
| smoke | `DONE`, 메모리 부족 없음, 실측 기록 |
| 회귀 | `src/` 변경 없음, pytest 전체 통과 |

## 위험 (단계별)

| 단계 | 위험 | 대응 |
|---|---|---|
| 3 | 로컬에서 `fetch_inputs.sh`·`upload_inputs.sh`가 Git Bash에서 돌지 않는다 (`stat -c`, `du -cb` 등) | Git Bash의 coreutils로 먼저 확인한다. 안 되면 개발 컨테이너에 일회용으로 `huggingface_hub`를 설치해 돌린다 (토큰은 작업지시자가 환경변수로 넣는다) |
| 3 | 매니페스트를 바꾼 뒤 서버에서 `fetch_inputs.sh` 전에 학습·덤프 작업이 시작되면 입력 확인 실패(종료 코드 2) | `git pull` 직후 바로 `fetch_inputs.sh`. 대기 중인 학습 작업이 없을 때 한다 |
| 4 | 캐시 작업이 시간 한도에 걸린다 | 기사별 작업이라 약 5시간(추정)이다. 끊기면 같은 명령으로 다시 내면 그 기사만 처음부터 다시 만든다 |
| 4 | 캐시 저장 중에 끊겨 파일이 깨진다 | 작업 안의 검증(해석 7 ①)이 실패하면 지운다. 다음 제출 때 다시 만든다 |
| 5 | 한도 48에서도 학습·평가 메모리가 부족하다 | 32로 낮춘다. 그래도 부족하면 평가 묶음(`--eval-max-moves`)도 낮추는 것을 결정 요청한다 |

## 예상 단계: 6단계

수행 계획서는 5단계였다. 결정 3(서버 캐시)에 따라 업로드가 두 번이 되어, 계획서 3단계 "캐시 생성과 입력 갱신"을 3단계(업로드 ①, 캐시 생성)와 4단계(회수·검증, 업로드 ②)로 나눴다.
