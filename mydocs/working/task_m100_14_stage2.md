# Task #14 단계2 완료 보고서: q 캐시 작업 도구

## 수행 내용

서버에서 기사별 270M 베이스 q 캐시(`base_q_270M.npz`)를 만드는 작업 틀과 캐시 검증 도구를 만들었다. 기존 덤프 작업 틀(`dump.sbatch`, `submit_dump.sh`)과 같은 방식이다.

| 파일 | 하는 일 |
|---|---|
| `gpu_train/check_cache.py` | 캐시 검증 (해석 7). ① 구조와 ② 다시 계산을 본다 (아래). 종료 코드 0 통과, 1 실패. 마지막 줄에 결과를 JSON 한 줄로 낸다 |
| `gpu_train/cache.sbatch` | 기사 하나의 작업 본문. 순서는 아래와 같다 |
| `gpu_train/submit_cache.sh` | 기사마다 작업 하나를 낸다 (해석 5). 작업 이름은 `cache_{기사}`다. `site.env` 옵션, `SBATCH_DRY_RUN=1`, `SITE_ENV`를 지원한다. 기사 이름을 확인하고 중복을 거부한다. 실제 제출 전에는 `REPO_DIR`와 `base_model/270M.pt`가 있는지 확인한다 |

`check_cache.py`의 두 검사:

- **① 구조**: split마다 국면 수가 데이터와 같은지, offsets와 길이가 맞는지, q가 유한하고 [0, 1] 안인지 본다. 국면마다 합법수 수와 행동 번호가 `encode_position` 순서와 같은지도 본다. 학습 시작 때 검사하는 내용을 미리 전부 확인하는 셈이다.
- **② 다시 계산** (`--recompute N`): split마다 `default_rng(0)`으로 국면 N개를 골라 q를 다시 계산하고 최대 |Δq|를 낸다. 기준은 1e-3이다.

`cache.sbatch`의 순서:

1. 매니페스트 대조. 베이스와 그 기사의 train·val·stats 4줄만 본다. 캐시는 아직 없을 수 있어서다.
2. 검증을 통과한 캐시가 있으면 건너뛴다. 검증에 실패한 캐시는 지운다.
3. 작업 기록을 남긴다: git HEAD, GPU, torch, 입력 SHA-256, 명령.
4. GPU 사용량을 60초마다 기록한다.
5. `scripts/cache_base_q.py`를 돌린다 (float32, 묶음 256).
6. 검증 ①. 실패하면 캐시를 지우고 종료 코드 1로 끝난다.
7. 캐시 크기와 SHA-256을 기록한다.

종료 코드는 0 완료(또는 이미 있음), 1 실패, 2 입력 확인 실패다. 작업 기록은 `runs/gpu/meta/cache_{기사}_{작업ID}.txt`, 로그는 `runs/gpu/logs/cache_{기사}_{작업ID}.out`이다. 4단계 회수 명령과 경로가 맞는다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/check_cache.py` | 새 파일 |
| `gpu_train/cache.sbatch` | 새 파일 |
| `gpu_train/submit_cache.sh` | 새 파일 |
| `tests/test_gpu_cache.py` | 새 파일. 테스트 12개 |

기존 파일과 `src/`는 바꾸지 않았다.

## 발견 사항

1. **같은 장비에서도 묶음 크기가 다르면 q의 float32 끝자리가 다르다.** 소형 모델·CPU에서 묶음 200과 256의 차이는 최대 3.0 × 10⁻⁸이었다. 그래서 테스트는 q를 정확히 같은지 대신 1e-6 이내로 비교한다. 국면·합법수 배열은 정확히 같은지 본다. 4단계 검증(서버 GPU 캐시와 로컬 CPU 재계산)의 기준 1e-3은 이보다 훨씬 넉넉하다.
2. **테스트를 위한 환경변수 4개를 두었다.** 기본값으로 돌리면 서버 동작은 계획 그대로다. 테스트에서는 작업 기록이 저장소의 `runs/`에 쌓이지 않게 임시 폴더로 보낸다.

| 환경변수 | 기본값 |
|---|---|
| `CACHE_BASE` | `base_model/270M.pt` |
| `CACHE_DATA_ROOT` | `data/processed` |
| `CACHE_LOG_ROOT` | `runs/gpu` |
| `CACHE_SKIP_INPUT_CHECK` | 대조함 |

3. **지금 매니페스트(136M 13개)로는 캐시 작업이 시작하지 않는다** (종료 코드 2). 3단계에서 `270M.pt`가 든 14개 매니페스트를 서버에 반영해야 돈다. 테스트로 확인했다.
4. **270M 실물 작업은 아직 돌리지 않았다.** 이 PC에는 GPU와 기사 데이터가 없다. 실제 작업은 3단계에서 서버가 돌린다.

## 검증

새 테스트 12개(`tests/test_gpu_cache.py`)가 모두 통과했다 ✅

| 테스트 | 확인 내용 |
|---|---|
| 문법·LF ×2 | `bash -n`, 줄 끝이 LF |
| `submit_cache.sh` dry run | 기사 3명 → sbatch 3줄. 작업 이름 `cache_{기사}`, chdir, partition, gres, time, CPU, 메모리, nodelist. 빈 account는 붙이지 않는다 |
| 잘못된 인자 ×3 | 모르는 기사, 중복, 인자 없음은 거부한다 (출력 없음, 종료 코드 1) |
| `check_cache` 통과 | 구조 통과, 다시 계산 차이 < 1e-6, `main` 종료 코드 0 |
| `check_cache` q 오류 | q를 0.01 옮긴 캐시는 구조는 통과하지만 다시 계산에서 실패한다. 기준을 0.05로 주면 통과한다 |
| `check_cache` 구조 오류 | 국면 수가 다름, 합법수 순서가 바뀜, 잘린 파일, 파일 없음을 각각 잡는다 |
| `cache.sbatch` 끝까지 | Slurm 밖, 소형 모델로 만들기 → 다시 실행하면 건너뛰기 → 잘린 캐시를 지우고 다시 만들기. 결과는 기준 캐시와 같다 (q는 1e-6 이내). 작업 기록에 입력 SHA-256이 남는다 |
| `cache.sbatch` 매니페스트 | 지금 매니페스트에서는 종료 코드 2 |
| `cache.sbatch` 기사 | 모르는 기사는 종료 코드 1 |

전체 pytest는 **332 passed**다 (1단계 320 + 12). 실패와 건너뛴 테스트는 없다 ✅ (개발 컨테이너, CPU, `runs/` 읽기 전용 마운트)

## 다음 단계

3단계: 업로드 ①과 서버 캐시 생성. 순서는 아래와 같다.

1. 호스트에 `hf-tools` 환경을 만든다 → (작업지시자) `hf auth login`
2. 기존 입력 13개를 worktree로 받는다 → `make_manifest.sh --base-only`
3. (작업지시자) 업로드 ①
4. `devel` 반영과 push
5. (작업지시자, 서버) `fetch_inputs.sh` → `submit_cache.sh tal karpov smyslov`

## 커밋

(승인 후 기록)
