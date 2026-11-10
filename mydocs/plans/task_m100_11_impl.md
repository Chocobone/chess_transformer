# Task #11: α = 0 학습 R1·R2·R8 (GPU 서버) — 구현 계획서

> **상태 (2026-11-10)**: 승인 (결정 1 권장안)
> 수행 계획서: `mydocs/plans/task_m100_11.md` (승인, 결정 3건 권장안)

## 공통

- `src/`는 바꾸지 않는다. 새 코드는 `gpu_train/eval_adapter.py`와 테스트 `tests/test_eval_adapter.py`뿐이다.
- 어댑터와 기록은 git 제외 경로(`runs/`)에 둔다. 기존 seed 0 폴더 `runs/{tal,smyslov,karpov}_136M/`은 읽기만 한다.
- 단계마다 `mydocs/working/task_m100_11_stage{N}.md`를 쓰고, 승인 뒤 그 단계 소스와 함께 `local/task11`에 커밋한다.

## 사전 조건: 로컬 실행 환경 (결정 요청 1)

2단계 로컬 재평가에는 GPU, torch, val 데이터·q 캐시가 필요하다. 이 작업 사본(Windows, `C:\Users\yooho\VSCODE\chess_transformer`)의 현재 상태는 다음과 같다 (2026-11-09 확인).

| 필요한 것 | 상태 |
|---|---|
| `base_model/136M.pt` | 있음 |
| `data/processed/{tal,smyslov,karpov}/` (val.jsonl, stats.json, base_q_136M.npz 등) | **없음** |
| torch (호스트 Git Bash) | 없음. 지금까지 학습·측정은 개발 컨테이너 `chess-dev`(`docker/compose.yml`, 저장소 루트를 `/workspace`로 마운트)에서 했다 |
| Docker | **데몬이 꺼져 있다** (Docker Desktop은 설치돼 있음) |

준비 방법(권장안):

1. 작업지시자가 Docker Desktop을 켠다
2. 저장소 루트에서 `docker compose -f docker/compose.yml up -d`로 컨테이너를 띄운다
3. 입력을 받는다. 컨테이너 안에서 `bash gpu_train/fetch_inputs.sh`를 실행하면 Hugging Face 비공개 저장소에서 받고 `check_inputs.sh`로 확인한다. 토큰(`hf auth login`)이 필요하다
   - 다른 안: 입력이 있는 다른 작업 사본에서 `data/processed/`를 복사한 뒤 `check_inputs.sh`로 확인한다

## 1단계: 배치와 무결성 확인

### 구현

1. 묶음 원본 세 개(`R1_20261109_135827.tar.gz`, `R2_20261109_135827.tar.gz`, `R8_20261109_135829.tar.gz`)와 `smoke_20261109_115519.tar.gz`를 저장소 루트에서 `runs/gpu/collect/`로 옮긴다
2. 이미 풀어 둔 `runs/gpu/collect/{R}/runs/gpu/` 아래 내용을 저장소의 `runs/gpu/`에 같은 경로로 복사한다. 복사 대상은 `runs/gpu/{R}/`, `runs/gpu/logs/`, `runs/gpu/meta/`다
   - `runs/gpu/`에 같은 이름 파일이 이미 있으면 덮어쓰지 않고 멈춘다
3. 확인. 결과는 1단계 보고서에 표로 적는다
   - `adapter.pt` SHA-256 = `SUMMARY.txt` 값
   - `DONE` 있음, `log.jsonl` 마지막 eval의 step = `config.json`의 `total_steps`
   - `config.json`을 seed 0 폴더와 비교한다: 다른 키가 `seed`, `out`뿐이어야 한다
   - 묶음 안의 어댑터 설정(`settings`: rank 16, alpha 32, dropout 0, targets q·k·v·o)이 8.1절과 같다
   - baseline 줄의 top1·top3가 seed 0과 정확히 같다

### 검증

- 위 확인이 세 학습 모두 통과
- `git status`에 `runs/` 아래 파일이 나오지 않는다 (git 제외 확인)

## 2단계: 로컬 재평가 (`gpu_train/eval_adapter.py`)

### 구현

```
python -m gpu_train.eval_adapter runs/gpu/R1 [--device cuda] [--num-workers 2] [--baseline] [--json 경로]
```

- 인자로 받은 학습 폴더의 `config.json`에서 평가에 필요한 값을 읽는다: `player`, `base`, `data_dir`, `temperature`, `tau`, `lam`, `alpha`, `gamma`, `batch_positions`, `eval_max_moves`, `eval_amp`, `amp`, `eval_positions`. 학습 때와 같은 조건으로 평가하기 위해서다
- val 로더는 `train.py`와 같게 만든다: `PositionDataset(data_dir, "val", base)`, `eval_positions`가 있으면 앞쪽만, `batch_size=batch_positions`, `shuffle=False`
- `load_base` → `load_adapter(model, 폴더/adapter.pt)` → `src.train.train.evaluate(model, loader, objective, eval_max_moves, amp and eval_amp)`
- `--baseline`이면 `evaluate(None, …)`도 계산해 `log.jsonl`의 baseline 줄과 비교한다
- **비교**: `log.jsonl`의 마지막 eval 줄과 항목별 차이를 출력한다
  - 판정 규칙은 서버 smoke와 같다. top1·top3는 정확히 같아야 하고, 나머지는 절대 차이 5 × 10⁻⁵ 미만이어야 한다(소수 넷째 자리)
  - 통과하면 종료 코드 0, 아니면 1이다
- 어댑터 파일의 SHA-256, 장치, torch 버전, 걸린 시간도 출력한다
- `--json`이면 결과(지표, 기준값, 차이, 판정, 환경)를 파일로 남긴다
- 판정 함수(`compare(metrics, reference) -> 결과`)는 따로 두어 테스트한다
- seed 0 폴더(`runs/tal_136M` 등)에도 그대로 쓸 수 있다. 후속 "특성 없는 평가" 이슈와 서버에서 다시 쓰기 위해서다

### 테스트 (`tests/test_eval_adapter.py`)

- `tests/test_train.py`의 작은 모델·합성 데이터 방식을 따른다. CPU에서 짧게 학습해 폴더를 만들고, `eval_adapter`가 그 폴더의 마지막 eval 줄을 정확히 재현하는지 본다(같은 장치이므로 차이 0)
- `--baseline`이 baseline 줄을 재현한다
- `compare`의 경계값을 본다: top1이 다르면 실패, 소수 값 4.9 × 10⁻⁵ 차이는 통과, 5.1 × 10⁻⁵ 차이는 실패
- 어댑터를 바꿔치기하면 실패한다 (다른 seed의 어댑터)

### 실행 (로컬 GPU)

- 개발 컨테이너에서 R1·R2·R8 각각 `--baseline`으로 실행한다
- 대조군으로 seed 0 폴더 하나(`runs/tal_136M`)도 실행해, 이 스크립트가 기존 로컬 학습 기록도 재현하는지 본다

### 검증

- 세 어댑터 모두 판정 통과 (마지막 eval과 baseline 둘 다)
- 전체 pytest 통과, `git diff --stat -- src/` 비어 있음

## 3단계: 결과 정리와 기준 문서 정정

### 구현

1. **`mydocs/test_result/task_m100_11_test.md`** (학습 가이드 7절 항목)
   - 환경: 서버, 노드, GPU, 드라이버, torch, 코드 커밋, 미커밋 변경(`site.env.example`), `PYTORCH_CUDA_ALLOC_CONF`
   - 학습별 명령 전체, 작업 ID, 시도 기록(`===` 줄), 이어서 제출한 작업의 예약·취소
   - 학습별 `config.json` 요약과 seed 0 대비 다른 키
   - baseline 줄과 모든 eval 줄 (표)
   - 처리량, `max_memory_gib`, GPU 전체 최대(MiB, ÷1024 GiB), 평가 1회 시간(eval 앞뒤 train 줄의 `elapsed` 차이로 추정)
   - `adapter.pt` 크기와 SHA-256
   - 2단계 로컬 재평가 결과
   - 수치는 `log.jsonl`·`config.json`·기록 파일을 읽는 스크립트로 뽑는다. 이 스크립트는 스크래치패드에 두고 커밋하지 않는다
2. **기준 문서 정정** (`mydocs/tech/evaluation_criteria.md`, 2.7절 절차)
   - 0절 정정 이력에 2건째를 추가한다: 날짜 2026-11-09, 대상 8.4절(그리고 8.1절 머리말의 시간 문장), 내용, 이유(서버 실측, Task #11)
   - 8.4절 "GPU 학습" 행: 원래 내용은 취소선으로 남기고 "**정정 (2026-11-09, Task #11)**: 10회, 서버(RTX A2000) 실측 기준 약 90시간. R1·R2·R8 합계 27시간 19분, 남은 7회 약 63시간(스타일 항 비용 제외)"을 붙인다
   - 8.1절 머리말 "약 30~36시간 (1회 3~3.6시간 …)" 뒤에도 같은 정정 표시를 붙인다
   - 8.1절 "학습 1회가 7.86 GiB였다 (… GPU 서버 사양은 미확인)": 서버 실측(RTX A2000 6 GB, GPU 전체 최대 5,020 MiB)을 정정으로 덧붙인다
   - 판정 기준과 학습 설계(R1~R10 설정)는 바꾸지 않는다
3. **`gpu_train/README.md` 7절 시간 안내**: "로컬 RTX 4070 Ti 기준 약 3.1~3.6시간" 뒤에 서버 실측(RTX A2000에서 7시간 56분~10시간 4분)을 덧붙인다

### 검증

- test 문서의 수치가 `SUMMARY.txt`·`log.jsonl`과 같다 (스크립트 출력 대조)
- 기준 문서 정정 표시와 0절 이력이 2.7절 형식(Task #10 정정과 같은 형식)을 따른다
- 링크와 경로 존재 확인

## 예상 단계: 3단계

| 단계 | 산출물 | 커밋 |
|---|---|---|
| 1 | 1단계 보고서 (배치는 git 제외) | 보고서만 |
| 2 | `gpu_train/eval_adapter.py`, `tests/test_eval_adapter.py`, 2단계 보고서 | 소스 + 보고서 |
| 3 | `task_m100_11_test.md`, 기준 문서 정정, README 한 줄, 3단계 보고서 | 문서 |

## 결정 요청

1. ~~로컬 실행 환경 준비~~: Docker Desktop + 컨테이너에서 `fetch_inputs.sh` (권장안 승인, 2026-11-10). **로컬 실행 환경 준비** (권장: Docker Desktop을 켜고 컨테이너에서 `fetch_inputs.sh`로 받기). 2단계 전에 작업지시자가 Docker Desktop을 켜 주셔야 한다. 다른 안은 두 가지다
   - 입력이 있는 다른 작업 사본에서 `data/processed/`를 복사한다
   - 재평가를 서버에서 sbatch로 돌린다. 이 경우 "로컬에서 같다"는 확인이 아니라 "받은 파일이 서버 기록과 같다"는 확인만 된다
