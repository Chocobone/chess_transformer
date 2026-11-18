# Task #14 단계1 완료 보고서: 학습 정의와 입력 목록

## 수행 내용

### 1. 270M 학습 정의 (`gpu_train/runs.py`)

- `Run`에 필드 두 개를 더했다 (해석 1).
  - `base`: 기본값 `base_model/136M.pt`
  - `max_moves`: 기본값 없음 (`train.py` 기본값 128을 쓴다)
- `train_args`가 바뀐 점은 두 가지다.
  - COMMON의 `--base` 값을 `run.base`로 바꿔 쓴다.
  - `max_moves`가 있으면 COMMON 바로 뒤에 `--max-moves-per-forward`를 붙인다.
- 상수 `BASE_136M`, `BASE_270M`, `MAX_MOVES_270M = 48`(해석 3)을 두었다.
- 270M 정의 4개를 더했다. 짝인 136M 정의와는 베이스, 합법수 한도, 출력 폴더만 다르다.

| ID | 짝 | 명령에서 달라진 부분 |
|---|---|---|
| `R1_270M` | R1 (tal) | `--out runs/gpu/R1_270M --base base_model/270M.pt … --max-moves-per-forward 48` |
| `R2_270M` | R2 (smyslov) | 〃 |
| `R8_270M` | R8 (karpov) | 〃 |
| `smoke_270M` | smoke (tal, 50 갱신) | 〃 |

`R1_270M` 명령:

```
python -u -m src.train.train --player tal --out runs/gpu/R1_270M --base base_model/270M.pt --rank 16 --lora-alpha 32 --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000 --checkpoint-every 200 --num-workers 2 --max-moves-per-forward 48 --alpha 0.0 --lam 1.0 --seed 1 --resume
```

- `list` 출력을 바꿨다.
  - 베이스 열(136M·270M)을 "실행" 열 뒤에 더했다.
  - ID 열 너비를 6에서 10으로 넓혔다. 기존 열 순서(ID, 기사, α, λ, seed, 실행)는 그대로라 기존 테스트가 그대로 통과한다.

### 2. 입력 목록 (`gpu_train/make_manifest.sh`)

해석 4대로 고쳤다.

| 모드 | 개수 | 목록 |
|---|---|---|
| 기본 | 17 | 기존 13개(같은 순서) → `base_model/270M.pt` → `data/processed/{tal,karpov,smyslov}/base_q_270M.npz` |
| `--base-only` | 14 | 기존 13개 → `base_model/270M.pt` (업로드 ①용) |

- 다른 인자를 주면 사용법을 보이고 종료 코드 1로 끝난다.
- `inputs.sha256`은 이 단계에서 만들지 않았다. 3단계에서 실제 파일로 만든다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/runs.py` | `Run.base`·`Run.max_moves`, 270M 정의 4개, `list` 베이스 열 (+21 −5) |
| `gpu_train/make_manifest.sh` | 270M 입력 4개, `--base-only` (+16 −1) |
| `tests/test_gpu_train.py` | 새 테스트 7개 |

`src/`는 바꾸지 않았다.

## 발견 사항

1. **기존 COMMON 순서를 지키는 방식으로 136M 명령이 그대로 남았다.** 270M 정의는 `--base` 값을 그 자리에서 바꾸고 한도를 COMMON 바로 뒤에 붙였다. 그래서 136M 정의의 명령 문자열은 바뀌지 않는다. `extra`로 `--base`를 한 번 더 붙이는 방법(뒤쪽 값이 쓰임)도 가능했다. 하지만 기록된 명령에 베이스가 두 번 나와 읽기 어려워서 쓰지 않았다.
2. **이 worktree에는 `runs/`와 `base_model/*.pt`가 없다.** 기존 어댑터 설정과 비교하는 테스트 3개와 Stockfish 테스트는 파일이 없으면 건너뛴다. 그래서 전체 테스트는 본 작업 사본의 `runs/`를 읽기 전용으로 붙여 건너뛰는 테스트 없이 돌렸다. `base_model/270M.pt`는 3단계에서 본 작업 사본에서 복사한다.

## 검증

- 136M 무변경: 고치기 전(`f715d0f`) `runs args` 출력 7개(R1·R2·R8·R1s0·R2s0·R8s0·smoke)를 따로 저장해 고친 뒤 출력과 `diff`로 비교했다. 같다 ✅
- 새 테스트 7개 ✅
  - `test_136m_commands_are_unchanged`: 실행 가능한 136M ID 목록과 명령 문자열이 고치기 전과 같다
  - `test_270m_runs_differ_from_pair_only_in_base_and_move_limit` ×4: 짝과 다른 인자가 `base`, `max_moves_per_forward`, `out`뿐이다
  - `test_270m_runs_are_listed_with_base`: `list`에 실행 가능·베이스가 나온다
  - `test_make_manifest_keeps_136m_order_and_appends_270m`: 가짜 입력으로 `--base-only` 14개, 기본 17개를 만든다. 앞 13개 경로가 지금 `inputs.sha256`과 같다. 잘못된 인자는 거부한다
- 전체 pytest: **320 passed** (313 + 7), 0 failed, 0 skipped ✅ (개발 컨테이너, CPU, `runs/` 읽기 전용 마운트, 대국 앱 테스트용 `fastapi`는 일회용 설치)

## 다음 단계

2단계: q 캐시 작업 도구 `gpu_train/cache.sbatch`, `submit_cache.sh`와 테스트

## 커밋

`d17935f` Task #14 단계1: 270M 학습 정의와 입력 목록 (gpu_train/runs.py, make_manifest.sh)
