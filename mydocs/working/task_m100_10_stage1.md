# Task #10 단계1 완료 보고서: 학습 정의와 입력 매니페스트

## 수행 내용

- **학습 정의** `gpu_train/runs.py`
  - R1~R10과 smoke를 한곳에 정의했다. 학습 ID를 `src.train.train` 명령으로 바꾼다.
  - 기준 문서 8.1절의 공통 설정 12개를 모두 명시해, `train.py` 기본값이 바뀌어도 학습이 달라지지 않게 했다.
  - α가 정해지지 않은 R3~R7, R9, R10은 실행을 거부한다.
- **테스트** `tests/test_gpu_train.py`: 17개
- **입력 매니페스트**
  - `make_manifest.sh`: 로컬에서 목록을 만든다.
  - `inputs.sha256`: 13개 파일, 660,418,156 bytes(약 630 MiB)
  - `check_inputs.sh`: 서버에서 확인한다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/__init__.py` | 신규 (패키지 표시) |
| `gpu_train/runs.py` | 신규. `Run`, `RUNS`, `COMMON`, `train_args`, `out_dir`, 명령줄 `list`·`args`·`out` |
| `gpu_train/make_manifest.sh` | 신규 |
| `gpu_train/check_inputs.sh` | 신규 |
| `gpu_train/inputs.sha256` | 신규. 크기 주석 줄 + `sha256sum` 출력 |
| `tests/test_gpu_train.py` | 신규. 17개 |
| `mydocs/working/task_m100_10_stage1.md` | 이 보고서 |

`src/` 변경은 없다.

## 학습 명령 (R1)

```
python -u -m src.train.train --player tal --out runs/gpu/R1 --base base_model/136M.pt --rank 16 --lora-alpha 32
  --lr 1e-4 --epochs 3 --batch-positions 32 --temperature 0.02 --tau 0.02 --gamma 0 --eval-every 1000
  --checkpoint-every 200 --num-workers 2 --alpha 0.0 --lam 1.0 --seed 1 --resume
```

`python -m gpu_train.runs list`의 출력은 다음과 같다. 실행할 수 있는 것은 R1, R2, R8, smoke다.

```
ID     기사       α      λ    seed 실행   역할
R1     tal      0      1    1    가능   α = 0. 잡음 바닥, D 기준선
R2     smyslov  0      1    1    가능   α = 0. 잡음 바닥, D 기준선
R8     karpov   0      1    1    가능   α = 0. 잡음 바닥, D 기준선
R3     tal      미정     1    1    불가   α = a₁
...
R7     tal      미정     0    1    불가   α = a₂, λ = 0. 기력 항 ablation
smoke  tal      0      1    1    가능   서버 확인용 50 갱신
```

## 발견 사항

1. **기존 어댑터와 설정이 같음을 테스트로 확인했다.**
   - 비교 대상은 `runs/{tal,karpov,smyslov}_136M/config.json`이다.
   - 데이터·실행에 따라 정해지는 키(`out`, `seed`, `total_steps` 등 9개)를 빼면, 세 기사 모두 R1·R2·R8과 다른 키가 없다. seed만 0 → 1이다.
   - 기준 문서 B3의 전제("같은 기사 쌍의 차이는 seed와 Task #5 결함 차이뿐")가 설정 쪽에서는 성립한다.
   - 이 테스트는 `runs/`가 없는 환경(서버 등)에서는 건너뛴다.
2. **smoke는 R1과 네 값만 다르다.** `out`, `max_steps` 50, `eval_every` 25, `eval_positions` 500이며, 테스트로 고정했다.
3. **입력 확인 시간은 로컬에서 약 3.7초다.** 작업을 시작할 때마다 확인해도 부담이 없다.
4. **`inputs.sha256`의 `#` 주석 줄**: 로컬 coreutils의 `sha256sum -c`에서는 경고 없이 통과했다. 서버의 coreutils가 오래되었으면 "형식이 잘못된 줄" 경고가 날 수 있다. `--strict`를 쓰지 않으므로 경고만 나고 판정에는 영향이 없다. 서버 smoke에서 확인한다.

## 검증

- `pytest tests/test_gpu_train.py`: 17 passed (기존 어댑터 설정 대조 3개 포함, 건너뜀 없음)
- 전체 `pytest`: 159 passed, 0 failed (기존 142 + 17)
- `check_inputs.sh` 로컬 통과 (13개)
- 손상 확인 (스크래치패드 사본, 원본은 그대로)
  - 정상 사본: 통과, 종료 코드 0
  - `tal/stats.json` 1바이트 변경: `FAILED`, 종료 코드 1
  - `karpov/val.jsonl` 삭제: `FAILED open or read`, 종료 코드 1
- `bash -n` 통과 (`make_manifest.sh`, `check_inputs.sh`)
- `git diff --stat -- src/`: 변경 없음

## 다음 단계

2단계: 환경 구성. `environment.yml`(Python 3.13), `setup_env.sh`, `site.env.example`(빈칸과 확인 명령), `.gitignore`에 `gpu_train/site.env`

## 커밋

`Task #10 단계1: 학습 정의와 입력 매니페스트` (작업지시자 승인, 2026-11-04)
