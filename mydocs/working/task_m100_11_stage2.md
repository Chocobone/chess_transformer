# Task #11 단계2 완료 보고서: 학습 정의 추가와 서버 평가 도구

## 수행 내용

재수정한 구현 계획서(5단계, 2026-11-10 승인)의 2단계다.

1. **학습 ID 추가 (`gpu_train/runs.py`)**: `R1s0`(탈)·`R2s0`(스미슬로프)·`R8s0`(카르포프)을 추가했다.
   - 대응하는 R1·R2·R8과 seed(0)만 다르다. 출력 폴더는 `runs/gpu/R1s0` 등이다.
   - `submit.sh`, `train.sbatch`, `collect.sh`는 ID만 받으므로 바꾸지 않았다.
2. **재평가 스크립트 (`gpu_train/eval_adapter.py`)**: 학습 폴더 하나를 다시 평가해 비교한다.
   - `config.json`에서 평가 조건(기사·데이터, 손실 설정, 배치, `eval_max_moves`, 정밀도, `eval_positions`)을 읽어 학습 중 평가와 같게 맞춘다.
   - `log.jsonl`의 마지막 eval 줄과 비교하고, `--baseline`이면 baseline 줄과도 비교한다.
   - 판정: top1·top3는 정확히 같아야 하고, 나머지 5개 지표는 |차이| < 5 × 10⁻⁵이어야 한다. 통과하면 종료 코드 0, 아니면 1이다.
   - val 국면 수가 `config.json`의 `val_positions`와 다르면 평가하지 않고 멈춘다.
   - 출력·JSON에 SHA-256, 장치, torch 버전, 걸린 시간을 남긴다.
3. **재평가 작업 (`gpu_train/eval.sbatch`)**: 여러 학습 폴더를 차례로 평가한다.
   - 준비는 `train.sbatch`와 같다: `site.env`, conda, `check_inputs.sh`.
   - `runs/gpu/eval/`에 남기는 것: 폴더별 `.json`·`.txt`, 기록 `meta_{작업ID}.txt`(git HEAD, 미커밋 변경, GPU, torch, 판정 표), GPU 로그.
   - 판정은 결과 JSON으로 읽는다. JSON은 평가 전에 지우므로, JSON이 없으면 "오류"로 적어 판정 "실패"와 구분한다.
   - 하나가 실패해도 나머지는 평가한다. 종료 코드는 0 모두 통과, 1 실패·오류, 2 입력 확인 실패다.
4. **제출 (`gpu_train/submit_eval.sh`)**: `submit.sh`와 같은 방식으로 Slurm 옵션을 붙인다.
   - 작업 이름 `eval`, 출력은 `runs/gpu/logs/eval_%j.out`이다. `SBATCH_DRY_RUN=1`을 지원한다.
   - 제출 전에 각 폴더의 `config.json`·`log.jsonl`·`adapter.pt`를 확인하고, 폴더 이름(결과 파일 이름)이 겹치면 거부한다.
5. **`gpu_train/README.md`**
   - 1절 파일 표·결과 경로 표, 개요 그림에 재평가를 추가했다.
   - 7절에 R1s0·R2s0·R8s0을 추가했다.
   - 9절에 "재평가" 소절(제출, 판정, 결과 파일, 묶기)을 넣었다. 절 번호를 바꾸지 않으려고 새 절이 아닌 소절로 넣었다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/runs.py` | 학습 ID `R1s0`·`R2s0`·`R8s0` |
| `gpu_train/eval_adapter.py` | 신규. 학습 폴더 재평가와 판정 |
| `gpu_train/eval.sbatch` | 신규. 재평가 작업 본문 |
| `gpu_train/submit_eval.sh` | 신규. 재평가 제출 |
| `gpu_train/README.md` | 1·7·9절 (아래 "발견 사항" 3의 한 줄 포함) |
| `tests/test_gpu_train.py` | seed 0 학습 ID 2종, 셸 스크립트 구문, 제출 dry-run, 제출 거부 |
| `tests/test_eval_adapter.py` | 신규. 재평가 재현, 판정 경계값, 바꿔치기 검출, 작업 스크립트 실행 |
| `mydocs/working/task_m100_11_stage2.md` | 이 보고서 |
| `mydocs/orders/20261110.md` | 진행 기록 |

`src/` 변경은 없다.

## 발견 사항

1. **이 PC에는 NVIDIA GPU가 없다** (Ryzen 7 7840U, Radeon 780M). 그래서 테스트는 개발 컨테이너 이미지를 GPU 없이 `docker run --rm`으로 돌렸다.
   - 이 이미지에는 `fastapi`가 없어 대국 앱 테스트를 모을 수 없었다. 일회용 컨테이너 안에서만 `pip install --user fastapi uvicorn httpx`를 하고 돌렸으며, 이미지와 볼륨은 바꾸지 않았다.
2. **같은 장치에서는 재평가가 학습 중 평가를 차이 0으로 재현한다** (CPU 테스트). 서버에서도 같은 GPU·같은 코드면 차이가 0이거나 매우 작을 것으로 본다.
3. **README 7절의 서버 실측 한 줄**(5단계 산출물)은 작업지시자 지시로 미리 써 두었다. 같은 파일이라 이번 커밋에 함께 들어간다.
4. 계획과 다른 점: `eval.sbatch` 실행 테스트는 `tests/test_gpu_train.py`가 아니라 `tests/test_eval_adapter.py`에 넣었다. 소형 학습 폴더 fixture가 그쪽에 있기 때문이다.
5. 테스트를 위해 `eval.sbatch`·`submit_eval.sh`에 `SITE_ENV`(설정 파일 경로)를 두었다. 기본값은 `gpu_train/site.env`라 서버 사용법은 같다. 기존 `submit.sh`는 그대로다.

## 검증

- 새 테스트: `test_eval_adapter.py` 29개, `test_gpu_train.py` 추가 8개, 모두 통과
  - 재평가가 마지막 eval·baseline을 차이 0으로 재현한다
  - 다른 seed 어댑터로 바꿔치기하면 실패한다. baseline은 어댑터와 무관하므로 통과한다
  - 판정 경계: 4.9 × 10⁻⁵ 통과, 5.1 × 10⁻⁵ 실패. top1·top3는 10⁻¹² 차이도 실패
  - val 국면 수가 다르면 거부한다
  - `eval.sbatch`(Slurm 밖)는 모두 통과하면 0이다. 실패, 오류(폴더 없음), 통과가 섞이면 1이고, 판정 표에 셋이 구분되어 남으며 뒤의 폴더도 평가한다
  - `submit_eval.sh` dry-run이 site.env 값대로 옵션을 붙이고(빈 값은 생략, `--dependency` 없음), 파일이 빠진 폴더와 이름이 겹치는 폴더를 거부한다
  - `R1s0`·`R2s0`·`R8s0`은 대응하는 R와 `seed`·`out`만 다르고, `list`에서 "가능"이다
- 전체 pytest: **196 passed**, 0 failed (컨테이너 이미지, CPU, torch 2.15.0)
- `git diff --stat -- src/` 비어 있음: ✅
- `python -m gpu_train.runs list`에 R1s0·R2s0·R8s0이 `seed 0`, `가능`으로 나온다: ✅

## 다음 단계

3단계: 배포와 서버 실행이다.

1. 이 단계 커밋을 `local/task11 → local/devel → devel`로 merge하고 `origin/devel`에 push한다. **push 전에 승인을 요청한다**
2. 작업지시자가 서버에서 `git pull`을 한다
3. `submit.sh R1s0`·`R2s0`·`R8s0`(약 27시간)과 `submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8`(약 45분)을 제출한다

## 커밋

(승인 후 기록)
