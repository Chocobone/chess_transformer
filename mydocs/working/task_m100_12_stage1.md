# Task #12 단계1 완료 보고서: 덤프 도구

## 수행 내용

### `src/eval/` (새 패키지)

| 파일 | 내용 |
|---|---|
| `__init__.py` | 패키지 설명 |
| `dump.py` | 학습 폴더 하나를 세 기사 val 전체(S_val)에서 평가해 합법수별 log π를 남긴다. 학습 기록과 대조한다 |
| `metrics.py` | 국면별 값(`per_position`), 1순위 수(`segment_argmax`), 동률 수(`count_ties`), 국면 평균(`summarize`), 기록 대조(`compare`). numpy만 쓴다 |

**`python -m src.eval.dump 학습폴더 --out runs/gpu/dump`가 남기는 파일**

| 파일 | 내용 |
|---|---|
| `positions.npz` | 모델과 무관한 국면 정보: `player`, `game`, `ply`, `offsets`, `target`, `regret`, `new`, `settings`. 처음 덤프할 때 한 번 만든다 |
| `base.npz` | 캐시의 베이스 q로 만든 log π (학습 baseline과 같은 계산, CPU) |
| `{모델 ID}.npz` | 어댑터의 log π. 모델 ID는 학습 폴더 이름이다 |
| `{모델 ID}.json` | 학습 기록 대조(마지막 eval, baseline), 1순위 동률 국면 수, 환경, 어댑터 SHA-256, 기사별 시간 |

- 정책과 평가 묶음은 학습 중 평가와 같다. 학습 `config.json`의 temperature, `batch_positions`(섞지 않음), `eval_max_moves`, 정밀도(`eval_amp` false면 float32)를 쓴다
- 세 기사 val을 `PLAYERS` 순서(탈, 카르포프, 스미슬로프)로 이어 붙인다. 합법수 순서는 캐시와 같다(행동 번호 순)
- 새 국면은 ply ≥ 20이고 FEN 앞 4필드가 세 기사 어느 train에도 없는 국면이다 (기준 문서 2.4절)
- 설정이 다른 모델은 거부한다. 덤프 하나는 한 가지 설정만 담는다
- 대조 규칙은 `gpu_train/eval_adapter.py`와 같다. top1·top3는 정확히 같고, 나머지는 |차이| < 5 × 10⁻⁵이다
- 이미 덤프했고 대조를 통과한 모델(어댑터 SHA-256 같음)은 건너뛴다
- 종료 코드: 0 대조 통과, 1 대조 실패, 3 덤프 중 오류

### 서버 작업 스크립트 (`gpu_train/`)

| 파일 | 내용 |
|---|---|
| `dump.sbatch` | 입력 확인 → 기록(호스트, GPU, git HEAD, torch) → 폴더마다 덤프 → 대조 판정 표. 하나가 실패해도 나머지를 덤프한다. 작업 기록은 `runs/gpu/dump/meta_{작업ID}.txt` |
| `submit_dump.sh` | sbatch 제출. 작업 이름 `dump`, 로그 `runs/gpu/logs/dump_{작업ID}.out`. 노드 고정은 `site.env`의 `SBATCH_EXTRA=--nodelist=moana-u1` |

- 두 파일은 Task #11의 `eval.sbatch`·`submit_eval.sh`와 같은 틀이다. `.gitattributes`의 `*.sbatch`·`*.sh` 규칙으로 LF로 고정된다

### 테스트 `tests/test_eval_dump.py` (16개)

- 소형 세 기사 데이터를 만든다. 국면마다 `game_id`·`ply`가 있고, 스미슬로프 val의 ply ≥ 20 국면 하나를 카르포프 train에 넣었다. 여기에 소형 베이스·캐시, 학습 2개(탈, 카르포프)를 쓴다

| 테스트 | 확인 |
|---|---|
| 국면 정보 | `game`·`ply`가 기록과, `offsets`·`target`·`regret`이 캐시와 같다 |
| 새 국면 | 독립 계산과 같다. 다른 기사 train에 있는 국면은 새 국면이 아니다 |
| 덤프 = `evaluate` (베이스, 두 어댑터 × 세 기사 val) | 7개 지표. top1·top3 정확히, 나머지 10⁻⁶ 이내 |
| 기록 대조 | 통과, step 6, SHA-256, 기사 |
| 다시 실행 | 통과한 덤프는 건너뛰고 파일을 다시 쓰지 않는다 |
| 어댑터 바꿔치기 | 대조 실패, 종료 코드 1 |
| 설정이 다른 모델 | 거부, 종료 코드 3 |
| 손 계산 | 1순위 동률(앞 번호), 국면별 7개 값 |
| `dump.sbatch` (Slurm 밖) | 실패·오류·통과를 판정 표에 남기고, 앞의 실패 뒤에도 덤프한다 |
| `submit_dump.sh` | 문법, 미리 보기 명령에 `--nodelist=moana-u1`, 불완전·겹치는 폴더 거부 |

## 계획과 다른 점

1. **국면별 지표 함수를 1단계에 넣었다.** 계획은 `metrics.py` 전체를 2단계에 두었다. 덤프 파일 자체로 학습 기록을 대조하려면 국면별 값 계산이 1단계에 있어야 한다. 2단계에서 같은 파일에 A1~C3 지표를 더한다
2. **분석도 컨테이너에서 한다.** 이 PC의 호스트 Python에는 numpy가 없다. 계획의 "호스트 Python에서도 분석할 수 있다"는 맞지 않는다. 분석 모듈이 torch를 쓰지 않는 원칙은 그대로 둔다(2단계 테스트로 확인)
3. **베이스 정책은 CPU에서 계산한다.** 학습 baseline은 GPU에서 캐시 q로 계산했다. 같은 식이라 차이는 반올림 수준이고, 덤프마다 baseline 줄과 대조한다

## 변경 파일

| 파일 | 변경 |
|------|------|
| `src/eval/__init__.py`, `dump.py`, `metrics.py` | 새 파일 |
| `gpu_train/dump.sbatch`, `gpu_train/submit_dump.sh` | 새 파일 |
| `tests/test_eval_dump.py` | 새 파일 (16개) |
| `mydocs/working/task_m100_12_stage1.md` | 이 보고서 |
| `mydocs/orders/20261112.md` | 1단계 기록 |

- `src/`의 기존 파일은 바꾸지 않았다 (`git diff --stat -- src/` 비어 있음)

## 검증

- 새 테스트: 16 passed ✅
- 전체 pytest: **212 passed** (기존 196 + 16), 0 failed ✅
  - 개발 컨테이너 이미지, CPU, torch 2.15.0, Python 3.13.18
  - 대국 앱 테스트를 위해 일회용 컨테이너 안에서만 `fastapi`·`uvicorn`·`httpx`를 설치했다 (Task #11과 같다)
- `bash -n gpu_train/dump.sbatch`, `gpu_train/submit_dump.sh` ✅

## 다음 단계

2단계: `src/eval/stats.py`(부트스트랩), `metrics.py`에 A1~A3·B1·B2·B3·B10·B11·C3 추가, `judge.py`(판정과 보고), 테스트.

## 커밋

`2c92dae` Task #12 단계1: 덤프 도구
