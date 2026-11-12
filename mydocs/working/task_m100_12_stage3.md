# Task #12 단계3 완료 보고서: 배포와 서버 실행 안내

## 수행 내용

### `gpu_train/README.md` 덤프 절 추가

| 위치 | 추가 내용 |
|---|---|
| 1절 개요 그림 | `submit_dump.sh runs/gpu/R1 … → dump.sbatch (모델 출력 덤프)` |
| 1절 파일 표 | `submit_dump.sh`, `dump.sbatch` (`src/eval/dump.py` 실행) |
| 1절 결과 위치 표 | `runs/gpu/dump/` (positions, base, 모델별 npz·json·txt, meta, GPU 로그) |
| 9절 "모델 출력 덤프 (`submit_dump.sh`)" (재평가 절 다음) | 목적, 노드 고정(`SBATCH_EXTRA=--nodelist=moana-u1`), 제출·확인 명령, 소요 시간과 `SBATCH_TIME` 5시간 이상, 끊겼을 때 재제출(통과한 모델 건너뜀), 학습 기록 대조, 판정 표와 종료 코드, 설정이 다른 폴더 거부, 결과 파일 표, 회수 명령 |

기존 줄은 바꾸지 않았다 (추가만 31줄).

## 검증

- 임시 `site.env`(`SBATCH_TIME=05:00:00`, `SBATCH_EXTRA=--nodelist=moana-u1`)로 README의 제출 명령을 `SBATCH_DRY_RUN=1`로 실행했다. 로컬의 `runs/gpu/R1, R2, R8, R1s0, R2s0, R8s0` 6개 폴더 확인을 통과하고 다음 명령을 낸다 ✅

```
sbatch --parsable --job-name=dump --output=runs/gpu/logs/%x_%j.out --chdir=/home/u/chess_transformer --gres=gpu:1 --time=05:00:00 --nodelist=moana-u1 gpu_train/dump.sbatch runs/gpu/R1 runs/gpu/R2 runs/gpu/R8 runs/gpu/R1s0 runs/gpu/R2s0 runs/gpu/R8s0
```

- 소스 변경은 없다. 테스트는 2단계 결과(243 passed) 그대로다

## 변경 파일

| 파일 | 변경 |
|------|------|
| `gpu_train/README.md` | 덤프 절 추가 |
| `mydocs/working/task_m100_12_stage3.md` | 이 보고서 |
| `mydocs/orders/20261112.md` | 3단계 기록 |

## 승인 후 진행할 일

1. 커밋: `Task #12 단계3: README 덤프 절`
2. `local/task12` → `local/devel` → `devel` merge (fast-forward), `origin/devel` push (구현 계획서에서 승인된 사전 반영)
3. 서버 실행 (작업지시자)

```bash
git pull
# gpu_train/site.env: SBATCH_EXTRA=--nodelist=moana-u1, SBATCH_TIME=05:00:00 이상
SBATCH_DRY_RUN=1 bash gpu_train/submit_dump.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8 runs/gpu/R1s0 runs/gpu/R2s0 runs/gpu/R8s0
bash gpu_train/submit_dump.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8 runs/gpu/R1s0 runs/gpu/R2s0 runs/gpu/R8s0
# 끝나면 (약 4.2시간)
tar -czf runs/gpu/collect/dump_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/dump runs/gpu/logs/dump_*.out
```

묶음 파일을 저장소 루트에 두면 4단계(회수, 검증, 판정)를 진행한다.

## 다음 단계

4단계: 덤프 회수와 검증, 컨테이너에서 `src.eval.judge` 실행.

## 커밋

`2ea924d` Task #12 단계3: README 덤프 절
