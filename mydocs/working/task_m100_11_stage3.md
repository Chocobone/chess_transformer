# Task #11 단계3 완료 보고서: 배포와 서버 실행 안내

## 수행 내용

1. **배포**: 2단계 커밋을 fast-forward로 `local/task11 → local/devel → devel`에 merge하고 `origin/devel`에 push했다 (작업지시자 승인, 2026-11-10)
   - push 범위: `96efc78..1391a82`
   - 서버가 받을 커밋은 `1391a82` (Task #11 단계2 보고서 커밋 해시 기록)다. 소스는 `47f8b8c`와 같다
2. **R1·R2·R8 학습 커밋과의 차이 확인**: `git diff f39318a 1391a82`로 학습 결과에 영향을 주는 파일을 비교했다
   - 비교한 파일: `src/`, `requirements.txt`, `gpu_train/environment.yml`, `inputs.sha256`, `inputs.hf`, `train.sbatch`, `submit.sh`
   - 모두 **변경 없음**이다. 바뀐 것은 문서, 테스트, `runs.py`(학습 ID 추가), 새 재평가 파일뿐이다
   - 그래서 seed 0 학습은 R1·R2·R8과 같은 코드·같은 입력으로 돌고, 다른 것은 seed뿐이다
   - 서버 conda 환경도 같은 것을 쓰면 torch(2.5.1+cu121)도 같다. 4단계에서 작업 기록으로 확인한다

## 서버 실행 안내 (작업지시자)

서버 저장소 루트에서 실행한다.

```bash
git pull                                   # devel → 1391a82
git log -1 --format=%h                     # 1391a82
python -m gpu_train.runs list              # R1s0·R2s0·R8s0이 seed 0, 가능
bash gpu_train/check_inputs.sh             # 입력 13개 통과

# seed 0 학습 3회 (약 27시간. 동시에 내도 되고, GPU가 빠듯하면 하나씩)
bash gpu_train/submit.sh R1s0
bash gpu_train/submit.sh R2s0
bash gpu_train/submit.sh R8s0

# R1·R2·R8 재평가 (약 45분. 학습 중에 내도 된다)
SBATCH_DRY_RUN=1 bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8   # 명령 확인
bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8
```

- `site.env`의 `SBATCH_TIME`이 45분보다 짧으면 재평가가 끊긴다. 학습은 이어서 제출되므로 상관없다.
- 진행 상황은 `squeue -u $USER`로 본다. 작업 이름은 `R1s0`·`R2s0`·`R8s0`·`eval`이다.
- 재평가의 판정은 `runs/gpu/eval/meta_<작업ID>.txt` 마지막의 판정 표에 남는다.

**끝나면 결과 묶음 4개를 로컬 저장소 루트로 옮겨 주시면 된다.**

```bash
bash gpu_train/collect.sh R1s0             # R2s0, R8s0도
tar -czf runs/gpu/collect/eval_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/eval runs/gpu/logs/eval_*.out
```

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `mydocs/working/task_m100_11_stage3.md` | 이 보고서 |
| `mydocs/orders/20261110.md` | 배포 기록 |

소스 변경은 없다.

## 발견 사항

- 이 작업 사본의 미커밋 파일은 push에 들어가지 않았다. 5단계 초안(`mydocs/tech/evaluation_criteria.md` 정정, `mydocs/test_result/task_m100_11_test.md`)과 #11 범위 밖의 `src/play/README.md`다.
- 서버의 미커밋 변경 `gpu_train/site.env.example`(이월 항목)이 `git pull`과 충돌하지 않는지 확인이 필요하다. 이번 push는 이 파일을 바꾸지 않았으므로 충돌하지 않을 것으로 본다.

## 다음 단계

4단계: 결과 묶음을 받으면 seed 0 학습 3회를 1단계 방식으로 확인하고, R1·R2·R8 재평가 판정을 대조한다.

## 커밋

`a7bdcfa` Task #11 단계3: 배포와 서버 실행 안내 보고서
