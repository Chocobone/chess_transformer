# Task #10 단계4 완료 보고서: 결과 회수와 문서

## 수행 내용

- **`gpu_train/collect.sh RUN`**: 결과를 `runs/gpu/collect/{RUN}_{시각}.tar.gz` 하나로 묶는다.
  - 넣는 것: `adapter.pt`, `config.json`, `log.jsonl`, `DONE`, 작업 출력(`{RUN}_*.out`), GPU 로그, 작업 기록(`{RUN}_*.txt`), `SUMMARY.txt`
  - **계획에 더한 것: 최신 `pip freeze` 기록(`env_*.txt`)도 넣는다.** 서버에서 받은 패키지 버전이 결과와 함께 남게 하기 위해서다.
  - `checkpoint.pt`는 넣지 않는다.
  - `DONE`이 없으면 경고만 하고 묶는다. `log.jsonl`이 없으면(학습 시작 전) 종료 코드 1이다.
  - `SUMMARY.txt`에 들어가는 것:
    - 완료 표시, `adapter.pt` 크기와 SHA-256
    - baseline 줄, 마지막 eval 줄, 마지막 train 줄
    - 재개 횟수와 step
    - 작업 기록 파일별 `===` 줄(시도, 예약, 취소, 완료)
- **`gpu_train/README.md`** (11절):
  1. 개요(흐름, 파일, 결과 경로)
  2. 저장소 받기
  3. 입력 옮기기 (`rsync -R` 명령, 13개 660,418,156 bytes)
  4. `site.env`
  5. 환경 구성
  6. smoke와 baseline 기준값
  7. 본 학습과 이어서 학습
  8. 상태 확인과 취소
  9. 결과 회수와 `test_result` 정리 항목
  10. 문제 해결 14가지
  11. R3~R10 추가
- **기준 문서 정정** (`mydocs/tech/evaluation_criteria.md`):
  - 0절 정정 이력의 "(없음)" 줄을 정정 1건으로 바꿨다. 날짜 2026-11-04, 대상 8.2절, 내용, 이유(작업지시자 결정, Task #10)를 적었다.
  - 8.2절 표 세 행을 정정했다. 원래 내용은 취소선으로 남기고, 바로 뒤에 "**정정 (2026-11-04, Task #10)**"과 새 내용을 붙였다.
    - 실행 환경: Docker → Miniconda
    - 실행: sbatch 제출과 이어서 학습을 추가
    - 평가 확장: "같은 이미지에" → "같은 conda 환경이나 서버에"
  - 8.2절 머리말 "디렉토리는 후속 이슈에서 만든다" 뒤에 "Task #10에서 만들었다"와 README 위치를 덧붙였다.
  - 판정 기준과 학습 설계(8.1절)는 바꾸지 않았다.
- **링크**:
  - `mydocs/manual/training_guide.md` 1절: Slurm·Miniconda 서버는 `gpu_train/README.md`를 따른다는 문단
  - `README.md`: 문서 표에 한 줄, 디렉토리 구조에 `gpu_train/` 한 줄

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/collect.sh` | 신규 |
| `gpu_train/README.md` | 신규 |
| `mydocs/tech/evaluation_criteria.md` | 0절 정정 이력, 8.2절 정정 (+5 −5줄) |
| `mydocs/manual/training_guide.md` | 1절에 링크 문단 |
| `README.md` | 문서 표, 디렉토리 구조 |
| `mydocs/working/task_m100_10_stage4.md` | 이 보고서 |

`src/` 변경은 없다 (`git diff --name-only -- src/` 0개).

## 검증

- **`collect.sh smoke`** (로컬 smoke 결과):
  - 묶음 3.9 MB, 파일 11개: 어댑터·설정·로그·`DONE`, GPU 로그, 작업 기록 4개, `pip freeze` 1개, `SUMMARY.txt`. `checkpoint.pt`는 없다.
  - `SUMMARY.txt`의 SHA-256 `806fc7ff…633f`는 3단계 값과 같다. baseline 줄도 3단계 기준값과 같다.
  - 재개 기록은 3회 `[25, 50, 50]`이었다. 3단계 시험(강제 종료 1회, 가짜 Slurm 재실행 2회)이 그대로 드러난 것이다.
  - 없는 ID(R99)와 시작하지 않은 학습(R1)은 이유를 출력하고 종료 코드 1로 끝났다.
- **README 대조**:
  - README·기준 문서·학습 가이드·저장소 README에 나오는 `gpu_train/` 파일 경로를 모두 존재 확인했다. 없는 것은 `site.env` 하나뿐인데, 사용자가 만드는 파일이므로 정상이다.
  - 명령과 인자를 스크립트와 맞춰 봤다: `submit.sh ID`, `SBATCH_DRY_RUN=1`, `collect.sh ID`, `runs list`.
  - README에 적은 수치를 3단계 실측과 학습 가이드에서 옮겼다: baseline, 메모리 4.57 GiB와 6,270 MiB, 2분, 3.1~3.6시간, 갱신 7,455~9,039, `log-every` 20.
- `bash -n gpu_train/collect.sh` 통과, `git check-attr eol`은 `lf`
- 전체 pytest: **159 passed**

## 발견 사항

1. **README 초안의 틀린 문장을 고쳤다.** 초안 5절에 "pytest 일부가 입력 파일을 쓴다"고 적었다. 확인해 보니 입력 파일을 읽는 테스트는 없었다(`tests/`에서 `base_model/`·`data/processed`는 경로 문자열로만 나온다). "pytest는 입력 파일 없이도 돈다"로 고쳤다.
2. **`make_manifest.sh`의 파일 목록은 스크립트 안에 고정돼 있다.** 그래서 입력이 바뀌면 그 목록, `inputs.sha256`, README 3절의 rsync 목록을 함께 고쳐야 한다. README 11절에 세 가지를 순서대로 적었다.
3. **학습 하나를 멈출 때는 이름으로 취소해야 한다** (`scancel -n R1 -u $USER`). 실행 중인 작업 번호만 취소하면 `afterany`로 예약된 다음 작업이 시작되어 학습을 이어 간다. README 8절에 굵게 적었다.
4. 구현 계획서의 서버 smoke 항목은 `sacct -n smoke`였다. 이렇게 쓰면 `-n`이 이름이 아니라 "머리글 없음" 옵션으로 해석된다. README에는 `sacct --name=smoke`로 적었다.
5. `runs/gpu/collect/`에 시험 묶음 1개가 남아 있다(git 제외).

## 서버 smoke (작업지시자 실행)

README 2~6절을 따라 서버에서 실행한다. 다음을 `mydocs/test_result/task_m100_10_test.md`에 저장해 주시면 최종 보고서에서 확인한다.

- `setup_env.sh` 출력 끝부분 (torch·CUDA 버전, pytest 결과)
- `runs/gpu/meta/smoke_*.txt`
- `bash gpu_train/collect.sh smoke`의 `SUMMARY.txt`
- `sacct --name=smoke --format=JobID,State,Elapsed,ExitCode` 출력. 두 번째 작업이 예약됐다가 `CANCELLED`가 됐는지 본다

## 커밋

`Task #10 단계4: 결과 회수와 문서` (승인 후 커밋)
