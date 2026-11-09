# Task #10 최종 결과보고서: GPU 서버 학습 디렉토리 `gpu_train/` 작성

## 개요

별도 GPU 서버(Slurm + Miniconda)에서 기사별 LoRA 어댑터를 학습하는 데 필요한 파일과 절차를 `gpu_train/`에 모았다. 학습 정의, 입력 확인, 환경 구성, 작업 제출, 자동 재개, 결과 회수, 입력 배포가 들어 있다. 서버 smoke(작업지시자 실행, 2026-11-05)에서 GPU 인식, 입력 확인, 50 갱신 학습, 다음 작업 예약·취소가 확인됐다. baseline 지표는 로컬과 10⁻⁸ 이하 차이로 같았다. 이로써 수행 계획서의 끝 상태인 "서버에서 smoke가 돌고, R1·R2·R8을 학습 ID 하나로 제출할 수 있는 상태"에 도달했다. 학습 코드(`src/`)는 바꾸지 않았다.

## 수행 결과

| 단계 | 내용 | 커밋 |
|------|------|------|
| 계획 | 수행 계획서 (승인) | `5140678` |
| 계획 | 구현 계획서 (4단계) | `2812f1e` |
| 1단계 | 학습 정의 `runs.py`(R1~R10, smoke), 입력 매니페스트(13개 SHA-256)와 확인 스크립트, 테스트 17개 | `fec7a64` |
| 2단계 | 환경 구성: `environment.yml`(Python 3.13), `setup_env.sh`, `site.env.example`, LF 고정 | `276dbd2` |
| 3단계 | 작업 스크립트: `submit.sh`, `train.sbatch`(재시도, 이어서 제출, 기록, 종료 신호 처리) | `14b04b6` |
| 4단계 | 결과 회수 `collect.sh`, `gpu_train/README.md`, 기준 문서 8.2절 정정, 링크 | `914ad42` |
| 선반영 | 서버 `git clone -b devel`을 위해 최종 보고서 전에 devel 반영 (작업지시자 결정) | `7691394` |
| 계획 | 구현 계획서 5단계 추가 (입력 배포) | `cfed322` |
| 5단계 | 입력 배포: Hugging Face 비공개 저장소 `upload_inputs.sh`·`fetch_inputs.sh`·`inputs.hf`, README 3~5절 재구성 | `f39318a` |
| 서버 smoke | 작업지시자 실행 (2026-11-05), 결과 정리 `mydocs/test_result/task_m100_10_test.md` | (이 보고서와 함께 커밋) |

단계별 상세는 `mydocs/working/task_m100_10_stage1.md`~`stage5.md`에 있다.

## 산출물

### `gpu_train/` 파일

| 파일 | 하는 일 |
|---|---|
| `runs.py` | 학습 ID(R1~R10, smoke)별 설정과 학습 명령. α가 정해지지 않은 R3~R7, R9, R10은 거부 |
| `inputs.sha256`, `check_inputs.sh`, `make_manifest.sh` | 입력 13개(660,418,156 bytes)의 SHA-256, 확인, 재생성 |
| `inputs.hf`, `upload_inputs.sh`, `fetch_inputs.sh` | Hugging Face 비공개 저장소 `chocobone/player_chessengine` 리비전 `5f48d15`(`inputs-20261104`)로 입력 배포 |
| `environment.yml`, `setup_env.sh` | conda 환경(Python 3.13), torch·requirements 설치, pytest, `pip freeze` 기록 |
| `site.env.example` | 서버별 빈칸(경로, conda, torch 휠 주소, Slurm 옵션, 사슬 상한) |
| `submit.sh` | 학습 하나를 sbatch로 제출. 미리 보기(`SBATCH_DRY_RUN=1`), 중복 제출 거부 |
| `train.sbatch` | 입력 확인 → 다음 작업 예약 → 기록 → 학습(최대 10회 재시도) → `DONE` → 다음 작업 취소 |
| `collect.sh` | 결과를 tar.gz 하나로 묶고 `SUMMARY.txt` 작성 |
| `README.md` | 서버 절차 11절 (저장소 받기 ~ 문제 해결, R3~R10 추가) |

### 사용 방법 (서버)

```bash
cp gpu_train/site.env.example gpu_train/site.env   # 빈칸 채우기
bash gpu_train/setup_env.sh
hf auth login && bash gpu_train/fetch_inputs.sh
bash gpu_train/submit.sh smoke                     # 본 학습 전에 반드시
bash gpu_train/submit.sh R1                        # R2, R8도 같다
bash gpu_train/collect.sh R1
```

### 그 밖의 변경

- `mydocs/tech/evaluation_criteria.md`: 0절 정정 이력 1건, 8.2절 실행 환경 Docker → Miniconda·Slurm (판정 기준과 8.1절 학습 설계는 그대로)
- `mydocs/manual/training_guide.md`, `README.md`: `gpu_train/` 링크
- `requirements.txt`: `huggingface_hub>=1.0`
- `.gitignore`: `gpu_train/site.env`, `/.cache/huggingface/`. `.gitattributes`: `*.sbatch`, `gpu_train/site.env*`, `inputs.sha256`, `inputs.hf` LF 고정
- `tests/test_gpu_train.py`: 17개

## 서버 smoke 결과 (2026-11-05, 작업지시자 실행)

| 항목 | 결과 |
|---|---|
| 서버 | moana, 계산 노드 `moana-u1`, NVIDIA RTX A2000 6138 MiB, 드라이버 535.216.03 |
| 환경 | torch 2.5.1+cu121, CUDA 12.1, `GPU 보임 True` |
| 코드 | `f39318a` |
| 입력 확인 | `입력 확인 통과: 13개` |
| 학습 | 시도 1회, step 50, `DONE` |
| 이어서 제출 | 다음 작업 153213 예약 → 완료 뒤 취소 (작업 기록) |
| baseline | top1 0.656·top3 0.896 정확히 같음. 소수 값 차이 최대 3.8 × 10⁻⁹ |
| 메모리 | `max_memory_gib` 4.57 (로컬과 같음), GPU 전체 최대 4,958 MiB |
| 처리량 | 10.4~14.0 국면/초 (로컬 약 25) |

상세와 원 출력은 `mydocs/test_result/task_m100_10_test.md`에 있다.

## 발견 사항

1. **서버 GPU 처리량이 로컬의 약 0.4~0.6배다.** 단순 비율로 학습 1회 약 7~9시간을 추정한다(로컬 3.1~3.6시간). 기준 문서 8.4절의 학습 자원 추정(10회 약 30~36시간)은 로컬 기준이라, 서버에서는 대략 두 배 이상 걸릴 수 있다. `SBATCH_TIME`이 짧아도 이어서 제출로 학습이 이어지므로 절차는 바뀌지 않는다. 실제 시간은 후속 이슈의 R1·R2·R8 결과로 확인한다.
2. **6 GB GPU에서도 학습이 돈다.** torch 최대 할당은 4.57 GiB로 로컬과 같았다. 여유는 약 1.2 GB다. 본 학습의 val 전체 평가에서 메모리가 더 드는지는 R1에서 본다. 넘치면 README 10절의 `--max-moves-per-forward` 조정을 따른다.
3. **baseline이 장비·torch 버전과 관계없이 같았다.** 기준선을 캐시된 q로 계산하므로 같아야 한다는 수행 계획서의 전제가 맞았다(차이 10⁻⁸ 이하). 이후 서버를 바꿔도 smoke baseline으로 입력과 코드가 같은지 확인할 수 있다.
4. **이 서버의 sbatch는 표준 오류로 안내 문구(`sbatch: MOANA: Job submitted`)를 낸다.** `submit.sh`는 작업 ID만 표준 출력에서 읽으므로 영향이 없었다.
5. **`inputs.sha256`의 주석 줄 경고가 없었다.** 단계1에서 오래된 coreutils를 우려했으나, 이 서버에서는 경고 없이 통과했다.
6. **서버 저장소의 `gpu_train/site.env.example`이 수정된 상태로 학습됐다.** 결과에는 영향이 없다. 다만 이후 `git pull` 충돌을 막으려면 서버 값은 `site.env`에 두고 예시 파일은 되돌려야 한다. 이 경우를 대비해 작업 기록에 `git status`를 남기도록 했는데, 그대로 드러났다.
7. **서버 권한 문제로 받지 못한 자료가 셋 있다.** `setup_env.sh` 출력(서버 pytest 결과), `fetch_inputs.sh` 출력(받은 리비전), `sacct` 회계 기록이다. 입력 내용은 `check_inputs.sh` 통과로, 다음 작업 예약·취소는 작업 기록으로 대신 확인했다. 서버 pytest 통과 여부는 확인하지 못했다.

## 검증

| 항목 | 결과 |
|------|------|
| 학습 정의 (성공 기준) | ✅ R1·R2·R8 명령이 기준 문서 8.1절 설정과 같다(테스트). R3~R7, R9, R10 거부 |
| 입력 확인 | ✅ 손상·누락 시 종료 코드 1, 작업은 종료 코드 2 (1단계, 로컬) |
| 로컬 smoke | ✅ Slurm 밖 실행, 강제 종료 뒤 재개(어댑터 SHA-256 일치), 결과 묶음 (3·4단계) |
| 서버 smoke | ✅ GPU 인식, 입력 확인, smoke 학습, 이어서 제출 예약·취소, 결과 회수. 서버 pytest와 sacct는 미확인 |
| 단위 테스트 (로컬) | 159 passed (기존 142 + 17, 5단계 시점) |
| 회귀 | 없음. `src/` 변경 없음 |

## 계획 대비 차이

| 항목 | 계획 | 실제 | 이유 |
|---|---|---|---|
| 단계 수 | 4단계 | 5단계 | 로컬에 `rsync`·`scp`·`ssh`가 없어 입력을 직접 옮길 수 없었다. Hugging Face 비공개 저장소 배포를 5단계로 추가했다 (작업지시자 결정, 2026-11-04) |
| devel 반영 시점 | 최종 보고서 뒤 | 4단계 뒤 선반영 + 최종 보고서 뒤 | 서버에서 `git clone -b devel`로 받기 위해서 (작업지시자 결정, 2026-11-04) |
| 3단계 동작 | 계획 1~7 | 실패로 멈출 때도 다음 작업 취소, SIGTERM 처리 추가 | 같은 실패의 반복과 시간 제한 직전의 헛된 재시도를 막기 위해서 |
| 4단계 회수 | 어댑터, 설정, 로그, 기록 | `pip freeze` 기록도 넣음 | 서버 패키지 버전을 결과와 함께 남기기 위해서. 이번 서버 묶음에는 `env_*.txt`가 없었다 (발견 사항 7) |
| 서버 smoke 자료 | 5가지 | 2가지 받음 (작업 기록, `SUMMARY.txt`) + 묶음 전체 | 서버 권한 문제 (작업지시자 확인, 2026-11-09) |
| 브랜치 | `local/task10` (`4524812`에서 분기) | 최종 보고서는 이 작업 사본에서 `devel` `f39318a`로 다시 만든 `local/task10`에서 커밋 | 이 작업 사본에는 처음 `local/task10`이 없었다 (작업지시자 승인, 2026-11-09). 내용은 `f39318a`까지 같다 |

## 후속

- 기준 문서 8.3절의 다음 이슈 "α = 0 학습"(R1·R2·R8)에서 서버 본 학습 결과를 정리한다. 작업지시자가 서버에서 R1·R2·R8을 이미 마쳤다고 알려 왔다 (2026-11-09). 결과 묶음은 그 이슈에서 다룬다.
- 서버의 `gpu_train/site.env.example` 수정을 되돌린다 (작업지시자, 서버).

## 이슈

- https://github.com/Chocobone/chess_transformer/issues/10
