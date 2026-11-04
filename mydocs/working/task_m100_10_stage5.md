# Task #10 단계5 완료 보고서: 입력 배포 (Hugging Face Hub 비공개 저장소)

4단계 뒤에 추가한 단계다 (작업지시자 결정, 2026-11-04). 로컬 개발 컨테이너에 `rsync`·`scp`·`ssh`가 없어서 README의 직접 전송을 여기서 실행할 수 없었다. 그래서 입력을 Hugging Face Hub 비공개 저장소로 배포한다.

## 수행 내용

- **`gpu_train/inputs.hf`** (커밋): 서버가 받을 저장소와 리비전. 토큰은 들어 있지 않다.

  ```bash
  HF_REPO=chocobone/player_chessengine
  HF_REVISION=5f48d15f01e4afce8e05bae8b5b50425a80fc368
  HF_TAG=inputs-20261104
  ```

- **`gpu_train/upload_inputs.sh 저장소ID`** (로컬): 다음을 차례로 한다.
  1. `check_inputs.sh`로 로컬 입력을 확인한다
  2. 저장소가 없으면 비공개로 만든다. 이미 있는데 공개면 멈춘다
  3. 13개 파일과 `inputs.sha256`을 커밋 하나로 올린다
  4. 태그 `inputs-{날짜}`를 붙인다. 같은 날 두 번째부터는 `-2`, `-3`을 붙인다
  5. `inputs.hf`를 채운다

  `huggingface_hub`의 Python API를 쓴다.
- **`gpu_train/fetch_inputs.sh`** (서버): 다음을 차례로 한다.
  1. `inputs.hf`가 비었거나 토큰이 없으면 멈춘다
  2. 리비전의 `inputs.sha256`을 git의 것과 비교한다. 다르면 멈춘다
  3. 매니페스트에 적힌 파일만 받는다(`snapshot_download`, `allow_patterns`)
  4. `check_inputs.sh`를 실행한다

  `site.env`에 `CONDA_SH`가 있으면 conda 환경을 활성화한다.
- **그 밖의 변경**:
  - `requirements.txt`: `huggingface_hub>=1.0` 추가. `src/`는 이 패키지를 쓰지 않는다
  - `.gitignore`: `/.cache/huggingface/` 추가
  - `.gitattributes`: `gpu_train/inputs.hf`를 LF로 고정
  - `check_inputs.sh`: 실패 안내 문구를 "fetch_inputs.sh를 다시 실행하거나 … (README 5절)"로 바꿨다
- **`gpu_train/README.md`**:
  - 절 순서를 바꿨다. 받기 스크립트가 환경 구성에서 설치되는 `huggingface_hub`를 쓰기 때문이다.

    | 바꾼 뒤 | 바꾸기 전 |
    |---|---|
    | 3절 서버 설정 | 4절 |
    | 4절 환경 구성 | 5절 |
    | 5절 입력 받기 | 3절 |

  - 5절의 내용은 다음과 같다.
    - Hugging Face 받기가 기본이다. 서버용 fine-grained 읽기 전용 토큰 만들기, `hf auth login`, `fetch_inputs.sh`를 적었다
    - 직접 전송(rsync)은 대안으로 남겼다
  - 1절 흐름도와 파일 표를 고쳤다.
  - 10절에 오류 4가지를 더했다: 토큰 없음, 401·403·Repository Not Found, Revision Not Found, 매니페스트 불일치.
  - 11절을 "R3~R10 추가와 입력 갱신"으로 바꾸고 갱신 순서 6단계를 적었다.
  - 절 번호 참조(6절의 "입력 파일(5절)", 10절)도 맞췄다.
- **링크**:
  - `training_guide.md` 2절 방법 A에 한 문장을 더했다.
  - `README.md` 문서 표의 설명을 고쳤다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/inputs.hf` | 신규 (업로드 결과로 채워짐) |
| `gpu_train/upload_inputs.sh`, `gpu_train/fetch_inputs.sh` | 신규 |
| `gpu_train/check_inputs.sh` | 실패 안내 문구 |
| `gpu_train/README.md` | 3~5절 재구성, 1·10·11절 |
| `requirements.txt`, `.gitignore`, `.gitattributes` | 한 줄씩 |
| `mydocs/manual/training_guide.md`, `README.md` | 한 문장씩 |
| `mydocs/working/task_m100_10_stage5.md` | 이 보고서 |

`src/` 변경은 없다.

## 검증

### 멈춰야 할 때 멈추는지 (토큰 없이)

| 경우 | 결과 |
|---|---|
| `fetch_inputs.sh`, `inputs.hf` 빈칸 | "HF_REPO나 HF_REVISION이 비어 있다", 종료 코드 1 |
| `fetch_inputs.sh`, 토큰 없음 | "Hugging Face 토큰이 없다", 종료 코드 1 |
| `upload_inputs.sh`, 인자 없음 / 토큰 없음 | 사용법 / "토큰이 없다", 종료 코드 1 |

### 업로드

- **업로드 전**: `chocobone/player_chessengine`이 공개 저장소였다(오늘 만든 기본 README와 `.gitattributes`만 있었다). 작업지시자가 비공개로 바꿨고, `private: True`를 확인한 뒤 올렸다.
- **`upload_inputs.sh chocobone/player_chessengine`**: 17초, 종료 코드 0.
  - 리비전 `5f48d15f01e4afce8e05bae8b5b50425a80fc368`, 태그 `inputs-20261104`
  - `inputs.hf`가 채워졌다.
- **원격 확인**: 13개 + `inputs.sha256`이 모두 있다. 원래 있던 README·`.gitattributes`는 그대로다.
  - 큰 파일 7개(LFS 저장)는 Hugging Face가 기록한 SHA-256이 매니페스트와 같았다.
  - 작은 파일 6개(val·stats)는 LFS가 아니라 SHA-256을 바로 볼 수 없다. 크기가 같은 것만 확인했고, 내용은 아래 받기 확인으로 확인했다.

### 받기 (저장소 사본, 서버 대신)

`git ls-files`로 추적·미추적 파일만 스크래치패드에 복사해 입력이 없는 사본을 만들었다. 시작 전 `check_inputs.sh`는 실패했다.

| 경우 | 결과 |
|---|---|
| 1. 로그인 없이 받기 (빈 `HF_HOME`) | 401 `RepositoryNotFoundError`로 거부. 비공개가 맞다 |
| 2. 빈 사본에서 `fetch_inputs.sh` | 19초, "받기 끝: 13개", **"입력 확인 통과: 13개"**, 종료 코드 0. 사본의 `README.md`는 Hugging Face README로 덮이지 않았다 |
| 3. `karpov/val.jsonl`을 지우고 다시 받기 | 통과. 수정 시각이 바뀐 파일은 그 파일 하나뿐이었다 (나머지 12개는 다시 받지 않음) |
| 4. `tal/stats.json`(git 저장) 1바이트 변경 후 받기 | 그 파일을 다시 받아 통과 |
| 5. `base_model/136M.pt`(LFS) 1바이트 변경 후 받기 | 그 파일을 다시 받아 통과 |
| 6. 사본의 `inputs.sha256`을 바꾼 뒤 받기 | "리비전의 inputs.sha256가 git의 것과 다르다", 종료 코드 1 |

- 시험이 끝난 사본은 지웠다.
- 전체 pytest **159 passed**, `bash -n` 통과, 새 파일 세 개 모두 `eol: lf`

## 발견 사항

1. **처음 로그인은 이 컨테이너가 아닌 곳에서 돼서 보이지 않았다.** 작업지시자가 프롬프트에서 `! hf auth login`으로 다시 로그인했다. 기기 인증(OAuth) 토큰이며 자동으로 갱신된다.
   - 이 토큰은 로컬 업로드에만 쓴다.
   - 서버에는 README 5절대로 그 저장소만 읽을 수 있는 fine-grained 토큰을 따로 만든다.
2. **지정한 저장소가 처음에는 공개였다.** 스크립트의 공개 확인보다 먼저, 업로드 전 사전 조회에서 발견해 멈췄다. 비공개로 바꾼 뒤 진행했다.
3. **파일을 바꿔도 받기가 다시 받아 고쳐 놓는다.** `snapshot_download`는 받은 뒤 수정된 파일을 감지해 다시 받는다. 수정 시각까지 되돌린 변경처럼 감지하지 못하는 경우가 있어도 마지막 `check_inputs.sh`에서 걸린다.
4. 업로드가 17초, 받기가 19초로 빨랐다. 내용이 같은 조각은 다시 보내지 않는 Hugging Face의 저장 방식 때문으로 보지만, 원인은 확인하지 않았다. 결과는 위 해시 비교와 받기 확인으로 검증했다.
5. **기준 문서 8.2절 "입력" 행의 요구 사항**(서버로 옮기는 방법과 체크섬 확인 방법을 적는다)은 이번 단계로 충족된다. 요구 사항 문구 자체는 바뀌지 않으므로 정정하지 않았다.

## 서버 smoke (작업지시자 실행)

README 2~6절 순서로 진행한다: 저장소 받기 → `site.env` → `setup_env.sh` → 토큰 로그인과 `fetch_inputs.sh` → `submit.sh smoke`.

다음을 `mydocs/test_result/task_m100_10_test.md`에 저장해 주시면 최종 보고서에서 확인한다.
- `setup_env.sh` 출력 끝부분 (torch·CUDA 버전, pytest 결과)
- `fetch_inputs.sh` 출력 끝부분 (리비전, "입력 확인 통과: 13개")
- `runs/gpu/meta/smoke_*.txt`
- `collect.sh smoke`의 `SUMMARY.txt`
- `sacct --name=smoke --format=JobID,State,Elapsed,ExitCode` 출력

## 커밋

`Task #10 단계5: 입력 배포 (Hugging Face 비공개 저장소)` (승인 후 커밋). 서버가 `git clone -b devel`로 받으므로, 승인 후 커밋과 함께 `devel`에도 반영해 push한다 (4단계 뒤 선반영과 같은 이유).
