# Task #10 단계2 완료 보고서: 환경 구성

## 수행 내용

- **`gpu_train/environment.yml`**: Python 3.13과 pip만 conda로 설치한다. 채널은 conda-forge만 쓰고 `nodefaults`로 defaults 채널을 뺀다. defaults 채널은 이용 약관 동의가 필요하기 때문이다.
- **`gpu_train/setup_env.sh`**: 로그인 노드에서 실행하며 다음 순서로 진행한다.
  1. `site.env`를 읽는다
  2. conda 환경이 없으면 만든다
  3. torch를 설치한다(`TORCH_INDEX_URL`이 있으면 그 주소에서)
  4. `requirements.txt`를 설치한다
  5. 버전을 출력하고 pytest 전체를 돌린다
  6. `pip freeze`를 `runs/gpu/meta/env_{시각}.txt`에 남긴다
- **`gpu_train/site.env.example`**: 서버 설정의 빈칸. 경로, conda, torch 휠 주소, Slurm 옵션(파티션, 계정, QoS, GPU, 시간, CPU, 메모리, 추가 옵션), 이어서 제출 상한이 있다. 항목마다 확인하는 명령(`conda info --base`, `nvidia-smi`, `sinfo`, `sacctmgr`)을 주석으로 달았다.
- **`.gitignore`**: `gpu_train/site.env`
- **`.gitattributes`**: `*.sbatch`, `gpu_train/site.env*`, `gpu_train/inputs.sha256`을 LF로 고정했다 (아래 발견 사항 1).

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/environment.yml` | 신규 |
| `gpu_train/setup_env.sh` | 신규 |
| `gpu_train/site.env.example` | 신규 |
| `.gitignore` | `gpu_train/site.env` 추가 |
| `.gitattributes` | LF 고정 3줄 추가 |
| `mydocs/working/task_m100_10_stage2.md` | 이 보고서 |

`src/` 변경은 없다.

## 발견 사항

1. **줄바꿈 위험을 막았다.**
   - 이 저장소의 `.git/config`에는 `core.autocrlf=true`가 있고, `.gitattributes`는 `*.sh`만 LF로 고정했었다. 그래서 1단계 커밋 때 "LF가 CRLF로 바뀐다"는 경고가 났다.
   - git 안에는 LF로 들어가므로 서버에서 새로 받으면 문제가 없다. 하지만 로컬에서 다시 체크아웃하면 다음 파일들이 CRLF가 될 수 있었다.
     - `.sbatch`: bash가 `$'\r'` 오류를 낸다
     - `site.env`: 값 끝에 `\r`이 붙는다
     - `inputs.sha256`: 경로 끝에 `\r`이 붙어 확인이 실패한다
   - 세 종류를 LF로 고정했고, `git check-attr eol`로 확인했다.
2. **setup_env.sh의 첫 실행이 pip 다운로드 해시 불일치로 멈췄다.**
   - `pip install --upgrade pip`에서 받은 pip 26.3 휠의 해시가 `1145e1…`로 나왔다. 같은 파일을 다시 받으니 PyPI 해시 `f3b60b…`와 같았다.
   - 로컬 네트워크의 일시적인 다운로드 손상으로 판단하고 스크립트는 고치지 않았다. 다시 실행하자 정상으로 끝났다.
   - 서버에서 같은 오류가 나면 다시 실행하면 된다. 이 내용을 4단계 README의 문제 해결에 적는다.
3. **로그인 노드에서는 GPU를 요구하지 않는다.** 로그인 노드에 GPU가 없을 수 있어서 GPU가 보이지 않아도 계속 진행한다. GPU 인식은 smoke 작업(3단계)이 확인한다.
4. **설치되는 버전(로컬 확인 기준)**: python 3.13.16, torch 2.15.0+cpu(CPU 휠로 시험), chess 1.11.2, numpy 2.5.4, pytest 9.2.0, fastapi 0.162.0, uvicorn 0.54.0, httpx 0.28.1. `requirements.txt`는 하한만 정하므로 서버에서 받는 버전은 다를 수 있다. 그래서 `pip freeze`를 남긴다.

## 검증

- **setup_env.sh 로컬 실행**: 로컬에는 conda가 없어서 `conda` 명령 대역을 스크래치패드에 만들어 실행했다. 대역은 `env list`, `env create`, `activate`를 venv로 흉내 낸다.
  - `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`로 휠 주소를 지정하는 경로를 시험했다.
  - 첫 실행: 환경 만들기 → pip 다운로드 해시 불일치로 종료 코드 1 (발견 사항 2)
  - 두 번째 실행: "환경 있음"으로 만들기를 건너뜀 → torch 설치 → requirements 설치 → 버전 출력 → "GPU 안 보임 (로그인 노드면 정상)" → **pytest 159 passed** → `pip freeze` 기록. 종료 코드 0
  - 실제 conda의 `env create`와 `activate`는 서버에서 확인한다.
- `bash -n gpu_train/setup_env.sh` 통과
- `git check-ignore -v gpu_train/site.env`: `.gitignore:18`에 걸린다
- `git check-attr eol`: `site.env.example`, `inputs.sha256`, `train.sbatch` 모두 `lf`
- 시험용 `gpu_train/site.env`는 지웠다. 시험 중 생긴 `runs/gpu/meta/env_20261104_095444.txt`(git 제외)는 남아 있다.

## 다음 단계

3단계: 작업 스크립트. `submit.sh`(sbatch 옵션 조합, 미리 보기 모드), `train.sbatch`(입력 확인, 기록, GPU 사용량, 재시도, 이어서 제출, 완료 표시, Slurm 밖 실행). 로컬 smoke, 강제 종료 뒤 재개 확인.

## 커밋

`Task #10 단계2: 환경 구성` (승인 후 커밋)
