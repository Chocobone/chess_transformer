# Task #1 단계0 완료 보고서: 개발 환경 설정

## 수행 내용

구현 계획서 승인 시 작업지시자가 지시한 대로 환경 설정을 0단계로 분리해 먼저 끝냈다.
로컬 개발 컨테이너 구성은 앞서 커밋했고(`8089a66`), 이번에 의존성 명세, git 제외 규칙,
패키지 골격, 테스트 실행 체계를 추가했다. 구현 계획서와 수행 계획서에도 0단계를 반영했다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `docker/Dockerfile`, `docker/compose.yml`, `.gitattributes` | 로컬 개발 컨테이너 (커밋 `8089a66`) |
| `requirements.txt` | `torch`, `chess`, `numpy`, `pytest` 최소 버전 명세 |
| `.gitignore` | `data/`, `runs/`, `checkpoints/`, `base_model/` 하위 체크포인트, 파이썬 캐시 제외 |
| `pytest.ini` | 테스트 경로 `tests/`, 저장소 루트를 import 경로에 추가 |
| `src/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py`, `src/train/__init__.py` | 패키지 골격 (빈 파일) |
| `tests/test_env.py` | 환경 점검 테스트 |
| `mydocs/plans/task_m100_1_impl.md` | 0단계 추가, 로컬 환경 설명 갱신, 단계 수 5단계로 변경 |
| `mydocs/plans/task_m100_1.md` | 예상 단계에 0단계 추가 |
| `mydocs/orders/20261007.md` | 작업일지 갱신 |

## 발견 사항

- 로컬 환경은 Python 3.13.16, torch 2.14.1+cpu, chess 1.11.2, numpy 2.5.3, pytest 9.1.1이다. CPU 20코어(torch 스레드 10), 메모리 15GB로 소형 모델 단위 테스트에는 충분하다.
- PyPI 패키지 이름은 `chess`다 (`python-chess`는 옛 이름). 계획서의 의존성 표기를 `chess`로 맞췄다.
- `Dockerfile`은 패키지를 직접 나열해 설치하며 `requirements.txt`를 읽지 않는다. 빌드 컨텍스트가 `docker/`라서 그렇다. 의존성을 추가할 때는 두 곳을 함께 고쳐야 한다.
- `base_model/`에는 `download.sh`만 있고 체크포인트는 없다. 2단계의 실제 가중치 검증은 계획대로 GPU 서버에서 수행한다.
- `scripts/` 폴더는 빈 폴더가 git에 남지 않으므로 파일이 처음 생기는 2단계에서 만든다.
- 저장소가 윈도우 호스트에서 bind mount 되어 있어 커밋 시 "LF will be replaced by CRLF" 경고가 나온다. `.gitattributes`는 `*.sh`만 LF로 고정하고 있어 다른 파일은 호스트 git 설정을 따른다. 동작에는 문제가 없다.

## 검증

- 패키지 import 및 버전 확인: ✅
- `pytest`: 6 passed, 0 failed (torch 순전파·역전파, 합법수 생성, 세 기사 PGN 첫 판 읽기, `src` 패키지 import)
- `git status`에 캐시(`__pycache__`, `.pytest_cache`)가 잡히지 않음: ✅

## 다음 단계

1단계: PGN 전처리 파이프라인 (`src/players.py`, `src/data/preprocess.py`, 단위 테스트, 세 기사 전처리 실행 및 `stats.json` 보고)

## 커밋

`Task #1 단계0: 개발 환경 설정` (해시는 오늘 할일 작업일지에 기재)
