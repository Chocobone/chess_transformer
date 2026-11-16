# Task #13 단계1 완료 보고서: Stockfish 연결, 승률 변환, 오프닝 세트

## 수행 내용

- Stockfish 버전을 **Stockfish 19**(`sf_19`, 2026-09-05, 최신 공식 릴리스)로 정했다. 공식 리눅스 x86-64 universal 빌드를 SHA-256을 확인해 받는 설치 스크립트를 만들었다.
- UCI 연결 모듈 `src/match/stockfish.py`를 만들었다. `Threads = 1`, `Hash = 16 MB`이고 매 탐색 전에 해시를 비운다(해석 3·8). 노드 수 제한 수(C2)와 고정 깊이 평가(C4, 오프닝 선별)를 제공하며, 고정 버전이 아니면 연결을 거부한다.
- 승률 변환 W(cp) = 1 / (1 + exp(−0.00368208 · cp))를 구현했다(결정 4). 메이트는 이기는 쪽 1, 지는 쪽 0이다.
- 오프닝 200개를 뽑아 `src/match/openings_200.json`에 저장했다.
- 깊이 18 탐색 시간을 쟀다.

### 오프닝 세트

| 항목 | 값 |
|---|---|
| 출처 | `official-stockfish/books` 커밋 `77ba28be`의 `noob_4moves.epd` (작업지시자 결정, 2026-11-16) |
| 라이선스 | CC0-1.0 (저장소 LICENSE). 커밋할 수 있다 |
| zip SHA-256 | `ca8e161f…e1bd9802` (코드에 고정, 다르면 거부) |
| 책 국면 | 1,837,365개 (FEN 앞 4필드 기준 중복 없음). 모두 양쪽 4수 뒤 백 차례, 수 번호 5 |
| 선별 | 깊이 18, 백 기준 \|cp\| ≤ 50, 메이트 제외, `default_rng(0)` 순열 순서 (해석 7) |
| 평가 | 214개 평가, 범위 밖 14개, 메이트 0개, 규칙 위반 0개 → 200개 (98.9초) |
| cp 분포 | −49 ~ +50, 평균 +7.0 |

### Stockfish 19 설치 정보 (로컬 개발 컨테이너)

| 항목 | 값 |
|---|---|
| 받은 파일 | `stockfish-linux-x86-64-universal.tar.gz`, SHA-256 `9defc0d4…5e03611f` (GitHub 릴리스 digest와 같음) |
| 실행 파일 | `runs/tools/stockfish/stockfish` → `release/stockfish/stockfish-linux-x86-64-universal` |
| 실행 파일 SHA-256 | `0f83d24c…4c12da19` |
| 선택된 명령어 집합 | `x86-64-avx512icl` (universal 빌드가 CPU에 맞춰 고른다) |
| bench 노드 수 | 2,497,913 (서버 대조용) |

### 깊이 18 탐색 시간 (로컬, 1스레드)

| 국면 | 횟수 | 평균 | 중앙값 | 최대 |
|---|---|---|---|---|
| 오프닝 선별 (8 ply 국면) | 214 | 0.46초 | — | — |
| `scripts/reference_fens.txt` 국면 (오프닝~종반 52개, 최선수 + 다른 수 1개 허용 탐색) | 105 | 0.53초 | 0.41초 | 4.29초 |

- C4 추정: S_val 13,846국면의 최선수 탐색에 모델 4개(베이스, R1·R8·R2)의 최선수가 아닌 1순위 수를 더하면 약 3만~4만 회다. 로컬 기준 약 4~6 CPU시간이고, 서버 CPU에서 16개 병렬이면 1시간 안팎으로 본다. 깊이 18을 그대로 쓸 수 있다고 판단한다 (서버 실측은 4단계)
- 같은 국면을 다른 국면 사이에 두고 다시 탐색해도 최선수가 같았다 (10/10). 해시를 비우는 방식이 결정성을 준다

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/install_stockfish.sh` | 새 파일. Stockfish 19 공식 바이너리 받기, SHA-256 확인, `runs/tools/stockfish/`에 풀기, 버전·명령어 집합·bench 출력 |
| `src/match/__init__.py` | 새 패키지 |
| `src/match/stockfish.py` | 새 파일. `cp_to_win`, `win_prob`, `find_binary`, `Stockfish`(`move`, `evaluate`) |
| `src/match/openings.py` | 새 파일. 책 받기(고정 커밋, SHA-256), EPD 읽기·중복 제거, 선별, `load_openings`, `parse_range`, CLI |
| `src/match/openings_200.json` | 새 파일. 오프닝 200개와 출처·버전·규칙·집계 메타데이터 |
| `tests/test_match_stockfish.py` | 새 파일. 테스트 12개 |

책 zip과 Stockfish 실행 파일은 git 제외 경로 `runs/tools/`에 있다.

## 발견 사항

1. **conda-forge에는 stockfish 패키지가 없다.** 결정 3의 권장안(conda-forge 설치)을 쓸 수 없어, 구현 계획서 위험표의 대안대로 공식 바이너리 설치 스크립트로 바꿨다. 설치 위치가 저장소 안(`runs/tools/`)이고 개발 컨테이너가 저장소를 마운트하므로, **`docker/Dockerfile`은 고칠 필요가 없어졌다.** 구현 계획서의 기존 파일 변경 3건 가운데 Dockerfile을 뺀다. 4단계의 `setup_env.sh` 변경은 이 스크립트를 부르는 한 줄로 줄어든다.
2. **`8moves_v3`는 양쪽 8수(16 ply) 세트였다.** 결정 2의 권장안은 기준 문서 5.3절(양쪽 4수, 8 ply)과 맞지 않았다. 작업지시자 결정으로 같은 저장소의 `noob_4moves.epd`(8 ply)로 바꿨다. 이 책은 수순 없이 FEN만 있어, 대국은 오프닝 FEN에서 시작하고 PGN `FEN` 헤더로 남긴다(2단계). 모델은 FEN만 보므로 수순이 없어도 모델 입력은 같다. 반복 판정은 오프닝 국면부터 센다.
3. **universal 빌드는 CPU에 맞춰 명령어 집합을 고른다.** 로컬은 AVX-512(ICL)를 골랐고, 서버 CPU는 다를 수 있다. Stockfish는 빌드와 무관하게 탐색이 같도록 만들어져 bench 노드 수가 같아야 한다. 4단계에서 서버 bench 노드 수가 2,497,913인지 대조한다.
4. 이 PC 호스트에는 Python 패키지가 없어, 모든 실행은 개발 컨테이너 이미지(`chess-transformer-dev`, torch 2.15.0, python-chess 1.11.2)를 `docker run --rm`으로 돌렸다. 대국 앱 테스트용 `fastapi` 등은 Task #11과 같이 일회용 컨테이너 안에서만 설치했다.

## 검증

- 새 테스트 `tests/test_match_stockfish.py`: 12 passed ✅
  - 승률 변환: cp 0 → 0.5, cp 100 → 0.5910(손 계산), 대칭, 단조, 관점 전환, 메이트 4종
  - EPD 읽기·중복 제거·수 정보 보충, zip 읽기, 규칙 확인, 선별(필터·재현·seed 의존·부족 시 오류), 번호 범위
  - 커밋한 오프닝 파일: 200개, 번호 순서, FEN 중복 없음, 합법·백 차례·수 번호 5, \|cp\| ≤ 50, 출처·라이선스·버전
  - Stockfish 실제 실행: 버전 확인, 같은 국면 같은 수(다른 국면을 사이에 둠), 1수 메이트의 최선수와 승률 1, 허용 수 탐색, 다른 버전 거부
- 전체 pytest: **255 passed** (기존 243 + 12), 0 failed ✅ (개발 컨테이너, CPU)
- 기존 `src/` 파일 변경 없음 ✅

## 다음 단계

2단계: 대국 실행기. `src/match/policy.py`(묶음 정책), `src/match/game.py`(종료 판정, 400 ply), `src/match/runner.py`(동시 대국, PGN, 이어서 실행)와 테스트. 로컬에서 베이스 136M 대 Stockfish 시험 대국으로 수당 시간을 잰다.

## 커밋

`fc227c6` Task #13 단계1: Stockfish 연결, 승률 변환, 오프닝 200개
