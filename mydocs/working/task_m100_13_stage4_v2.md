# Task #13 단계4 보완 보고서: 서버 glibc 문제와 Stockfish 정적 빌드

## 수행 내용

### 문제

작업지시자가 서버에서 `install_stockfish.sh`를 실행하자 공식 바이너리가 돌지 않았다 (2026-11-16).

```
runs/tools/stockfish/stockfish: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.35' not found
(GLIBC_2.32, 2.33, 2.34도 같음)
```

Stockfish 19 공식 리눅스 바이너리(`stockfish-linux-x86-64-universal`)는 glibc 2.35에 동적 링크되어 있다. 서버의 glibc는 2.32보다 오래되었다. 파일 SHA-256은 로컬과 같았다(받기와 확인은 정상).

### 해결 (작업지시자 결정: 로컬에서 정적 빌드 후 전송)

- **`gpu_train/build_stockfish.sh`** (새 파일, 로컬): 공식 릴리스 묶음(`sf_19`, 1단계와 같은 SHA-256)에 들어 있는 소스로 빌드한다.
  - Stockfish 스크립트(`make net`)가 NNUE 네트 `nn-1a298aa575a0.nnue`를 받아 SHA-256으로 확인한다
  - `make build ARCH=x86-64-avx2 COMP=gcc EXTRALDFLAGS=-static`으로 빌드한다
  - 정적 링크인지, 버전이 `Stockfish 19`인지, bench가 2,497,913인지 확인한다
- **`gpu_train/install_stockfish.sh`** (다시 씀): 정적 빌드 파일을 설치한다.
  - CPU가 AVX2·BMI1·POPCNT·SSE4.1을 지원하는지 확인한다
  - SHA-256이 `1d65ab6d…65784632`인지 확인하고 `runs/tools/stockfish/`에 설치·링크한다
  - bench가 2,497,913이 아니면 실패로 끝낸다
  - 로컬과 서버가 같은 파일(같은 SHA-256)을 쓴다
- **`gpu_train/README.md`**: Stockfish 설치 문단을 "로컬 빌드 → scp → 서버 설치"로 바꿨다.
- **`tests/test_gpu_match_scripts.py`**: `build_stockfish.sh` 문법·LF 확인과 설치 거부 테스트를 더했다. 설치 거부 테스트는 빈 저장소 사본에서 원본 파일이 없거나 SHA-256이 다르면 설치하지 않는지 본다.

### 정적 빌드 결과

| 항목 | 값 |
|---|---|
| 파일 | `runs/tools/stockfish-dist/stockfish-19-x86-64-avx2-static` (101.8 MB, 네트 포함) |
| SHA-256 | `1d65ab6d2cf0a35d89c5643588502aaf25458d4a10ca8774e940cd8865784632` |
| 링크 | 정적 (`ldd`: not a dynamic executable) |
| 컴파일러 | g++ 12.2.0 (개발 컨테이너, Debian 12) |
| ARCH | `x86-64-avx2` (AVX2, BMI1, POPCNT, SSE4.1) |
| 네트 | `nn-1a298aa575a0.nnue`, SHA-256 `1a298aa5…1e52dfc2` |
| 버전 | `id name Stockfish 19` |
| bench | 2,497,913 (공식 바이너리와 같다) |
| 재현 | 스크립트로 세 번 빌드해 세 번 모두 같은 SHA-256 |

### 공식 바이너리와 탐색이 같은지 확인

- bench 노드 수가 같다 (2,497,913)
- **오프닝 선별을 정적 빌드로 다시 돌렸다.** 깊이 18로 국면 214개를 평가한 결과, 고른 200개와 각 cp 값, 집계(범위 밖 14, 메이트 0)가 커밋한 `openings_200.json`과 완전히 같았다. 그래서 오프닝 파일은 그대로 둔다. 파일 메타데이터의 `binary_sha256`은 선별에 쓴 공식 바이너리(`0f83d24c…`) 값으로 남는다

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/build_stockfish.sh` | 새 파일. 공식 릴리스 소스로 정적 빌드 |
| `gpu_train/install_stockfish.sh` | 다시 씀. 정적 빌드 파일 설치(CPU·SHA-256·bench 확인) |
| `gpu_train/README.md` | Stockfish 설치 문단, 파일 표, 결과 경로 표 |
| `tests/test_gpu_match_scripts.py` | 스크립트 목록에 `build_stockfish.sh`, 설치 거부 테스트 |

`src/`는 바꾸지 않았다. 코드는 `runs/tools/stockfish/stockfish` 링크만 찾으므로 실행 파일이 바뀌어도 그대로다.

## 발견 사항

1. **서버 glibc가 2.32 미만이다.** 앞으로 서버에서 돌릴 외부 바이너리는 정적 링크로 준비해야 한다. torch는 conda/pip 휠이라 문제가 없었다.
2. **AVX2 빌드는 계산 노드 CPU가 AVX2를 지원해야 한다.** `install_stockfish.sh`는 로그인 노드 CPU만 확인한다. 계산 노드 `moana-u1`도 지원하는지 아래 명령으로 확인이 필요하다. 지원하지 않으면 `ARCH=x86-64-sse41-popcnt`로 다시 빌드한다(bench는 같아야 한다).
3. 로컬 개발 컨테이너도 이제 정적 빌드를 쓴다. 이후 로컬 테스트·시험은 서버와 같은 파일로 돈다.

## 검증

- 정적 빌드: 정적 링크, 버전, bench 2,497,913, SHA-256 재현 ✅
- 오프닝 선별 재실행: 200개·cp·집계가 커밋 파일과 같음 ✅
- 로컬 설치(`install_stockfish.sh`): SHA-256 확인, CPU 확인, bench 2,497,913 ✅
- 전체 pytest: **313 passed** (4단계 310 + 3), 0 failed ✅ (개발 컨테이너, CPU, 정적 빌드로 Stockfish 테스트 포함)

## 승인 요청

1. 이 보완을 `local/task13`에 커밋하고, `devel`에 다시 반영해 `origin/devel`에 push한다.
2. 서버 실행 (작업지시자):

```bash
# 계산 노드 CPU 확인 (avx2 bmi1 popcnt sse4_1 네 줄이 모두 나와야 한다)
srun --nodelist=moana-u1 bash -c "grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -x -E 'avx2|bmi1|popcnt|sse4_1'"

# 로컬 → 서버: 정적 빌드 파일 전송 (로컬 파일 경로: runs/tools/stockfish-dist/stockfish-19-x86-64-avx2-static)
ssh <서버> mkdir -p <저장소>/runs/tools/stockfish-dist
scp runs/tools/stockfish-dist/stockfish-19-x86-64-avx2-static <서버>:<저장소>/runs/tools/stockfish-dist/

# 서버
git pull
bash gpu_train/install_stockfish.sh        # bench: 2497913 확인
bash gpu_train/submit_match.sh cal1_n100:base:sf:100:0-9 cal1_n300:base:sf:300:0-9 cal1_n1000:base:sf:1000:0-9 \
  cal1_n3000:base:sf:3000:0-9 cal1_n10000:base:sf:10000:0-9
```

## 커밋

(승인 뒤 기록)
