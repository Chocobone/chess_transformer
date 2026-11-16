# Task #13 단계4 완료 보고서: 서버 실행 틀 (배포 전)

## 수행 내용

GPU 서버에서 대국과 C4 탐색을 돌리는 실행 틀을 만들었다. Task #12 덤프 틀(`submit_dump.sh`, `dump.sbatch`)과 같은 방식이다. `site.env`의 Slurm 값을 쓰고, `SBATCH_DRY_RUN=1`이면 제출하지 않고 sbatch 명령만 출력하며, 같은 명령을 다시 내면 이어서 돈다.

| 파일 | 하는 일 |
|---|---|
| `gpu_train/submit_match.sh` | 묶음 표기 `이름:모델:상대:노드수:오프닝`을 제출 전에 확인한다. 표기 형식, 이름 문자·중복, 오프닝 범위, 상대와 노드 수의 짝, 어댑터·Stockfish 존재를 본다. 그다음 `match.sbatch`를 제출한다. CPU 수는 `MATCH_CPUS`(없으면 `SBATCH_CPUS`) |
| `gpu_train/match.sbatch` | conda 활성화, Stockfish·베이스 확인 → 기록 → 묶음마다 `python -m src.match.play` → 결과 표. 기록에는 git HEAD, GPU, torch, CPU, Stockfish 버전·SHA-256·명령어 집합·bench가 들어간다. Stockfish 프로세스는 할당 CPU − 1개, 동시 대국은 64. GPU 사용량을 60초마다 남긴다 |
| `gpu_train/submit_quality.sh` | C4 작업을 제출한다. GPU를 쓰지 않아 기본으로 `--gres`를 붙이지 않는다(`QUALITY_GRES`로 붙일 수 있다). CPU 16개, 메모리 16G가 기본 |
| `gpu_train/quality.sbatch` | Stockfish·덤프 확인 → 기록 → `python -m src.match.quality`(깊이 18, 프로세스 = 할당 CPU) |
| `gpu_train/setup_env.sh` (기존 파일 변경) | requirements 설치 뒤 `bash gpu_train/install_stockfish.sh` 한 줄. 이어지는 pytest가 Stockfish 테스트까지 돈다 |
| `gpu_train/README.md` (기존 파일 변경) | 1절 그림·파일 표·결과 경로 표에 새 파일을 넣었다. 9절에 "대국" 절(설치 확인, 묶음 표기, 보정→기준선 순서, 예비 보정 명령, 회수)과 "C4 수 품질" 절을 더했다 |

구현 계획서의 기존 파일 변경은 1단계에서 `docker/Dockerfile`이 빠져 이 두 파일뿐이다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `gpu_train/submit_match.sh`, `gpu_train/match.sbatch` | 새 파일 |
| `gpu_train/submit_quality.sh`, `gpu_train/quality.sbatch` | 새 파일 |
| `gpu_train/setup_env.sh` | Stockfish 설치 한 줄 (+3줄) |
| `gpu_train/README.md` | 대국·C4 절 (+72줄) |
| `tests/test_gpu_match_scripts.py` | 새 파일. 테스트 24개 |

## 발견 사항

1. **로컬 끝까지 실행 (Slurm 밖, CPU)**: `match.sbatch e2e_n100:base:sf:100:0`(오프닝 0, 2국)이 끝까지 돌았고 종료 코드는 0이었다. 기록 파일에 Stockfish 19, SHA-256, 명령어 집합 `x86-64-avx512icl`, bench 2,497,913이 남았다. 결과는 베이스가 흑으로 진 대국(체크메이트, 195 ply)과 백으로 둔 대국(400 ply 한도 무승부, 392 ply)이었다. 표본 2국이라 판정에는 쓰지 않는다. 다만 **400 ply 한도에 닿는 대국이 실제로 나온다.** 모델이 이긴 국면을 마무리하지 못하는 경우로 보이며, C7(승리 전환율, 반복 비율)과 관련이 있다. 보정 결과에서 종료 사유 분포를 함께 본다.
2. **서버 GPU 대국 속도는 아직 모른다.** 로컬 CPU는 바퀴당 약 2.4~9초로 서버 추정에 쓸 수 없다. 예비 보정(100국)의 `runs.jsonl`에서 처음 잰다. 그 값으로 본 보정과 기준선 대국(약 2,800국)의 작업 시간과 `SBATCH_TIME`을 정한다.
3. 서버 CPU가 로컬과 다르면 universal 빌드가 다른 명령어 집합을 고른다. 작업 기록의 bench 값이 2,497,913인지 꼭 확인한다(README에 적었다).

## 검증

- `tests/test_gpu_match_scripts.py`: 24 passed ✅
  - 스크립트 6개 문법(`bash -n`), LF 줄 끝
  - `submit_match.sh` dry run: sbatch 옵션(작업 이름, 출력, chdir, 파티션, gres, 시간, CPU, 메모리, nodelist), `MATCH_CPUS`, 묶음 인자 전달
  - 잘못된 묶음 8가지 거부, 이름 중복 거부
  - `submit_quality.sh` dry run: `--gres` 없음, CPU 16·메모리 16G, `QUALITY_GRES`·`QUALITY_CPUS`, 모델 중복 거부
  - `quality.sbatch`: 덤프가 없으면 종료 코드 2
- 로컬 끝까지 실행: `match.sbatch` 1묶음 2국, 종료 코드 0 ✅
- 전체 pytest: **310 passed** (3단계 286 + 24), 0 failed ✅ (개발 컨테이너, CPU)

## 승인 요청: `devel` 배포

서버에서 `git pull`로 받으려면 `devel`에 반영해야 한다. Task #10·#11·#12와 같이, 최종 보고서 전에 다음을 하려 한다.

1. 이 단계 커밋 (`local/task13`)
2. `local/task13` → `local/devel` merge
3. `local/devel` → `devel` merge, `origin/devel` push

## 서버 실행 안내 (배포 뒤, 작업지시자)

```bash
cd <저장소>; git checkout devel; git pull
bash gpu_train/install_stockfish.sh        # Stockfish 19 설치. 출력의 bench가 2497913인지 확인
# site.env: SBATCH_EXTRA=--nodelist=moana-u1 (덤프와 같은 노드), SBATCH_TIME은 2시간 이상

# 예비 보정: 베이스 대 SF, 노드 수 5개 × 20국 (오프닝 0~9)
bash gpu_train/submit_match.sh cal1_n100:base:sf:100:0-9 cal1_n300:base:sf:300:0-9 cal1_n1000:base:sf:1000:0-9 \
  cal1_n3000:base:sf:3000:0-9 cal1_n10000:base:sf:10000:0-9

# 끝나면 묶어서 로컬로 가져온다
tar -czf runs/gpu/collect/match_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/match runs/gpu/logs/match_*.out
```

- 예비 보정은 100국이다. 소요 시간은 서버에서 처음 재는 값이다. 로컬 CPU 측정으로는 추정할 수 없어, `SBATCH_TIME`을 넉넉히(예: 4시간) 둔다. 끊기면 같은 명령으로 다시 내면 이어서 둔다
- 회수하면 5단계에서 `python -m src.eval.games calibrate`로 본 보정 후보 3개를 정해 다시 안내한다

## 다음 단계

5단계: 예비·본 보정 결과 회수, 노드 수 선택, 기준 문서 정정(오프닝 세트, Stockfish 버전, C2 노드 수, C4 깊이·승률 변환식, 소요 시간), 기준선 대국·C4 실행 안내.

## 커밋

`03d3199` Task #13 단계4: 서버 실행 틀
