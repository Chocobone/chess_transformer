# GPU 서버 학습 (Slurm + Miniconda)

평가 기준 문서(`mydocs/tech/evaluation_criteria.md` 8.1절)의 학습 R1~R10을 Slurm GPU 서버에서 돌리는 절차다. 모든 명령은 **서버의 저장소 루트**에서 실행한다. 학습 코드(`src/train/train.py`)와 옵션의 뜻은 `mydocs/manual/training_guide.md`에 있다.

## 1. 개요

```
로컬                                서버 (로그인 노드)                   서버 (계산 노드, sbatch)
────                                ──────────────────                   ────────────────────────
upload_inputs.sh ─→ Hugging Face (비공개)
                         └──────→   site.env 채우기
                                    setup_env.sh (conda env, pytest)
                                    fetch_inputs.sh (받기, check_inputs.sh)
                                    submit.sh smoke  ─────────────→      train.sbatch smoke
                                    submit.sh R1 / R2 / R8  ──────→      train.sbatch R1 …
                                    collect.sh R1
                                    submit_eval.sh runs/gpu/R1 … ─→      eval.sbatch (재평가)
결과 묶음 scp  ←──────────────
mydocs/test_result/에 정리
```

| 파일 | 하는 일 |
|---|---|
| `runs.py` | 학습 ID(R1~R10, seed 0 대조군 R1s0·R2s0·R8s0, smoke)별 설정. `python -m gpu_train.runs list`로 목록을 본다 |
| `inputs.sha256`, `check_inputs.sh` | 입력 13개의 SHA-256과 확인 스크립트 |
| `make_manifest.sh` | `inputs.sha256`을 다시 만든다 (입력을 바꿨을 때만, 로컬에서) |
| `inputs.hf` | 입력을 받을 Hugging Face 저장소 ID와 고정 커밋 해시 |
| `upload_inputs.sh` | 로컬에서 입력을 Hugging Face 비공개 저장소에 올리고 `inputs.hf`를 채운다 |
| `fetch_inputs.sh` | 서버에서 `inputs.hf`의 리비전을 받고 `check_inputs.sh`로 확인한다 |
| `environment.yml`, `setup_env.sh` | conda 환경 (Python 3.13). torch와 `requirements.txt`는 pip로 설치한다 |
| `site.env.example` | 서버마다 다른 값의 빈칸. `site.env`로 복사해 채운다 (git 제외) |
| `submit.sh` | 학습 하나를 sbatch로 제출한다 |
| `train.sbatch` | 작업 본문: 입력 확인, 기록, 학습, 재시도, 다음 작업 이어서 제출 |
| `collect.sh` | 결과를 tar.gz 하나로 묶는다 |
| `eval_adapter.py` | 학습 폴더 하나의 어댑터로 val 전체를 다시 재서 `log.jsonl`의 마지막 eval·baseline과 비교한다 (9절 재평가) |
| `submit_eval.sh`, `eval.sbatch` | 학습 폴더 여러 개의 재평가를 sbatch 작업 하나로 제출한다 |

결과는 모두 `runs/gpu/` 아래에 쌓인다 (git 제외).

| 경로 | 내용 |
|---|---|
| `runs/gpu/{ID}/` | `adapter.pt`, `checkpoint.pt`, `config.json`, `log.jsonl`, `DONE` |
| `runs/gpu/logs/{ID}_{작업ID}.out` | 작업의 표준 출력·오류 |
| `runs/gpu/logs/gpu_{ID}.log` | 60초마다 GPU 메모리(MiB)·사용률·온도 |
| `runs/gpu/meta/{ID}_{작업ID}.txt` | 작업 기록: 호스트, 커밋, GPU, torch, 명령, 시도별 결과 |
| `runs/gpu/meta/env_{시각}.txt` | `setup_env.sh`가 남긴 `pip freeze` |
| `runs/gpu/collect/` | `collect.sh`가 만든 묶음 |
| `runs/gpu/eval/` | 재평가 결과: `{폴더 이름}.json`·`.txt`, 기록 `meta_{작업ID}.txt`, GPU 로그 `gpu_{작업ID}.log` |

## 2. 저장소 받기

```bash
git clone -b devel https://github.com/Chocobone/chess_transformer.git
cd chess_transformer
git log --oneline -1        # 학습할 커밋을 확인해 둔다. 작업 기록에도 남는다
```

- 계산 노드에서도 같은 경로로 보이는 곳(공유 파일 시스템)에 받는다. 이 경로를 `site.env`의 `REPO_DIR`에 적는다.
- 학습 도중에는 `git pull`을 하지 않는다. 이어서 제출된 작업이 바뀐 코드로 돌게 된다. 학습마다 작업 기록에 `git HEAD`와 미커밋 변경이 남는다.

## 3. 서버 설정 (`site.env`)

```bash
cp gpu_train/site.env.example gpu_train/site.env
```

빈칸을 채운다. 항목마다 확인하는 명령이 주석에 있다.

| 항목 | 내용 | 확인 |
|---|---|---|
| `REPO_DIR` | 저장소 절대 경로. `submit.sh`는 이 값이 실행한 저장소와 다르면 제출하지 않는다 | `pwd -P` |
| `CONDA_SH` | `<conda base>/etc/profile.d/conda.sh` | `conda info --base` |
| `CONDA_ENV` | 환경 이름 (기본 `chess-train`) | |
| `TORCH_INDEX_URL` | 비우면 PyPI 기본 torch 휠. 드라이버가 오래되어 기본 휠이 안 돌면 `https://download.pytorch.org/whl/cuXXX` | 계산 노드의 `nvidia-smi` 오른쪽 위 "CUDA Version" |
| `SBATCH_PARTITION`, `SBATCH_ACCOUNT`, `SBATCH_QOS` | 비우면 옵션을 붙이지 않는다 | `sinfo -s`, `sacctmgr -n show assoc user=$USER format=account,partition,qos` |
| `SBATCH_GRES` | GPU 요청 (기본 `gpu:1`) | `sinfo -o "%P %G"` |
| `SBATCH_TIME` | 작업 하나의 최대 시간. 파티션 한도 이하 | `sinfo -o "%P %l"` |
| `SBATCH_CPUS`, `SBATCH_MEM` | 기본 4개, 32G. 데이터 로더 worker가 2개다 | |
| `SBATCH_EXTRA` | 그 밖의 sbatch 옵션, 공백으로 구분 (예: `--constraint=a100`) | |
| `CHAIN_MAX` | 학습 하나에서 이어서 제출하는 작업 수 상한 (첫 작업 포함, 기본 10) | |

- `site.env`는 bash가 `source`하는 파일이다. `=` 앞뒤에 공백을 넣지 않는다.
- 값 앞에 `export`를 붙이지 않는다. `SBATCH_PARTITION` 같은 이름은 sbatch가 직접 읽는 환경변수 이름과 같다. export하지 않으면 `submit.sh`가 옵션으로 바꿔 붙인 값만 쓰인다.
- 설정이 맞는지는 제출하지 않고 명령만 출력해 본다:

  ```bash
  SBATCH_DRY_RUN=1 bash gpu_train/submit.sh smoke
  ```

## 4. 환경 구성

로그인 노드에서 실행한다. pytest는 입력 파일 없이도 돈다.

```bash
bash gpu_train/setup_env.sh
```

1. 환경 `CONDA_ENV`가 없으면 `environment.yml`로 만든다 (conda-forge, Python 3.13). 있으면 만들지 않는다
2. pip로 torch를 설치하고(`TORCH_INDEX_URL`이 있으면 그 주소에서), 이어서 `requirements.txt`를 설치한다. 입력 받기에 쓰는 `huggingface_hub`도 여기서 설치된다
3. 버전을 출력하고 `pytest -q` 전체를 돌린다
4. `pip freeze`를 `runs/gpu/meta/env_{시각}.txt`에 남긴다

- 끝에 `N passed`, 종료 코드 0이면 된다.
- 로그인 노드에 GPU가 없으면 "GPU 안 보임 (로그인 노드면 정상)"이 나온다. GPU 인식은 smoke 작업이 확인한다.
- 다시 실행해도 된다. 환경은 그대로 두고 설치와 확인만 다시 한다.

## 5. 입력 받기

베이스 가중치와 전처리 데이터·q 캐시는 git에 없다. 13개 파일, 660,418,156 bytes(약 630 MiB)다. 목록과 SHA-256은 `gpu_train/inputs.sha256`에 있다.

### Hugging Face에서 받기 (기본)

입력은 Hugging Face Hub의 **비공개** 저장소에 있다. `gpu_train/inputs.hf`에 저장소 ID(`HF_REPO`)와 받을 커밋 해시(`HF_REVISION`)가 고정돼 있다.

**토큰** (처음 한 번):

1. huggingface.co → Settings → Access Tokens → **Fine-grained** 토큰을 만든다
2. 권한은 그 저장소(`HF_REPO`)의 **Read access to contents** 하나만 준다. 쓰기 권한은 주지 않는다
3. 서버에서 로그인한다. 토큰은 `~/.cache/huggingface/token`에 저장되고, 저장소에는 남지 않는다

```bash
conda activate chess-train      # 4절의 환경. hf 명령이 여기 들어 있다
hf auth login                   # 토큰을 붙여 넣는다. git 자격 증명 저장은 묻지 않으면 건너뛴다
hf auth whoami                  # 계정 이름이 나오면 된다
```

`hf auth login` 대신 `HF_TOKEN` 환경변수를 써도 된다. 이때도 토큰을 `site.env`나 저장소 안 파일에 적지 않는다.

**받기**:

```bash
bash gpu_train/fetch_inputs.sh
```

1. `inputs.hf`의 리비전에서 `inputs.sha256`을 먼저 받아 git의 `inputs.sha256`과 비교한다. 다르면 멈춘다
2. 매니페스트에 적힌 파일만 저장소 루트의 같은 경로로 받는다. 이미 받은 같은 파일은 다시 받지 않는다
3. `check_inputs.sh`를 돌린다. 끝에 `입력 확인 통과: 13개`가 나와야 한다

- 받는 중 생기는 `.cache/huggingface/`(저장소 루트)는 git 제외다. 다음 받기에서 같은 파일인지 판단하는 데 쓰이므로 지우지 않는다.
- **입력이 갱신되면**: 로컬에서 새로 올리고 `inputs.hf`를 커밋한다 (11절, 처음 올리기도 같다). 서버에서는 학습이 돌고 있지 않을 때 `git pull` 후 `fetch_inputs.sh`를 다시 실행한다.

### 직접 전송 (대안)

로컬에서 서버로 ssh가 되면 Hugging Face 없이 옮길 수 있다. ssh가 되는 머신의 저장소 루트에서 실행한다 (로컬 개발 컨테이너에는 `rsync`·`scp`·`ssh`가 없다).

```bash
# -R은 상대 경로를 그대로 유지한다
rsync -avR --progress \
  base_model/136M.pt \
  data/processed/{tal,karpov,smyslov}/{train.jsonl,val.jsonl,stats.json,base_q_136M.npz} \
  사용자@서버:/서버의/저장소/경로/
```

rsync가 없으면 같은 상대 경로로 `scp`한다. 옮긴 뒤 서버에서 `bash gpu_train/check_inputs.sh`로 확인한다.

### 확인

```bash
bash gpu_train/check_inputs.sh      # "입력 확인 통과: 13개"가 나와야 한다
```

하나라도 다르거나 없으면 그 파일 이름이 나오고 종료 코드 1로 끝난다. `fetch_inputs.sh`를 다시 실행하거나 그 파일을 다시 옮긴다. `train.sbatch`도 학습 전에 이 확인을 하고, 실패하면 종료 코드 2로 끝난다.

## 6. smoke (본 학습 전에 반드시)

탈 데이터로 50 갱신, 평가는 val 앞쪽 500국면만 한다. 로컬 RTX 4070 Ti에서 2분 걸렸다.

```bash
bash gpu_train/submit.sh smoke
squeue -u $USER                       # 대기·실행 중인지 본다
cat runs/gpu/logs/smoke_*.out         # 끝나면
```

확인할 것:

| 항목 | 어디서 | 기대 |
|---|---|---|
| 입력 확인 | 작업 출력 첫 줄 | `입력 확인 통과: 13개` |
| GPU | `runs/gpu/meta/smoke_*.txt`의 `GPU:`, `torch:` 줄 | GPU 이름이 나오고 `GPU 보임 True` |
| 완료 | 작업 출력 끝, `runs/gpu/smoke/DONE` | `smoke 완료 (시도 1회, step 50)` |
| 다음 작업 | 작업 출력, `sacct --name=smoke` | `다음 작업 … 예약` 뒤 `다음 작업 … 취소`. sacct에 두 번째 작업이 `CANCELLED`로 남는다 |
| baseline | `runs/gpu/smoke/log.jsonl` 첫 줄 | 아래 기준값과 같아야 한다 |
| 메모리 | `log.jsonl`의 `max_memory_gib`, `gpu_smoke.log` | 로컬은 4.57 GiB, GPU 전체 최대 6,270 MiB |

**baseline 기준값** (로컬 RTX 4070 Ti, torch 2.14.1+cu130). 베이스 모델로 val 앞쪽 500국면을 평가한 값이라 학습 설정·seed와 관계없다:

| 항목 | 값 |
|---|---|
| imitation | 1.3561140506193041 |
| top1 | 0.656 |
| top3 | 0.896 |
| regret | 0.011584872650681064 |
| band_violation | 0.2269073169492185 |
| strength | 0.005139145441353321 |

- top1·top3는 정확히 같아야 한다. 나머지는 GPU·torch 버전에 따라 끝자리가 다를 수 있다. 소수 넷째 자리까지 같으면 정상이다.
- 다르면 입력 파일(5절)이나 코드 커밋(2절)이 다르다. 본 학습을 시작하지 않는다.
- 로컬 smoke 어댑터의 SHA-256은 `806fc7ff6923b7a5fa590252100898af16c731d133e3be2b8df2a0a7b1fe633f`다. 다른 GPU에서는 달라도 된다. 참고로만 적는다.

smoke를 다시 돌리려면 `runs/gpu/smoke/`를 지운다. `DONE`이 있으면 작업이 바로 끝난다.

## 7. 본 학습

지금 실행할 수 있는 것은 R1(탈)·R2(스미슬로프)·R8(카르포프), α = 0, seed 1과, 그 seed 0 대조군 R1s0·R2s0·R8s0이다. R3~R7, R9, R10은 α가 정해지기 전에는 제출을 거부한다 (11절).

- R1s0·R2s0·R8s0은 대응하는 R1·R2·R8과 seed(0)만 다르다. 잡음 바닥(기준 문서 4.3절 B3) 쌍으로 쓴다 (Task #11). 기존 seed 0 어댑터 `runs/{player}_136M/`은 Task #5 수정 전 코드로 로컬에서 학습한 것이라 쓰지 않는다

```bash
python -m gpu_train.runs list        # ID, 기사, α, λ, seed, 실행 가능 여부
bash gpu_train/submit.sh R1
bash gpu_train/submit.sh R2
bash gpu_train/submit.sh R8
bash gpu_train/submit.sh R1s0         # seed 0 대조군. R2s0, R8s0도 같은 방식
```

- 세 학습은 출력 폴더와 데이터가 서로 달라 동시에 제출해도 된다. 각각 GPU 1개를 쓴다. GPU 할당이 빠듯하면 하나씩 제출한다.
- 학습 1회는 로컬 RTX 4070 Ti 기준 약 3.1~3.6시간이다 (7,455~9,039 갱신, 평가 1회 약 3분 × 8~10회). 서버 GPU에 따라 다르다. smoke의 `positions_per_sec`로 짐작한다 (로컬은 약 25).
  - 서버 실측 (RTX A2000 6 GB, Task #11): 탈 7시간 56분, 스미슬로프 9시간 19분, 카르포프 10시간 4분. 처리량 약 10.5~10.8국면/초, 평가 1회 12~16분, GPU 전체 최대 5,020 MiB. 결과는 `mydocs/test_result/task_m100_11_test.md`
- **`SBATCH_TIME`이 학습 시간보다 짧아도 된다.** 작업은 시작할 때 같은 학습의 다음 작업을 `--dependency=afterany`로 예약해 둔다. 시간 제한으로 끊기면 다음 작업이 체크포인트에서 이어 간다.
  - 체크포인트는 200 갱신(로컬 약 4~5분)마다 저장된다. 끊길 때마다 최대 그만큼과 시작 시간을 잃는다.
  - 학습이 끝나면 예약해 둔 다음 작업을 취소한다.
  - 다음 작업은 시작할 때 체크포인트가 앞 작업 동안 전진했는지 본다. 전진하지 않았으면 더 예약하지 않고 멈춘다.
  - 이어서 제출은 `CHAIN_MAX`개에서 멈춘다.
- 학습 도중 오류(예: 원인 불명 CUDA 오류)가 나면 같은 작업 안에서 20초 뒤 체크포인트에서 다시 시작한다. 최대 10회까지 시도하고, 체크포인트가 전진하지 않은 채 연속 3회 실패하면 멈춘다. 멈출 때는 예약한 다음 작업도 취소한다.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`는 `train.sbatch`가 붙인다 (학습 결과는 바꾸지 않는다, 학습 가이드 6절).
- 같은 이름의 작업이 이미 대기 중이거나 실행 중이면 `submit.sh`가 제출하지 않는다.

## 8. 상태 확인과 취소

```bash
squeue -u $USER                                      # 실행 중(R)과 이어서 대기 중(PD, Dependency)
tail -f runs/gpu/logs/R1_<작업ID>.out                # 진행 (train 줄은 20 갱신마다)
grep '"eval"' runs/gpu/R1/log.jsonl | tail -1        # 마지막 평가
grep '^===' runs/gpu/meta/R1_*.txt                   # 작업별 시도·예약·완료 기록
sacct --name=R1 --format=JobID,State,Elapsed,ExitCode,NodeList
```

- **학습 하나를 멈출 때는 이름으로 취소한다: `scancel -n R1 -u $USER`.** 실행 중인 작업 번호만 취소하면 `afterany`로 예약된 다음 작업이 시작되어 학습을 이어 간다.
- 멈춘 학습을 다시 시작할 때는 같은 명령(`submit.sh R1`)으로 제출한다. 항상 `--resume`이 붙어 있어 체크포인트에서 이어 간다.
- 처음부터 다시 하려면 `runs/gpu/R1/`을 지우고 제출한다.

## 9. 결과 회수

서버에서:

```bash
bash gpu_train/collect.sh R1          # → runs/gpu/collect/R1_{시각}.tar.gz
```

- 넣는 것: `adapter.pt`, `config.json`, `log.jsonl`, `DONE`, 작업 출력(`R1_*.out`), GPU 로그, 작업 기록(`R1_*.txt`), 최신 `pip freeze`, `SUMMARY.txt`
- `checkpoint.pt`(재개용)는 넣지 않는다.
- `DONE`이 없으면 경고하고 묶는다. 중간 확인용이다.
- `SUMMARY.txt`에 들어가는 것:
  - 완료 표시, `adapter.pt` 크기와 SHA-256
  - baseline 줄, 마지막 eval·train 줄
  - 재개 횟수와 step
  - 작업별 시도 기록

로컬로 가져온다. 저장소 루트에 바로 풀지 않는다:

```bash
scp 사용자@서버:/서버의/저장소/경로/runs/gpu/collect/R1_*.tar.gz runs/gpu/collect/
mkdir -p runs/gpu/collect/R1 && tar -xzf runs/gpu/collect/R1_*.tar.gz -C runs/gpu/collect/R1
```

`mydocs/test_result/task_m100_{N}_test.md`에 정리한다. `{N}`은 그 학습을 맡은 이슈 번호다. 정리할 항목은 학습 가이드 7절과 같다.

- 실행 명령 전체와 환경변수, GPU 종류, 코드 커밋 → 작업 기록 파일
- 시도별 시작·종료 시각, 종료 코드, 재개 step → 작업 기록의 `===` 줄. 중단 지점은 작업 출력에서 실패한 시도의 마지막 `train` 줄로 적는다
- `config.json`
- baseline 줄, 모든 eval 줄, 마지막 train 줄 몇 개
- 처리량, `max_memory_gib`, GPU 전체 사용량 최대(MiB와 ÷1024 GiB), 평가 1회 시간
- `adapter.pt` 크기와 SHA-256
- 이어서 제출한 작업 목록 (`sacct --name=R1` 출력)
- 오류가 났다면 오류 메시지 전문

### 재평가 (`submit_eval.sh`)

받은 어댑터가 학습 끝의 가중치 그대로인지 확인한다. 학습 폴더마다 val 전체를 다시 재서, 그 폴더 `log.jsonl`의 마지막 eval 줄과 baseline 줄과 비교한다. 평가 조건(데이터, 손실 설정, 묶음 크기, float32)은 폴더의 `config.json`에서 읽어 학습 때와 같게 맞춘다.

```bash
SBATCH_DRY_RUN=1 bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8   # 명령만 확인
bash gpu_train/submit_eval.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8
squeue -u $USER                                   # 작업 이름 eval
tail -f runs/gpu/logs/eval_<작업ID>.out
```

- 폴더를 차례로 평가한다. 서버(RTX A2000)에서 폴더 하나에 12~16분, 셋에 약 45분이다. `SBATCH_TIME`이 이보다 짧으면 끊긴다. 재평가는 이어서 제출하지 않는다.
- 학습과 GPU를 따로 받으므로 학습 중에 제출해도 된다.
- 판정은 서버 smoke(6절)와 같다. **top1·top3는 정확히 같고, 나머지 지표는 차이가 5 × 10⁻⁵ 미만**이어야 통과다.
- 하나가 실패해도 나머지는 평가한다. 작업 종료 코드는 모두 통과면 0, 하나라도 실패나 오류면 1, 입력 확인 실패면 2다.
- 결과는 `runs/gpu/eval/`에 쌓인다.

| 파일 | 내용 |
|---|---|
| `{폴더 이름}.json` | 지표별 다시 잰 값, 기록값, 차이, 판정. `adapter.pt` SHA-256, 장치, torch 버전, 걸린 시간 |
| `{폴더 이름}.txt` | 사람이 읽는 출력. 마지막 줄이 `판정: 통과` 또는 `판정: 실패` |
| `meta_{작업ID}.txt` | 대상, 호스트, git HEAD와 미커밋 변경, GPU, torch, 폴더별 시작·끝, 마지막의 판정 표 |
| `gpu_{작업ID}.log` | 60초마다 GPU 메모리·사용률·온도 |

- 판정 표에서 `실패`는 값이 다르다는 뜻이다. `오류(종료 코드 N)`는 평가가 끝나지 않았다는 뜻이다(파일 없음, 입력 불일치 등). 해당 `.txt`에 원인이 있다.
- 같은 폴더 이름(`runs/gpu/R1`과 `runs/R1` 등)은 결과 파일이 겹치므로 함께 제출할 수 없다.
- 다시 돌리면 같은 이름의 결과 파일을 덮어쓴다. 기록 파일은 작업마다 따로 남는다.
- 서버 밖(로컬)에서 직접 돌릴 수도 있다: `python -m gpu_train.eval_adapter runs/gpu/R1 --baseline --json 결과.json`. 입력(`data/processed/`)과 베이스가 있어야 한다.

로컬로 가져올 때는 묶어서 옮긴다:

```bash
tar -czf runs/gpu/collect/eval_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/eval runs/gpu/logs/eval_*.out
```

## 10. 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| `입력 확인 실패`, 작업 종료 코드 2 | 입력 파일이 다르거나 없다. `fetch_inputs.sh`를 다시 실행하거나 그 파일을 다시 옮긴 뒤 `check_inputs.sh`로 확인한다 (5절) |
| `Hugging Face 토큰이 없다` | 서버에서 `hf auth login`을 하지 않았다 (5절) |
| `401`, `403`, `Repository Not Found` (받기) | 토큰이 그 저장소를 읽을 수 없다. 비공개 저장소는 권한이 없어도 "없다"로 나온다. fine-grained 토큰의 저장소 권한과 `inputs.hf`의 `HF_REPO`를 확인한다 |
| `Revision Not Found` | `inputs.hf`의 `HF_REVISION`이 저장소에 없다. 로컬에서 올린 뒤 `inputs.hf`를 커밋·push했는지, 서버에서 `git pull`했는지 본다 |
| `…의 gpu_train/inputs.sha256가 git의 …와 다르다` | `inputs.hf`와 `inputs.sha256`이 서로 다른 커밋에서 왔다. 로컬에서 `upload_inputs.sh`를 다시 실행하고 두 파일을 함께 커밋한다 |
| `setup_env.sh`에서 `THESE PACKAGES DO NOT MATCH THE HASHES` | 내려받는 중 파일이 손상됐다 (로컬 시험에서 한 번 났다). 그대로 다시 실행한다. 되풀이되면 프록시·미러 설정을 확인한다 |
| `conda 환경 chess-train 활성화 실패` | `CONDA_SH` 경로가 틀렸거나 환경을 아직 만들지 않았다. `setup_env.sh`를 먼저 실행한다 |
| `site.env의 REPO_DIR(…)가 이 저장소(…)와 다르다` | `REPO_DIR`를 `pwd -P` 값으로 고친다 |
| `이미 대기 중이거나 실행 중이다` | 같은 학습이 이미 있다. `squeue -u $USER`로 확인한다 |
| `R3의 α가 아직 없다` | α가 정해지지 않은 학습이다 (11절) |
| 작업이 `PD`에서 움직이지 않음 | `squeue`의 사유(`Resources`, `Priority`, `QOSMaxJobsPerUserLimit` 등)를 본다. 파티션·QoS·`SBATCH_TIME` 한도를 확인한다 |
| `다음 작업 예약 실패` | 이 서버가 작업 안에서의 제출을 막았을 수 있다. 작업 출력의 sbatch 오류를 본다. 막혀 있으면 `SBATCH_TIME`을 학습 1회보다 길게 잡거나, 끊길 때마다 `submit.sh`로 다시 제출한다 |
| `앞 작업 동안 체크포인트가 전진하지 않았다` | 앞 작업이 체크포인트를 남기기 전에 끝났다. 앞 작업의 출력과 기록에서 이유를 찾는다. 예: 시작 직후 오류, `SBATCH_TIME`이 너무 짧음. 고친 뒤 다시 제출한다 |
| `CUDA out of memory` | 로컬 12GB에서는 GPU 전체 최대가 약 6.3 GiB였다. 이보다 작은 GPU면 `runs.py`의 그 학습 `extra`에 `--max-moves-per-forward`를 낮춰 넣는다. 주의할 점이 셋 있다. ① 처음 값과 다르면 재개가 거부된다. 출력 폴더를 비우고 새로 시작한다. ② 테스트 `test_matches_existing_adapter_config`가 기존 어댑터 설정과 달라졌다고 실패한다. 이유를 결과 기록에 적고 테스트를 고친다. ③ 바꾼 값은 결과 기록에도 적는다 |
| `체크포인트의 설정이 지금 인자와 다르다` | 학습 도중 `runs.py`나 코드가 바뀌었다. 바꾸기 전 커밋으로 돌아가거나, 폴더를 비우고 새로 시작한다 |
| `수정 전 데이터 순서 방식의 체크포인트다` | Task #5 전 코드가 남긴 체크포인트다. `runs/gpu/`에서는 나오지 않아야 한다. 폴더를 비우고 새로 시작한다 |
| `CUDA error: … unspecified launch failure` 등 학습 도중 오류 | 자동으로 다시 시도한다 (7절). 되풀이되면 작업 출력의 오류 전문을 결과 기록에 남긴다 |
| 처리량이 갑자기 수십 배 느려짐 | GPU 메모리가 넘친 상태일 수 있다. `gpu_{ID}.log`의 메모리를 본다 (학습 가이드 6절) |

## 11. R3~R10 추가와 입력 갱신

α 값(a₁, a₂)은 스타일 특성 이슈에서 정한다 (기준 문서 6.2절). 정해지면 로컬에서 다음을 하고 `devel`에 올린 뒤, 서버에서 `git pull`해서 제출한다.

1. `gpu_train/runs.py`의 `RUNS`에서 그 학습의 `None`을 α 값으로 바꾼다. 특성 학습에 옵션이 더 필요하면 `extra`에 넣는다.
2. `tests/test_gpu_train.py`를 맞춘다. 그 ID를 `PENDING`에서 빼고, 기대 설정을 검사하는 테스트를 더한다.
3. `pytest`가 통과하는지 본다.
4. 서버에서 `python -m gpu_train.runs list`로 "가능"이 됐는지 확인하고 제출한다.

- **실행 중인 학습이 있을 때는 `git pull`하지 않는다** (2절). 이어 가는 작업이 새 코드로 돌게 된다.
- 입력 파일이 바뀌면(예: 특성 캐시 추가) 다음 순서로 갱신한다. 1~5는 로컬, 6은 서버에서 한다. 처음 올릴 때도 3~4를 한다.
  1. `make_manifest.sh` 안의 파일 목록을 고친다
  2. `bash gpu_train/make_manifest.sh`로 `inputs.sha256`을 다시 만든다
  3. `bash gpu_train/upload_inputs.sh 저장소ID`로 새로 올린다. `inputs.hf`의 리비전과 태그가 바뀐다
  4. `inputs.sha256`과 `inputs.hf`를 함께 커밋하고 `devel`에 올린다
  5. 5절 직접 전송의 rsync 목록도 고친다
  6. 서버에서 학습이 돌고 있지 않을 때 `git pull` 후 `fetch_inputs.sh`
