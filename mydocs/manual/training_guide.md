# 학습 실행 가이드

기사별 LoRA 어댑터를 GPU 서버에서 학습하는 절차다. 모든 명령은 저장소 루트에서 실행한다.
세 기사(`tal`, `karpov`, `smyslov`)에는 `--player`만 바꿔 **같은 설정**을 쓴다.

전체 순서:

```
환경 설치 → 베이스 가중치 준비 → 이식 검증 → 전처리 → q 캐시 → 학습 → 결과 저장
```

## 1. 환경 설치

Python 3.11 이상이 필요하다 (개발은 3.13에서 했다).

```bash
# CUDA 버전에 맞는 torch를 먼저 설치한다 (https://pytorch.org 의 설치 명령 참고)
pip install torch
pip install -r requirements.txt
pytest          # 전부 통과해야 한다
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True여야 한다
```

로컬 컨테이너를 쓸 때는 `docker/compose.yml`이 GPU를 연결한다.

## 2. 베이스 가중치 준비

학습 코드는 변환된 PyTorch 체크포인트(`base_model/{크기}.pt`)만 읽는다. 준비 방법은 두 가지다.

### 방법 A: 변환된 파일을 복사한다 (권장)

로컬에서 이미 변환과 검증을 마친 파일이 있다. 서버의 `base_model/`에 복사하면 JAX 환경이 필요 없다.

| 파일 | 크기 |
|------|------|
| `base_model/136M.pt` | 약 546 MB |
| `base_model/270M.pt` | 약 1.08 GB |
| `base_model/reference_136M.npz`, `reference_270M.npz` | 이식 검증용 기준 출력 |

### 방법 B: 원본 체크포인트에서 다시 변환한다

원본 체크포인트는 orbax 형식이라 추출 단계에만 JAX 환경(`jax`, `orbax-checkpoint`)이 필요하다. 학습 환경과 분리된 가상환경에서 실행하는 것을 권한다.

```bash
cd base_model && bash download.sh && cd ..                      # 원본 체크포인트 내려받기
python scripts/export_haiku_params.py --checkpoint base_model/270M --out base_model/270M_params.npz   # JAX 환경
python -m src.model.convert --params base_model/270M_params.npz --size 270M --out base_model/270M.pt  # 학습 환경
```

변환 시 키 누락·잉여가 하나라도 있으면 실패한다. `--size`는 `9M`, `136M`, `270M` 중 하나다.

## 3. 이식 검증

서버의 GPU에서 PyTorch 모델이 원본과 같은 값을 내는지 확인한다. 학습 전에 한 번 실행한다.

```bash
python scripts/verify_port.py --base base_model/270M.pt --reference base_model/reference_270M.npz
```

마지막 줄이 `결과: 통과`여야 한다 (구간 로그확률 최대 절대 오차 1e-4 이하, 최선수 53/53 일치).
기준 출력을 직접 다시 만들려면 원본 searchless_chess 환경에서 `scripts/dump_jax_reference.py`를 실행한다 (파일 머리말 참고).

## 4. 전처리

```bash
python -m src.data.preprocess --player tal
python -m src.data.preprocess --player karpov
python -m src.data.preprocess --player smyslov
```

`data/processed/{player}/`에 `train.jsonl`, `val.jsonl`, `stats.json`이 생긴다. 기본 설정(속기·다면기 등 제외, val 5%, seed 0)의 국면 수는 다음과 같다.

| 기사 | train | val |
|------|-------|-----|
| tal | 79,493 | 3,880 |
| karpov | 96,393 | 5,213 |
| smyslov | 93,178 | 4,753 |

## 5. 베이스 q 캐시

동결 베이스의 q(s, a)를 모든 국면·합법수에 대해 한 번 계산해 둔다. 기력 항, 모방 항 가중치(γ), 베이스 기준선 평가가 이 값을 쓴다.

```bash
python scripts/cache_base_q.py --player tal --base base_model/270M.pt
```

- 출력: `data/processed/{player}/base_q_{베이스 파일명}.npz` (예: `base_q_270M.npz`). 베이스 크기마다 따로 만든다.
- **전처리를 다시 하면 캐시도 다시 만든다.** 국면 수가 다르거나 수 순서가 어긋난 캐시는 학습 시작 시 오류로 걸러진다.
- `--batch-moves`(기본 256)는 키워도 빨라지지 않는다. GPU 메모리를 넘기면 오히려 수십 배 느려진다 (12GB GPU에서 1,024로 확인).
- `--amp`를 주면 약 2.5배 빠르지만 q가 평균 5e-4, 최대 1e-2 정도 달라진다. 기력 항의 밴드 τ=0.02에 비해 무시하기 어려운 크기이므로 기본값(float32)을 권한다.

소요 시간 (RTX 4070 Ti 12GB 실측 처리량 기준 추정, 세 기사 합계 합법수 약 900만 개):

| 베이스 | float32 처리량 | 세 기사 합계 | `--amp` 처리량 |
|--------|----------------|--------------|----------------|
| 136M | 약 880 수/초 | 약 2.8시간 | 약 2,150 수/초 |
| 270M | 약 450 수/초 | 약 5.6시간 | 약 1,100 수/초 |

## 6. 학습

```bash
python -m src.train.train --player tal --base base_model/270M.pt --out runs/tal_270M
```

기본값은 아래와 같다. 값은 출발점이며 첫 학습 결과를 보고 조정한다.

| 옵션 | 기본값 | 의미 |
|------|--------|------|
| `--rank`, `--lora-alpha` | 16, 32 | LoRA 랭크와 스케일 |
| `--lora-targets` | `q k v o` | 주입 대상. `mlp`를 더하면 MLP 선형층에도 주입 |
| `--temperature` | 0.02 | 정책 π = softmax(q / T)의 T |
| `--lam`, `--tau` | 1.0, 0.02 | 기력 항 가중치 λ, 허용 regret 밴드 τ |
| `--gamma` | 0.0 | 모방 항 가중치 (1 − π_base(a*))^γ. 0이면 일반 교차엔트로피 |
| `--alpha` | 0.0 | 스타일 항 가중치 (스타일 특성이 아직 없어 효과 없음) |
| `--lr`, `--epochs` | 1e-4, 3 | AdamW, 선형 워밍업 후 코사인 감쇠 |
| `--batch-positions` | 32 | 갱신 1회에 쓰는 국면 수 |
| `--max-moves-per-forward` | 128 | 한 번에 순전파·역전파하는 합법수 수의 한도 |
| `--eval-every` | 500 | 평가 주기 (갱신 횟수) |
| `--checkpoint-every` | 200 | 재개용 체크포인트 저장 주기 (갱신 횟수). 평가 때는 항상 저장한다. 0이면 평가 때만 |
| `--resume` | 꺼짐 | `--out`에 체크포인트가 있으면 이어서 돈다 (아래 "중단과 재개") |

### GPU 메모리와 `--max-moves-per-forward`

국면 32개의 합법수(약 600~1,300개)를 한 번에 역전파하면 메모리가 부족하므로, 합법수 한도 이하의 묶음으로 나눠 기울기를 누적한 뒤 한 번 갱신한다. **이 값을 바꿔도 학습 결과는 같고 속도와 메모리만 달라진다.**

| 베이스 | 한도 | 최대 GPU 메모리 (12GB GPU 실측) |
|--------|------|--------------------------------|
| 136M | 128 | 약 4.7 GiB |
| 270M | 128 | 약 9.2 GiB |

- 메모리는 한도에 거의 비례한다 (합법수 1개당 136M 약 35 MiB, 270M 약 65 MiB).
- 12GB GPU에서 270M을 돌릴 때는 `--max-moves-per-forward 96`으로 낮춘다.
- 메모리가 큰 GPU에서는 한도를 올려도 되지만 속도 이득은 크지 않다. **메모리 한계 근처까지 올리지 않는다** — 넘치면 실패하거나(리눅스) 극단적으로 느려진다(WSL).
- 로그의 `max_memory_gib`로 실제 사용량을 확인한다.

### 학습 중 평가와 GPU 메모리

`max_memory_gib`는 PyTorch가 실제로 할당한 최대량이다. GPU 전체 사용량(`nvidia-smi`)은 이보다 훨씬 클 수 있다. PyTorch가 잡아 둔 예약 메모리, 그리고 데스크톱이 쓰는 0.3~1.2 GiB가 더해지기 때문이다. 학습 중 평가는 합법수 256개 묶음을 float32로 순전파하는데, 이때 예약 메모리가 학습이 잡아 둔 양 위에 더해진다.

| 기사 (136M, 기본 할당 방식) | 학습만 돌 때 | 평가가 시작된 뒤 | 평가 1회 |
|---|---|---|---|
| 탈 | 7.41 GiB | 최대 9.75 GiB | 161초 |
| 카르포프 | 8.01 GiB | 11.5~11.7 GiB (12 GiB 카드 한도) | 8.7분 → 뒤로 갈수록 38분 |

카르포프처럼 전용 메모리를 넘기면 WSL에서는 오류 없이 시스템 메모리를 빌려 써서 극단적으로 느려진다. 카르포프는 평가 10회에 약 3.5시간이 걸렸고, 학습 구간 처리량도 떨어졌다.

**대책: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 실행한다.** 스미슬로프에서 사전 측정으로 확인했다 (Task #4).

| 스미슬로프 사전 측정 (60 갱신 + 평가 3회) | GPU 전체 최대 | 평가 1회 |
|---|---|---|
| 기본 할당 방식 | 11.29 GiB | 195초 |
| `--eval-max-moves 128` | 9.95 GiB | 같음 |
| `expandable_segments:True` | 5.57 GiB | 같음 |

- `expandable_segments:True`는 학습 결과를 바꾸지 않는다. 60 갱신 어댑터의 SHA-256이 같았고, 본 학습 어댑터를 기본 방식으로 다시 평가한 값도 차이 0이었다.
- 스미슬로프 본 학습은 이 설정으로 GPU 전체 최대 5.92 GiB, 평가 1회 약 194초로 끝까지 일정했다.
- **환경변수는 `config.json`에 남지 않는다.** 실행 기록(`mydocs/test_result/`)에 명령과 함께 적는다. 탈·카르포프 136M은 이 환경변수 없이, 스미슬로프 136M은 이 환경변수로 학습했다.
- GPU 전체 사용량을 적을 때는 `nvidia-smi`의 MiB를 1024로 나눠 GiB로 쓰고, MiB 값을 함께 적는다. Task #2~#4 기록에는 1000으로 나눈 값이 섞여 있어 정정 표시를 붙였다 (Task #6).

### 136M 본 학습 실측

RTX 4070 Ti 12GB(WSL2) 기준이다. `--eval-every 1000`, 나머지는 기본값이다.

| | 탈 | 카르포프 | 스미슬로프 |
|---|---|---|---|
| 갱신 수 (3 epoch) | 7,455 | 9,039 | 8,736 |
| 처리량 중앙값 | 24.7 국면/초 (갱신당 약 1.3초) | 23.7 | 25.5 |
| `max_memory_gib` | 4.61 | 4.62 | 4.57 |
| 평가 1회 (val 전체) | 161초 | 8.7~38분 (메모리 넘침) | 194초 |
| 전체 소요 | 3시간 8분 | 7시간 18분 | 3시간 37분 |

- 평가 지연이 없으면 136M 본 학습은 3.1~3.6시간이다.
- 기본값 표의 `--eval-every`는 500이지만, 136M 본 학습에서는 평가 시간을 줄이려고 1000을 썼다.
- val imitation 개선의 약 70%가 첫 1,000 갱신에서, 99% 이상이 2 epoch 안에서 났다. 세 기사의 결과 종합은 `mydocs/tech/adapters_136M_summary.md`에 있다.

### 저장물 (`--out` 폴더)

| 파일 | 내용 |
|------|------|
| `adapter.pt` | 어댑터 가중치와 주입 설정. 평가할 때마다 갱신된다. 베이스 가중치는 들어 있지 않다 |
| `checkpoint.pt` | 재개용 상태 (아래 "중단과 재개"). 학습된 어댑터를 쓸 때는 필요 없다 |
| `config.json` | 실행 설정 전체 |
| `log.jsonl` | 한 줄에 한 기록 (`baseline`, `train`, `eval`, 재개했다면 `resume`) |

학습된 어댑터를 쓰려면:

```python
from src.model.lora import load_adapter
from src.model.transformer import load_base

model = load_base("base_model/270M.pt", "cuda")
load_adapter(model, "runs/tal_270M/adapter.pt")   # LoRA 주입 + 가중치 로드
model.eval()
```

### 로그 읽는 법

- `baseline`: 학습 전 베이스 모델의 val 지표 (캐시의 q로 계산). `eval`과 비교하는 기준이다.
- `eval`: 어댑터를 붙인 모델의 val 지표. float32로 계산하므로 `baseline`과 바로 비교할 수 있다.

| 지표 | 의미 | 기대 방향 |
|------|------|-----------|
| `imitation` | 기사가 둔 수에 대한 교차엔트로피 | 내려간다 |
| `top1`, `top3` | 기사 수가 정책의 상위 1/3개 안에 든 비율 | 올라간다 |
| `regret` | 정책대로 둘 때 베이스 기준으로 잃는 승률의 기댓값 | 크게 오르지 않아야 한다 |
| `regret_top1` | 정책의 최선수 하나의 regret. 베이스는 정의상 0이다 | 크게 오르지 않아야 한다 |
| `band_violation` | regret이 τ를 넘는 수에 실린 확률 | 크게 오르지 않아야 한다 |
| `strength` | 기력 항의 값 | 크게 오르지 않아야 한다 |

`train` 기록의 `positions_per_sec`로 남은 시간을 가늠한다. 평가는 전체 val을 float32로 돌리므로 시간이 걸린다. 줄이려면 `--eval-positions 1000`(val 앞쪽만 평가)이나 `--eval-amp`를 쓴다.

### 중단과 재개

학습이 중간에 멈추면 같은 명령에 `--resume`을 붙여 다시 실행한다. 마지막 체크포인트에서 이어서 돈다.

```bash
python -u -m src.train.train --player tal --base base_model/136M.pt \
    --out runs/tal_136M --eval-every 1000 --resume
```

- `--resume`은 처음부터 붙여 두어도 된다. 체크포인트가 없으면 처음부터 시작하고, 이미 끝난 학습이면 갱신 없이 끝난다.
- 체크포인트(`checkpoint.pt`)에는 LoRA 가중치, 옵티마이저·스케줄러 상태, 갱신 횟수, 데이터 순서와 난수 상태가 들어 있다. 136M·rank 16에서 약 12.7MB이고, 임시 파일에 쓴 뒤 바꿔치기하므로 저장 도중에 멈춰도 이전 체크포인트가 남는다.
- **재개한 학습의 결과는 끊기지 않은 학습과 같다.** 어댑터 가중치와 로그의 `train`·`eval` 수치가 같다 (시간 관련 값 제외). 테스트와 136M 60 갱신 실행에서 확인했다.
  - **정정 (2026-10-15, Task #5)**: Task #5 전 코드는 `--num-workers` > 0일 때 epoch 1 이후에 재개하면 데이터 순서가 달라져 이 말이 성립하지 않았다. 탈·카르포프·스미슬로프 136M 어댑터는 그 코드로 재개했다 (`mydocs/troubleshootings/resume_data_order.md`). 지금 코드는 `--num-workers`와 재개 여부에 관계없이 같다.
- Task #5 전 코드가 남긴 체크포인트로는 끝나지 않은 학습을 이을 수 없다. `수정 전 데이터 순서 방식의 체크포인트다`로 거부하므로 폴더를 비우고 새로 시작한다. 이미 끝난 학습은 지금처럼 갱신 없이 끝난다.
- 중단되면 마지막 체크포인트 뒤의 갱신만 잃는다 (기본값으로 최대 200 갱신, 136M에서 약 4~5분).
- 재개할 때 `log.jsonl`에서 체크포인트 뒤의 줄을 잘라 내고 `{"type": "resume", "step": N}` 줄을 남긴 뒤 이어 쓴다. 기준선은 다시 재지 않는다.
- 학습 설정은 처음 실행과 같아야 한다. 다르면 `체크포인트의 설정이 지금 인자와 다르다`로 거부한다. `--num-workers`, `--log-every`, `--eval-every`, `--checkpoint-every`, `--device`는 바꿔도 된다.
- `--resume` 없이 체크포인트가 있는 폴더에 쓰려고 하면 거부한다. 새로 시작하려면 폴더를 비우거나 다른 `--out`을 쓴다.

오래 걸리는 학습은 터미널과 분리해 실행하고, 오류로 끝나면 같은 명령을 다시 실행하게 해 두면 사람이 지켜보지 않아도 된다. `--resume`이 붙어 있으므로 다시 실행하는 것만으로 이어진다.

세 기사 136M 학습에는 아래 형태의 스크립트(`runs/logs/train_player.sh`, git 제외)를 썼다. 시도마다 시작·실패 시각과 체크포인트 step을 남기고, GPU 전체 사용량을 60초마다 기록한다. 최대 10회 다시 실행하며, 체크포인트가 전진하지 못한 채 연속 3회 실패하면 멈춘다.

```bash
#!/bin/bash
# 사용: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup runs/logs/train_player.sh tal \
#           > runs/logs/train_tal_136M.log 2>&1 < /dev/null &
player=$1; out=runs/${player}_136M
ckpt_step() {
  [ -f "$out/checkpoint.pt" ] || { echo 0; return; }
  python -c "import torch; print(torch.load('$out/checkpoint.pt', map_location='cpu', weights_only=True)['step'])" 2>/dev/null || echo 0
}
( while :; do
    echo "$(date '+%F %T'), $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
    sleep 60
  done ) >> runs/logs/gpu_mem_${player}_136M.log 2>&1 &
trap "kill $! 2>/dev/null" EXIT
stuck=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  before=$(ckpt_step)
  echo "=== $player 시도 $attempt 시작 $(date '+%F %T') (체크포인트 step $before)"
  python -u -m src.train.train --player "$player" --base base_model/136M.pt \
      --out "$out" --eval-every 1000 --resume
  rc=$?; after=$(ckpt_step)
  if [ $rc -eq 0 ]; then echo "=== $player 완료 $(date '+%F %T') (시도 $attempt회, step $after)"; exit 0; fi
  echo "=== $player 시도 $attempt 실패 (종료 코드 $rc) $(date '+%F %T') (체크포인트 step $after)"
  if [ "$after" -gt "$before" ]; then stuck=0; else stuck=$((stuck + 1)); fi
  if [ $stuck -ge 3 ]; then echo "=== $player 중단: 진전 없이 연속 3회 실패"; exit 1; fi
  sleep 20
done
echo "=== $player 중단: 시도 10회 소진"; exit 1
```

- **중단 지점은 실행 로그에서 실패한 시도의 마지막 `train` 줄로 적는다.** `log.jsonl`은 재개할 때 체크포인트 뒤의 줄을 잘라 내므로, 실패한 시도가 체크포인트 뒤에 어디까지 갔는지는 실행 로그에만 남는다. 잃은 갱신은 "마지막 train 줄 − 체크포인트 step"에서 그 값 + 19 사이다. Task #2·#4 기록은 이것을 확인하지 않아 틀렸다 (Task #5에서 정정).
- 사람이 학습을 끊어야 할 때(예: 처리량 저하)는 체크포인트가 저장된 직후에 끊으면 잃는 갱신이 거의 없다. 카르포프 학습에서 이렇게 두 번 끊었다.

로컬 GPU(RTX 4070 Ti 12GB, WSL2)에서는 학습이 원인 불명의 CUDA 오류로 멈춘 적이 있다.

| 학습 | 오류 | 멈춘 지점 |
|---|---|---|
| 탈 1차 (재개 기능 전) | `device-side assert triggered` (gather 인덱스 범위 초과) | 1,840 갱신 직후 |
| 탈 2차 (재개 기능 전) | `unspecified launch failure` | step 4000 평가 직후 |
| 탈 4차 시도 1 | `device-side assert triggered` | step 5000~5020 |
| 스미슬로프 시도 1 | `device-side assert triggered` | step 5580~5600 |

- 카르포프는 CUDA 오류 없이 끝났다.
- 멈춘 지점은 실행마다 달랐다. 탈 1차 구간은 다시 돌려도 재현되지 않았다.
- 오류는 프로세스를 끝내고, 재개는 마지막 체크포인트부터 다시 시작하므로 가중치가 오염되지 않는다.
- 원인 조사는 보류 중이다. 자동 재시도로 대응한다.

### 짧게 먼저 돌려 보기

본 학습 전에 설정과 메모리를 확인한다.

```bash
python -m src.train.train --player tal --base base_model/136M.pt --out runs/smoke \
    --max-steps 50 --eval-every 25 --eval-positions 500
```

## 7. 결과 저장

학습이 끝나면 `mydocs/test_result/task_{milestone}_{번호}_test.md`에 아래를 정리한다. 후속 수정은 `_v2`처럼 접미어를 붙인다. 표의 수치는 `log.jsonl`·`config.json`·실행 로그를 읽는 스크립트로 뽑는다.

- 실행한 명령 전체, **환경변수**(`PYTORCH_CUDA_ALLOC_CONF` 등), GPU 종류, 코드 커밋
- 시도별 시작·종료 시각, 종료 코드, 중단 지점(실패한 시도의 마지막 `train` 줄), 재개 step
- `config.json`
- `log.jsonl`의 `baseline` 줄과 모든 `eval` 줄, 마지막 `train` 줄 몇 개
- 처리량, `max_memory_gib`, GPU 전체 사용량 최대(MiB와 ÷1024 GiB), 평가 1회 시간
- `adapter.pt`의 크기와 SHA-256
- 오류가 났다면 오류 메시지 전문

어댑터 파일(`runs/`), 실행 로그·GPU 기록(`runs/logs/`), 캐시(`data/`)는 git에 올리지 않는다.

## 자주 나는 오류

| 메시지 | 원인과 조치 |
|--------|-------------|
| `베이스 q 캐시가 없다` | 5단계를 그 베이스로 먼저 실행한다. 캐시 이름은 `--base` 파일명을 따른다 |
| `캐시의 국면 수가 … 다르다` | 전처리를 다시 한 뒤 캐시를 갱신하지 않았다. 캐시를 다시 만든다 |
| `캐시의 합법수 순서가 국면과 맞지 않는다` | 캐시와 JSONL이 서로 다른 전처리 결과다. 캐시를 다시 만든다 |
| `CUDA out of memory` | `--max-moves-per-forward`를 낮춘다 |
| `checkpoint.pt가 이미 있다` | 그 폴더에 이전 학습의 체크포인트가 있다. 이어서 돌리려면 `--resume`을 붙이고, 새로 시작하려면 폴더를 비운다 |
| `체크포인트의 설정이 지금 인자와 다르다` | 처음 실행과 다른 학습 설정으로 재개하려 했다. 메시지에 나온 항목을 처음 값으로 맞춘다 |
| `수정 전 데이터 순서 방식의 체크포인트다` | Task #5 전 코드가 남긴 체크포인트로 끝나지 않은 학습을 이으려 했다. 이어도 끊기지 않은 학습과 순서가 달라지므로 거부한다. 폴더를 비우고 새로 시작한다 |
| `CUDA error: device-side assert triggered`, `unspecified launch failure` (학습 도중) | 로컬 GPU에서 원인 불명으로 발생했다 (탈 3회, 스미슬로프 1회). `--resume`으로 다시 실행한다. 자동 재시도 스크립트를 쓰면 사람이 없어도 이어진다 |
| 스텝이 갑자기 수십 배 느려짐 | GPU 메모리가 넘쳐 시스템 메모리를 빌려 쓰는 상태다 (WSL). 학습 중 평가 뒤라면 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 다시 실행한다. 학습 구간이라면 `--max-moves-per-forward`를 낮춘다 |
