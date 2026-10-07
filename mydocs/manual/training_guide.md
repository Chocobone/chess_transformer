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

### 저장물 (`--out` 폴더)

| 파일 | 내용 |
|------|------|
| `adapter.pt` | 어댑터 가중치와 주입 설정. 평가할 때마다 갱신된다. 베이스 가중치는 들어 있지 않다 |
| `config.json` | 실행 설정 전체 |
| `log.jsonl` | 한 줄에 한 기록 (`baseline`, `train`, `eval`) |

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

### 짧게 먼저 돌려 보기

본 학습 전에 설정과 메모리를 확인한다.

```bash
python -m src.train.train --player tal --base base_model/136M.pt --out runs/smoke \
    --max-steps 50 --eval-every 25 --eval-positions 500
```

## 7. 결과 저장

학습이 끝나면 아래를 `mydocs/test_result/`에 저장한다. 파일명은 `task_m100_1_test_v2.md`처럼 기존 결과와 겹치지 않게 한다.

- 실행한 명령 전체와 GPU 종류
- `config.json`
- `log.jsonl`의 `baseline` 줄과 모든 `eval` 줄, 마지막 `train` 줄 몇 개
- 오류가 났다면 오류 메시지 전문

어댑터 파일(`runs/`)과 캐시(`data/`)는 git에 올리지 않는다.

## 자주 나는 오류

| 메시지 | 원인과 조치 |
|--------|-------------|
| `베이스 q 캐시가 없다` | 5단계를 그 베이스로 먼저 실행한다. 캐시 이름은 `--base` 파일명을 따른다 |
| `캐시의 국면 수가 … 다르다` | 전처리를 다시 한 뒤 캐시를 갱신하지 않았다. 캐시를 다시 만든다 |
| `캐시의 합법수 순서가 국면과 맞지 않는다` | 캐시와 JSONL이 서로 다른 전처리 결과다. 캐시를 다시 만든다 |
| `CUDA out of memory` | `--max-moves-per-forward`를 낮춘다 |
| 스텝이 갑자기 수십 배 느려짐 | GPU 메모리가 넘쳐 시스템 메모리를 빌려 쓰는 상태다 (WSL). 한도를 낮춘다 |
