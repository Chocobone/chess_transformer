# Task #2: 기사별 LoRA 어댑터 본 학습 (탈·카르포프·스미슬로프, 136M) — 구현 계획서

## 공통

이 타스크는 Task #1의 코드를 실행하는 작업이다. 소스 수정은 2단계의 학습 재개 기능(`src/train/train.py`, `tests/test_train.py`, 2026-10-12 작업지시자 지시)뿐이다. 그 밖에 실행 중 코드 결함이 드러나면 실행을 멈추고 원인과 수정안을 보고한 뒤 승인을 받아 수정한다.

**실행 환경**: 로컬 개발 컨테이너, RTX 4070 Ti 12GB, torch 2.14.1+cu130, 베이스 `base_model/136M.pt`.

**실행 방식**

- 캐시와 학습은 `setsid nohup`으로 세션과 분리해 돌리고 표준 출력을 `runs/logs/`에 남긴다.
- GPU에는 한 번에 하나의 작업만 올린다.
- `runs/`, `data/`는 git 제외 대상이다. 단계별 보고서에 실행 명령과 로그 발췌를 적어 기록을 남긴다.

**산출물 위치**

```
data/processed/{player}/base_q_136M.npz   ← q 캐시 (1단계)
runs/logs/                                ← 캐시·학습의 표준 출력
runs/smoke_136M/                          ← 짧은 확인 학습 (1단계)
runs/{player}_136M/                       ← adapter.pt, checkpoint.pt, config.json, log.jsonl (2·3단계)
mydocs/test_result/task_m100_2_test.md    ← 학습 결과 정리 (2단계에 작성, 3·4단계에 보강)
```

**학습 명령 (세 기사 공통, `{player}`만 다르다)**

```bash
python -u -m src.train.train --player {player} --base base_model/136M.pt \
    --out runs/{player}_136M --eval-every 1000 --resume
```

`--resume`은 2단계에서 추가한다. `--out`에 체크포인트가 있으면 이어서 돌고, 없으면 처음부터 시작한다. 실행 스크립트는 학습이 오류로 끝나면 같은 명령을 다시 실행한다 (최대 10회, 체크포인트가 한 번도 전진하지 못한 채 연속 3회 실패하면 중단).

나머지 옵션은 기본값이다 (rank 16, lora-alpha 32, targets q k v o, T 0.02, λ 1.0, τ 0.02, γ 0, α 0, lr 1e-4, 3 epoch, 국면 32개/갱신, 합법수 한도 128, seed 0). 평가는 val 전체를 float32로 한다.

---

## 1단계: 사전 확인과 q 캐시 생성

### 목표

학습에 필요한 베이스 q 캐시를 세 기사 모두 만들고, 본 학습 전에 메모리·처리량·평가 시간을 실측해 소요 시간 추정을 갱신한다.

### 구현

1. **사전 확인** (완료): `pytest` 101건 통과, `scripts/verify_port.py`로 136M의 GPU 이식 재검증 통과 (로그확률 최대 오차 8.011e-05, 최선수 53/53).
2. **q 캐시 생성** (2026-10-11 09:58 시작, 작업지시자 지시로 구현 계획서 승인 전에 시작):

   ```bash
   python -u scripts/cache_base_q.py --player tal --base base_model/136M.pt       # karpov, smyslov 순서로 이어서 실행
   ```

   float32, `--batch-moves 256`(기본값). 세 기사를 차례로 실행하고 로그는 `runs/logs/cache_136M.log`에 남긴다.
3. **짧은 확인 학습**: 캐시가 모두 끝난 뒤 탈 데이터로 실행한다.

   ```bash
   python -u -m src.train.train --player tal --base base_model/136M.pt --out runs/smoke_136M \
       --max-steps 60 --eval-every 1000
   ```

   val 전체 평가가 마지막에 한 번 돈다. 최대 GPU 메모리, 갱신당 시간, val 전체 평가 1회 시간을 잰다.

### 검증

- 캐시 스크립트가 세 기사 모두 오류 없이 끝나고 `base_q_136M.npz`가 생긴다.
- 캐시의 국면 수가 전처리 통계(`stats.json`)의 train·val 국면 수와 같다.
- `PositionDataset`으로 세 기사의 train·val을 열 때 오류가 없다 (국면 수와 합법수 순서 대조가 여기서 이뤄진다).
- 캐시의 q에 NaN·무한대가 없고 모두 [0, 1] 범위다.
- 기사마다 무작위 국면 200개의 q를 캐시와 별도로 다시 계산해 캐시 값과 비교한다 (최대 절대 오차 보고, 기대치 1e-4 이하).
- 확인 학습에서 최대 GPU 메모리가 12GB에 여유 있게 들어오고(Task #1 실측 약 4.7 GiB), 처리량이 Task #1 실측(갱신당 약 1.3초)과 크게 다르지 않다.
- 실측값으로 2·3단계 소요 시간을 다시 계산해 보고한다.

### 보고서

`mydocs/working/task_m100_2_stage1.md` — 캐시 생성 시간·합법수 수, 검증 결과, 실측 처리량과 갱신된 시간 추정.

---

## 2단계: 학습 재개 기능 추가와 탈 어댑터 학습

2026-10-12에 시작했다. 탈 학습이 CUDA 오류로 두 번 중단되어(수행 계획서 "학습 재개 기능 추가" 참조) 작업지시자 지시로 재개 기능을 먼저 넣는다.

### 2-1. 재개 기능 구현

**`src/train/train.py`**

- **체크포인트 저장**: `{out}/checkpoint.pt`. 내용은 LoRA 가중치, 옵티마이저 상태, 스케줄러 상태, 끝낸 갱신 횟수(step), 현재 epoch, 그 epoch에서 끝낸 배치 수, 그 epoch를 시작할 때의 데이터 순서 난수(generator) 상태, python·numpy·torch 난수 상태, 대조용 설정값.
  - 저장 시점은 평가 때마다(`adapter.pt` 저장 직후)와 `--checkpoint-every` 갱신마다. 새 옵션 `--checkpoint-every`의 기본값은 200이고 0이면 평가 때만 저장한다. 중단 시 잃는 양은 최대 200 갱신(약 4.5분)이다.
  - 임시 파일에 쓴 뒤 `os.replace`로 바꿔치기한다. 저장 도중 중단되어도 직전 체크포인트가 남는다.
  - 크기는 약 12.6MB로 예상한다 (LoRA 파라미터 1,048,576개 × (가중치 + Adam 모멘트 2개) × 4바이트).
- **`--resume`**: `{out}/checkpoint.pt`가 있으면 불러와 이어서 돈다. 없으면 처음부터 시작한다 (같은 명령으로 첫 실행과 재실행을 모두 처리하기 위해서다).
  - 체크포인트의 설정값이 현재 인자와 다르면 오류로 멈춘다 (`resume`, `num_workers`, `log_every`, `eval_every`, `checkpoint_every`, `device`는 비교에서 뺀다).
  - 이미 `total_steps`까지 끝난 체크포인트면 학습하지 않고 종료한다.
  - `--resume` 없이 `--out`에 `checkpoint.pt`가 있으면 덮어쓰지 않고 오류로 멈춘다 (실수로 진행분을 지우지 않게).
- **데이터 순서 복원**: 지금의 `DataLoader(shuffle=True, generator=...)`를 같은 `RandomSampler`·`BatchSampler`를 감싼 배치 샘플러로 바꾼다. 감싼 샘플러는 재개한 epoch에서 이미 끝낸 배치의 인덱스만 건너뛴다 (국면을 다시 인코딩하지 않으므로 바로 넘어간다). 난수 소비 순서는 수정 전과 같으므로 끊기지 않은 학습의 배치 순서는 바뀌지 않는다.
- **로그**: 재개할 때 `log.jsonl`에서 체크포인트 step보다 뒤의 줄을 잘라 내고 이어 쓴다. 재개 지점에 `{"type": "resume", "step": N}` 줄을 남긴다. 기준선은 다시 재지 않는다. `config.json`은 다시 쓰지 않는다.
- `elapsed`는 체크포인트에 누적값을 저장해 재개 후에도 이어지게 한다. `max_memory_gib`는 실행마다 새로 잰다.

손실(`losses.py`), 정책(`policy.py`), 모델(`src/model/`), 데이터(`src/data/`)는 수정하지 않는다. `save_adapter`·`load_adapter`도 그대로다.

**`tests/test_train.py`** — 추가 (CPU, 작은 모델)

- 끊기지 않은 학습과 재개한 학습의 동일성: N 갱신을 한 번에 돈 결과와, 중간에 멈췄다가 `--resume`으로 N까지 돈 결과에서 `adapter.pt`의 가중치와 `log.jsonl`의 train·eval 줄이 완전히 같다 (시간 관련 값 제외). epoch 중간, epoch 경계 직후, 평가 직후에 멈춘 경우를 각각 확인한다.
- `--resume`인데 체크포인트가 없으면 처음부터 돈다.
- 설정이 다른 체크포인트는 거부한다. `--resume` 없이 체크포인트가 있으면 거부한다.
- 끝난 학습을 `--resume`으로 다시 실행하면 갱신 없이 끝난다.
- 기존 테스트 101건이 그대로 통과한다.

**실행 스크립트** (`runs/logs/`, git 제외): 학습이 오류로 끝나면 `--resume`으로 다시 실행한다. 최대 10회, 체크포인트의 step이 전진하지 못한 채 연속 3회 실패하면 중단한다. 시도마다 시작·종료 시각과 종료 코드를 로그에 남긴다. GPU 전체 사용량을 60초마다 `runs/logs/gpu_mem_{player}_136M.log`에 기록한다.

### 2-2. 재개 기능 검증 (실제 모델, GPU)

탈 본 학습 전에 한다. 실행 중인 3차 시도(재개 기능 없는 코드, 04:53:31 시작)는 이 검증을 시작할 때 멈춘다.

- `pytest` 전부 통과.
- 136M·탈 데이터로 60 갱신을 `--eval-every 20 --checkpoint-every 10`으로 세 번 돈다: (가) 끊지 않고, (나) 실행 중 프로세스를 강제 종료한 뒤 `--resume`, (다) 두 번 강제 종료하고 재개. 세 실행의 `adapter.pt` SHA-256과 `log.jsonl`의 train·eval 수치가 같은지 확인한다. GPU 연산이 비트 단위로 같지 않아 차이가 나면 차이의 크기를 그대로 보고한다.
- 수정 전 코드와의 동일성: 본 학습의 train 기록과 step 1000~4000의 eval 값을 수정 전 2차 실행의 기록(`runs/tal_136M_failed_0311/log.jsonl`)과 대조한다. 재개가 끼어도 같아야 한다.

### 2-3. 탈 어댑터 학습

시작 전에 `pytest`를 돌리고, 세 기사 캐시 파일의 SHA-256이 1단계 보고서의 값과 같은지 확인한다 (2026-10-12 02:07 실행 전에 확인했고, 재개 기능을 넣은 뒤 다시 확인한다).

### 목표

탈의 어댑터를 3 epoch 끝까지 학습하고, 결과를 보고해 같은 설정으로 나머지 두 기사를 학습할지 결정할 근거를 만든다.

### 구현

```bash
python -u -m src.train.train --player tal --base base_model/136M.pt \
    --out runs/tal_136M --eval-every 1000 --resume
```

- 7,455 갱신, 평가는 1000 갱신마다와 종료 시 (총 8회). 로그는 `runs/logs/train_tal_136M.log`.
- 중단되면 실행 스크립트가 마지막 체크포인트에서 재개한다. 중단 횟수, 중단 시각·step·오류, 그때의 GPU 사용량을 보고서에 적는다.
- 학습 중 `max_memory_gib`와 `positions_per_sec`를 확인한다. 처리량이 급격히 떨어지면(메모리 초과) 멈추고 보고한다.
- 학습이 끝나면 `mydocs/test_result/task_m100_2_test.md`를 만들어 실행 명령, GPU, `config.json`, `baseline` 줄, 모든 `eval` 줄, 마지막 `train` 줄 몇 개, `adapter.pt`의 크기와 SHA-256을 적는다.

### 검증

- 학습이 마지막 갱신까지 끝나고(중간에 재개가 있었어도 된다) `adapter.pt`, `config.json`, `log.jsonl`이 생긴다.
- `log.jsonl`의 train 줄에 빠지거나 겹친 step이 없다.
- train 기록과 step 1000~4000의 eval 값이 수정 전 2차 실행의 기록과 같다.
- `log.jsonl`의 마지막 `eval`의 step이 `config.json`의 `total_steps`와 같다.
- 손실·기울기 노름에 NaN·무한대가 없다.
- 저장된 `adapter.pt`를 새로 올린 베이스에 `load_adapter`로 붙여 val 전체를 다시 평가한 지표가 `log.jsonl`의 마지막 `eval` 값과 일치한다 (저장물이 실제로 쓸 수 있는지 확인).
- 베이스 기준선 대비 val 지표 변화를 표로 보고한다: `imitation`, `top1`, `top3`, `regret`, `regret_top1`, `band_violation`, `strength`. 개선되지 않은 지표도 그대로 적는다.

### 보고와 결정

보고서에 설정 유지 여부에 대한 의견을 적고 작업지시자의 결정을 받는다. 판단에 볼 것:

- `imitation`이 내려가고 `top1`/`top3`가 올라갔는가, 평가 시점마다 꾸준한가
- 후반 평가에서 `imitation`이 다시 오르는가 (과적합)
- `regret`, `band_violation`이 기준선보다 얼마나 올랐는가
- train 손실과 val 손실의 차이

설정을 바꾸기로 하면 구현 계획서를 수정해 승인받고, 탈을 새 설정으로 다시 학습한다.

### 보고서

`mydocs/working/task_m100_2_stage2.md`

---

## 3단계: 카르포프·스미슬로프 어댑터 학습

### 목표

2단계에서 확정한 설정으로 나머지 두 기사의 어댑터를 학습한다. 설정은 탈과 같게 유지한다 (2026-10-12 작업지시자 결정).

하루에 한 기사씩 진행한다 (2026-10-12 작업지시자 지시): 카르포프는 2026-10-13, 스미슬로프는 2026-10-14.

### 구현

```bash
python -u -m src.train.train --player karpov --base base_model/136M.pt \
    --out runs/karpov_136M --eval-every 1000 --resume
python -u -m src.train.train --player smyslov --base base_model/136M.pt \
    --out runs/smyslov_136M --eval-every 1000 --resume
```

- 카르포프 9,039 갱신(평가 10회), 스미슬로프 8,736 갱신(평가 9회). 두 학습은 날짜를 나누어 따로 실행한다 (`runs/logs/train_player.sh {player}`). 중단되면 2단계와 같이 재개한다. 카르포프 학습이 재시도 한도 안에 끝나지 못하면 스미슬로프 학습을 시작하지 않고 보고한다.
- 각 날 학습을 시작하기 전에 `pytest`를 돌리고 캐시 파일의 SHA-256이 1단계 보고서의 값과 같은지 확인한다.
- 로그는 `runs/logs/train_{player}_136M.log`.
- 결과를 `mydocs/test_result/task_m100_2_test.md`에 2단계와 같은 형식으로 추가한다. 카르포프 결과는 2026-10-13에 추가하고 그날 보고한다.

### 검증

2단계의 검증 항목을 두 기사에 똑같이 적용한다. 추가로:

- 세 기사의 `config.json`이 `player`, `out`, 데이터 크기에 따른 값(`total_steps`, `train_positions`, `val_positions`)을 빼고 동일하다.

### 보고서

`mydocs/working/task_m100_2_stage3.md` — 스미슬로프 학습이 끝난 뒤(2026-10-14) 두 기사의 결과를 함께 적는다.

---

## 4단계: 결과 종합과 문서 갱신

### 목표

세 기사의 결과를 한 문서로 비교할 수 있게 정리하고, 실행 가이드의 추정치를 이번 실측값으로 바꾼다.

### 구현

**`mydocs/test_result/task_m100_2_test.md`** — 종합 부분 추가:

- 기사별 베이스 기준선과 최종 val 지표를 나란히 놓은 표, 평가 시점별 추이 표
- 실측 소요 시간 (캐시, 학습, 평가), 최대 GPU 메모리
- 한계: 이 지표로 말할 수 있는 것과 없는 것 (val 모방·기력 지표만 봤고 스타일 반영 여부는 평가하지 않았다는 점 등)

**`mydocs/manual/training_guide.md`** — 갱신:

- 5·6절의 소요 시간과 처리량을 전체 데이터 실측값으로 수정 (136M)
- 평가 시간과 `--eval-every` 선택 기준 추가
- 세션과 분리해 실행하는 방법 추가
- 체크포인트와 `--resume`, 중단 시 재개 방법 추가
- 7절의 결과 파일명 예시를 타스크 번호에 맞게 일반화

4단계에서는 소스 코드(`src/`, `scripts/`, `tests/`)를 수정하지 않는다.

### 검증

- 종합 표의 모든 수치가 `runs/{player}_136M/log.jsonl`의 값과 일치하는지 스크립트로 대조한다 (손으로 옮겨 적지 않는다).
- `pytest` 전부 통과.
- `git status`에 미커밋 파일이 없다.

### 보고서

`mydocs/working/task_m100_2_stage4.md`, 이어서 최종 결과보고서 `mydocs/report/task_m100_2_report.md`.

---

## 검증 (전체)

- `pytest` 통과
- 세 기사의 `runs/{player}_136M/adapter.pt`가 있고 새로 올린 베이스에 붙여 평가할 수 있다
- 세 기사가 같은 설정, 같은 코드(재개 기능을 넣은 뒤의 커밋)로 학습되었다
- `mydocs/test_result/task_m100_2_test.md`의 수치가 로그와 일치한다

## 일정과 예상 소요 시간

2~3단계의 시간은 1단계 실측(갱신당 1.33초, val 평가 1회 142~190초)으로 계산한 추정이다.

| 단계 | 내용 | 시간 | 일정 |
|------|------|------|------|
| 1단계 | q 캐시 + 검증 + 확인 학습 | 캐시 2시간 49분 (실측) | 2026-10-11 완료 |
| 2단계 | 재개 기능 구현·검증, 탈 학습 약 2.8시간 + 평가 8회 약 19분 (중단이 없을 때) | 학습 3시간 8분 (실측, 중단 1회 포함) | 2026-10-12 완료 |
| 3단계 | 카르포프 약 3.9시간, 스미슬로프 약 3.7시간 (평가 19회 포함) | 약 7.6시간 | 카르포프 2026-10-13, 스미슬로프 2026-10-14 |
| 4단계 | 종합·문서 | 실행 대기 없음 | 3단계 승인 후 |

## 예상 단계: 4단계
