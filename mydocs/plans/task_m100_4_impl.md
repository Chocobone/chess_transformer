# Task #4: 스미슬로프 LoRA 어댑터 본 학습 (136M) — 구현 계획서

## 공통

이 타스크는 Task #1·#2의 코드를 실행하는 작업이다. 소스(`src/`, `scripts/`, `tests/`)는 수정하지 않는다. 대책 (다)를 택하게 되면 수정안을 이 계획서에 단계로 더해 따로 승인받는다. 실행 중 코드 결함이 드러나면 실행을 멈추고 원인과 수정안을 보고한 뒤 승인을 받아 수정한다.

**실행 환경**: 로컬 개발 컨테이너(WSL2), RTX 4070 Ti 12GB, 베이스 `base_model/136M.pt`. 코드는 `local/task4`(소스는 `06b57fa` 이후 변경 없음). 호스트의 "CUDA - Sysmem Fallback Policy"는 켠 상태다. GPU 메모리를 넘기면 오류로 끝나지 않고 느려진다.

**산출물 위치**

```
runs/smyslov_probe_base/                  ← 사전 측정 (git 제외). 대책 비교는 runs/smyslov_probe_{이름}/
runs/logs/probe_smyslov_{이름}.log        ← 사전 측정의 표준 출력 (줄마다 시각), GPU 5초 기록은 probe_smyslov_{이름}_gpu.log (git 제외)
runs/smyslov_136M/                        ← adapter.pt, checkpoint.pt, config.json, log.jsonl (git 제외)
runs/logs/train_smyslov_136M.log          ← 학습의 표준 출력, 시도별 시작·종료 기록 (git 제외)
runs/logs/gpu_mem_smyslov_136M.log        ← GPU 전체 사용량, 60초마다 (git 제외)
mydocs/test_result/task_m100_4_test.md    ← 사전 측정과 학습 결과 정리 (1단계에 사전 측정 부분, 3단계에 학습 부분)
```

**측정·검증 스크립트**는 스크래치패드에서 실행하고 저장소에 넣지 않는다.

---

## 1단계: 사전 확인, 사전 측정, 대책 결정

### 목표

학습을 시작해도 되는 상태인지 확인하고, 학습 뒤 평가가 GPU 메모리 안에 들어가는지 실측한다. 정상 기준을 벗어나면 대책을 비교해 작업지시자가 정할 수 있게 한다.

### 구현

**사전 확인**

```bash
python -m pytest -q
sha256sum data/processed/smyslov/base_q_136M.npz
git diff 06b57fa HEAD -- src scripts tests
bash -n runs/logs/train_player.sh
nvidia-smi
```

**사전 측정** (기준 측정, 약 13분)

```bash
# GPU 전체 사용량·사용률·온도·전력을 5초마다 기록 (측정이 끝나면 멈춘다)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader -l 5 > runs/logs/probe_smyslov_base_gpu.log &

python -u -m src.train.train --player smyslov --base base_model/136M.pt \
    --out runs/smyslov_probe_base --max-steps 60 --eval-every 20 2>&1 \
  | while IFS= read -r line; do echo "$(date '+%T.%N' | cut -c1-12) $line"; done \
  > runs/logs/probe_smyslov_base.log
```

- 시작 직전 20초 동안의 GPU 전체 사용량을 데스크톱 사용량으로 적는다.
- 평가는 step 20, 40, 60(종료 시)에 3회 돈다. 평가 1회의 시간은 `eval` 줄의 시각에서 그 직전 `train` 줄의 시각을 뺀 값이다 (`baseline`·`train`·`eval` 줄은 표준 출력에도 찍힌다).
- `--max-steps 60`이면 학습률 일정이 본 학습과 다르다. 메모리와 시간을 재는 실행이며 지표의 값 자체는 쓰지 않는다 (대책 간 대조에만 쓴다).

**판정** — 수행 계획서의 정상 기준:

| 항목 | 기준 |
|------|------|
| 평가 1회 시간 | 세 번 모두 260초 이하 |
| GPU 전체 사용량 최대 | 11.0 GiB(11,264 MiB) 이하 |
| 평가 사이의 증가 | 3회째 평가 뒤 사용량 − 1회째 평가 뒤 사용량 ≤ 0.3 GiB |

- 셋 모두 만족 → 대책 없이 본 학습. 결과를 보고하고 승인을 받는다.
- 하나라도 벗어남 → 대책 비교로 간다.

**대책 비교** (기준을 벗어날 때만. 안마다 약 13분, 사이에 GPU 사용량이 데스크톱 수준으로 내려온 것을 확인하고 시작)

| 이름 | 기준 측정 명령에 더하는 것 |
|------|---------------------------|
| `emm128` | `--eval-max-moves 128` |
| `emm64` | `--eval-max-moves 64` (`emm128`이 기준에 못 미칠 때만) |
| `expand` | 환경변수 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `split` | 환경변수 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (`expand`가 듣지 않거나 기준에 못 미칠 때만) |

안마다 아래를 표로 만든다.

- 평가 3회의 시간, GPU 전체 사용량 최대, 1회째→3회째 증가, 60 갱신 구간의 처리량(`positions_per_sec`), `max_memory_gib`
- step 20·40·60의 val 지표 7개와 기준 측정의 차이 (최대 절대 차이). `emm*`은 1e-6 이하, `expand`·`split`은 0이어야 채택 후보다. `expand`·`split`은 `adapter.pt`의 SHA-256도 기준 측정과 같은지 본다
- 환경변수가 실제로 적용됐는지 (`expand`는 경고 없이 돌았는지, 예약 메모리의 모양이 달라졌는지)

(가)·(나) 어느 것도 기준을 만족하지 못하면 (다) `torch.cuda.empty_cache()`를 평가 전후에 넣는 수정안(수정 위치, 테스트, 탈·카르포프와의 차이)을 만들어 보고한다. 이때는 이 계획서에 소스 수정 단계를 더한다.

(가)를 택하면 `runs/logs/train_player.sh`(git 제외)가 추가 인자를 학습 명령에 넘기도록 한 줄을 고친다 (`"${@:2}"`). 이 수정은 대책 결정 보고에 넣어 함께 승인받는다. (나)는 스크립트 수정 없이 환경변수만 주면 된다.

**`mydocs/test_result/task_m100_4_test.md`** — 신규. 이 단계에서는 "사전 측정" 절을 쓴다 (명령, 데스크톱 사용량, 평가별 시간과 사용량, 대책 비교 표).

### 검증

- `pytest` 111건 통과
- 캐시의 SHA-256이 Task #2 1단계 보고서의 값(`b0904be63c28c1f4318e816f99c5ec6f398b1f4aab51915fa8a61e6b1b82f87e`)과 같다
- `git diff 06b57fa HEAD -- src scripts tests`가 비어 있다
- GPU에 다른 학습·측정 프로세스가 없고 `runs/smyslov_136M/`과 `runs/logs/*smyslov_136M*`가 없다
- 기준 측정의 `baseline` 줄이 어제 단독 측정의 스미슬로프 기준선(imitation 1.4497, top1 0.5639, top3 0.8515)과 같다
- 기준 측정이 60 갱신과 평가 3회를 끝내고 종료 코드 0으로 끝났다
- 판정에 쓴 수치가 로그에서 스크립트로 뽑은 값이다

### 보고서

`mydocs/working/task_m100_4_stage1.md` — 사전 확인 결과, 사전 측정 수치와 판정, (했다면) 대책 비교 표와 권고, 본 학습의 소요 시간 재추정. **이 보고의 승인(대책 결정 포함)을 받은 뒤 2단계를 시작한다.**

---

## 2단계: 스미슬로프 어댑터 학습

### 목표

스미슬로프의 어댑터를 3 epoch(8,736 갱신) 끝까지 학습한다.

### 구현

```bash
setsid nohup runs/logs/train_player.sh smyslov > runs/logs/train_smyslov_136M.log 2>&1 < /dev/null &
```

스크립트가 실행하는 명령 (대책이 없을 때):

```bash
python -u -m src.train.train --player smyslov --base base_model/136M.pt \
    --out runs/smyslov_136M --eval-every 1000 --resume
```

- 대책을 택했으면 1단계에서 승인받은 형태로 실행한다 ((가): 인자 추가, (나): 명령 앞에 환경변수).
- 평가는 1000 갱신마다 8회와 종료 시 1회, 합계 9회. 체크포인트는 200 갱신마다와 평가 때마다.
- 오류로 끝나면 스크립트가 20초 뒤 마지막 체크포인트에서 재개한다 (최대 10회, 진전 없이 연속 3회 실패하면 중단). 시도별 시작·종료 시각, 중단 step, 오류 메시지, 그때의 GPU 사용량을 기록한다.

**감시**

- 시작 직후: `baseline` 줄, 처음 몇 개 train 줄의 처리량과 `max_memory_gib`, 학습만 돌 때의 GPU 전체 사용량을 확인한다.
- 첫 평가(step 1000, 시작 약 25분 뒤): 평가 시간과 평가 뒤 GPU 전체 사용량을 확인해 바로 보고한다.
- 이후 20~30분 간격으로 `log.jsonl`과 `gpu_mem_smyslov_136M.log`를 확인한다.

**멈출 기준** — 하나라도 해당하면 다음 체크포인트가 저장된 직후 학습 프로세스와 스크립트를 종료하고 보고한다. 다시 시작할지, 대책을 바꿀지는 작업지시자가 정한다. 임의로 재시작하지 않는다.

| 항목 | 멈출 기준 |
|------|-----------|
| 평가 1회 시간 | 400초 초과 |
| 학습 구간 처리량 | `positions_per_sec` 18 미만이 연속 train 줄 20개(약 10분) 넘게 이어짐. 평가 직후 첫 줄은 뺀다 |
| GPU 전체 사용량 | 11.8 GiB(12,083 MiB) 이상 |

- 평가가 도는 동안에는 체크포인트가 저장되지 않는다. 평가 도중 400초를 넘긴 것을 보면 그 평가가 끝나 체크포인트가 저장될 때까지 기다렸다가 멈춘다. 평가 한 번이 20분을 넘기면 기다리지 않고 보고해 지시를 받는다.
- 멈출 때는 스크립트가 자동 재개하지 않도록 스크립트를 먼저 종료한 뒤 학습 프로세스를 종료한다.
- 재시도 한도 안에 끝나지 못하면 멈춘 상태로 두고 보고한다. 한도나 설정을 임의로 바꾸지 않는다.

### 검증

- 로그에 `=== smyslov 완료` 줄이 있고 종료 코드가 0이다
- `runs/smyslov_136M/`에 `adapter.pt`, `checkpoint.pt`, `config.json`, `log.jsonl`이 있다
- `log.jsonl`의 마지막 `eval`의 step이 `config.json`의 `total_steps`(8,736)와 같다

### 보고서

`mydocs/working/task_m100_4_stage2.md` — 실행 경과(시작·종료 시각, 시도 횟수, 중단 기록), 소요 시간, 처리량, 평가별 소요 시간, 최대 GPU 메모리와 GPU 전체 사용량의 범위, 멈출 기준에 닿았는지.

---

## 3단계: 학습 검증과 결과 정리

### 목표

학습 결과가 온전한지 확인하고, 베이스 기준선 대비 val 지표 변화를 문서로 남긴다.

### 구현

**검증 스크립트**: `log.jsonl`·`config.json`을 읽어 아래 검증 항목을 확인하고 결과 문서의 표를 출력한다.

**어댑터 재평가**: 저장된 `adapter.pt`를 새로 올린 베이스에 `load_adapter`로 붙여 val 4,753국면 전체를 `src.train.train.evaluate`로 다시 평가한다 (약 4분. 학습 종료 뒤 GPU 사용량이 데스크톱 수준으로 내려온 것을 확인하고 시작).

**`mydocs/test_result/task_m100_4_test.md`** — 1단계에서 쓴 "사전 측정" 절 뒤에 학습 결과를 더한다. `task_m100_3_test.md`와 같은 형식:

- 실행 환경, 실행 명령, 실행 경과(시도 표), 평가별 소요 시간
- `adapter.pt`의 크기와 SHA-256, `config.json`
- 베이스 기준선 대비 최종 val 지표 표와 평가 시점별 표: `imitation`, `top1`, `top3`, `regret`, `regret_top1`, `band_violation`, `strength`
- `baseline`·`resume`·`eval` 줄 전부와 마지막 `train` 줄 몇 개 (로그 원문)
- 이 지표로 말할 수 있는 것과 없는 것, 확인하지 못한 것

### 검증

- `log.jsonl`의 train 줄에 빠지거나 겹친 step이 없다 (20 간격, 마지막 줄 포함)
- 손실·기울기 노름에 NaN·무한대가 없다
- 재평가한 지표가 `log.jsonl`의 마지막 `eval` 값과 일치한다
- 스미슬로프의 `config.json`이 탈·카르포프와 `player`, `out`, 데이터 크기에 따른 값(`total_steps`, `train_positions`, `val_positions`)을 빼고 같다. 대책 (가)를 택했으면 `eval_max_moves`가 더 다르며, 그 사실과 1단계의 지표 대조 결과를 함께 적는다
- 결과 문서의 표 수치가 `log.jsonl`의 값과 일치한다 (스크립트 출력을 그대로 쓴다)
- `pytest` 통과, `git status`에 미커밋 파일 없음

### 보고서

`mydocs/working/task_m100_4_stage3.md`, 이어서 최종 결과보고서 `mydocs/report/task_m100_4_report.md`.

---

## 검증 (전체)

- `pytest` 통과
- `runs/smyslov_136M/adapter.pt`가 있고 새로 올린 베이스에 붙여 평가할 수 있다
- 탈·카르포프와 같은 학습 설정, 같은 코드로 학습되었다 (대책으로 달라진 것이 있으면 명시)
- `mydocs/test_result/task_m100_4_test.md`의 수치가 로그와 일치한다

## 일정과 예상 소요 시간

| 단계 | 내용 | 시간 |
|------|------|------|
| 1단계 | 사전 확인, 사전 측정 | 약 20분. 대책 비교를 하면 안마다 약 13분 추가 |
| 2단계 | 학습 8,736 갱신 + 평가 9회 | 약 3.7시간 (추정, 중단이 없고 평가가 정상일 때). 1단계 결과로 다시 계산한다 |
| 3단계 | 검증(재평가 약 4분 포함)·문서 | 실행 대기 거의 없음 |

## 예상 단계: 3단계

대책 (다)를 택하면 1단계와 2단계 사이에 소스 수정 단계가 더해져 4단계가 된다.
