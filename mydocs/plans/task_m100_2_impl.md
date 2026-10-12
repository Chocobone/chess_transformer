# Task #2: 기사별 LoRA 어댑터 본 학습 (탈·카르포프·스미슬로프, 136M) — 구현 계획서

## 공통

이 타스크는 Task #1의 코드를 실행하는 작업이다. 계획상 소스 수정은 없고, 4단계의 문서 갱신만 저장소에 남는다. 실행 중 코드 결함이 드러나면 실행을 멈추고 원인과 수정안을 보고한 뒤 승인을 받아 수정한다.

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
runs/{player}_136M/                       ← adapter.pt, config.json, log.jsonl (2·3단계)
mydocs/test_result/task_m100_2_test.md    ← 학습 결과 정리 (2단계에 작성, 3·4단계에 보강)
```

**학습 명령 (세 기사 공통, `{player}`만 다르다)**

```bash
python -u -m src.train.train --player {player} --base base_model/136M.pt \
    --out runs/{player}_136M --eval-every 1000
```

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

## 2단계: 탈·카르포프 어댑터 학습

**2026-10-12에 시작한다** (2026-10-11 작업지시자 지시). 시작 전에 `pytest`를 다시 돌리고, 세 기사 캐시 파일의 SHA-256이 1단계 보고서의 값과 같은지 확인한다.

**2026-10-12 변경 (작업지시자 지시)**: 원래 2단계는 탈 학습만이었고 카르포프는 3단계였다. 탈 학습이 도는 중에 카르포프 학습을 2단계에 포함하기로 했다. 탈이 끝나면 승인 대기 없이 같은 설정으로 카르포프를 이어서 학습하고, 두 기사의 검증과 비교를 함께 보고한다.

### 목표

탈과 카르포프의 어댑터를 3 epoch 끝까지 학습하고, 두 결과를 비교해 보고한다. 같은 설정으로 스미슬로프를 학습할지 결정할 근거를 만든다.

### 구현

```bash
python -u -m src.train.train --player tal --base base_model/136M.pt \
    --out runs/tal_136M --eval-every 1000
python -u -m src.train.train --player karpov --base base_model/136M.pt \
    --out runs/karpov_136M --eval-every 1000
```

- 탈 7,455 갱신(평가 8회), 카르포프 9,039 갱신(평가 10회). 평가는 1000 갱신마다와 종료 시. 로그는 `runs/logs/train_{player}_136M.log`.
- 두 학습을 차례로 이어서 실행한다. 탈 학습이 정상 종료하지 않으면 카르포프 학습은 시작하지 않는다.
- 학습 중 `max_memory_gib`와 `positions_per_sec`를 확인한다. 처리량이 급격히 떨어지면(메모리 초과) 멈추고 보고한다.
- 학습이 끝나면 `mydocs/test_result/task_m100_2_test.md`를 만들어 기사마다 실행 명령, GPU, `config.json`, `baseline` 줄, 모든 `eval` 줄, 마지막 `train` 줄 몇 개, `adapter.pt`의 크기와 SHA-256을 적는다.

### 검증

두 기사에 똑같이 적용한다.

- 학습이 마지막 갱신까지 오류 없이 끝나고 `adapter.pt`, `config.json`, `log.jsonl`이 생긴다.
- `log.jsonl`의 마지막 `eval`의 step이 `config.json`의 `total_steps`와 같다.
- 손실·기울기 노름에 NaN·무한대가 없다.
- 저장된 `adapter.pt`를 새로 올린 베이스에 `load_adapter`로 붙여 val 전체를 다시 평가한 지표가 `log.jsonl`의 마지막 `eval` 값과 일치한다 (저장물이 실제로 쓸 수 있는지 확인).
- 베이스 기준선 대비 val 지표 변화를 표로 보고한다: `imitation`, `top1`, `top3`, `regret`, `regret_top1`, `band_violation`, `strength`. 개선되지 않은 지표도 그대로 적는다.
- 두 기사의 `config.json`이 `player`, `out`, 데이터 크기에 따른 값(`total_steps`, `train_positions`, `val_positions`)을 빼고 동일하다.

### 두 기사 비교

- 기사별 기준선과 최종 val 지표, 기준선 대비 변화를 나란히 놓은 표
- 평가 시점별 추이를 나란히 놓은 표
- 두 기사의 val은 서로 다른 국면이므로 절대값이 아니라 각자의 기준선 대비 변화를 비교한다

### 보고와 결정

보고서에 설정 유지 여부에 대한 의견을 적고 작업지시자의 결정을 받는다. 판단에 볼 것:

- `imitation`이 내려가고 `top1`/`top3`가 올라갔는가, 평가 시점마다 꾸준한가
- 후반 평가에서 `imitation`이 다시 오르는가 (과적합)
- `regret`, `band_violation`이 기준선보다 얼마나 올랐는가
- train 손실과 val 손실의 차이
- 위 경향이 두 기사에서 같은가

설정을 바꾸기로 하면 구현 계획서를 수정해 승인받고, 새 설정으로 다시 학습한다.

### 보고서

`mydocs/working/task_m100_2_stage2.md`

---

## 3단계: 스미슬로프 어댑터 학습

### 목표

2단계에서 확정한 설정으로 스미슬로프의 어댑터를 학습한다.

### 구현

```bash
python -u -m src.train.train --player smyslov --base base_model/136M.pt \
    --out runs/smyslov_136M --eval-every 1000
```

- 8,736 갱신(평가 9회). 로그는 `runs/logs/train_smyslov_136M.log`.
- 결과를 `mydocs/test_result/task_m100_2_test.md`에 2단계와 같은 형식으로 추가한다.

### 검증

2단계의 검증 항목을 똑같이 적용한다. 추가로:

- 세 기사의 `config.json`이 `player`, `out`, 데이터 크기에 따른 값(`total_steps`, `train_positions`, `val_positions`)을 빼고 동일하다.

### 보고서

`mydocs/working/task_m100_2_stage3.md`

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
- 7절의 결과 파일명 예시를 타스크 번호에 맞게 일반화

소스 코드(`src/`, `scripts/`, `tests/`)는 수정하지 않는다.

### 검증

- 종합 표의 모든 수치가 `runs/{player}_136M/log.jsonl`의 값과 일치하는지 스크립트로 대조한다 (손으로 옮겨 적지 않는다).
- `pytest` 전부 통과 (소스 수정이 없었음을 확인).
- `git status`에 미커밋 파일이 없다.

### 보고서

`mydocs/working/task_m100_2_stage4.md`, 이어서 최종 결과보고서 `mydocs/report/task_m100_2_report.md`.

---

## 검증 (전체)

- `pytest` 통과
- 세 기사의 `runs/{player}_136M/adapter.pt`가 있고 새로 올린 베이스에 붙여 평가할 수 있다
- 세 기사가 같은 설정으로 학습되었다
- `mydocs/test_result/task_m100_2_test.md`의 수치가 로그와 일치한다

## 일정과 예상 소요 시간

2~3단계의 시간은 1단계 실측(갱신당 1.33초, val 평가 1회 142~190초)으로 계산한 추정이다.

| 단계 | 내용 | 시간 | 일정 |
|------|------|------|------|
| 1단계 | q 캐시 + 검증 + 확인 학습 | 캐시 2시간 49분 (실측) | 2026-10-11 완료 |
| 2단계 | 탈 약 3.1시간, 카르포프 약 3.9시간 (평가 18회 포함) | 약 7.0시간 | 2026-10-12 시작 |
| 3단계 | 스미슬로프 약 3.7시간 (평가 9회 포함) | 약 3.7시간 | 2단계 승인 후 |
| 4단계 | 종합·문서 | 실행 대기 없음 | 3단계 승인 후 |

## 예상 단계: 4단계
