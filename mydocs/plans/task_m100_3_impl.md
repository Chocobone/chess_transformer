# Task #3: 카르포프 LoRA 어댑터 본 학습 (136M) — 구현 계획서

## 공통

이 타스크는 Task #1·#2의 코드를 실행하는 작업이다. 소스(`src/`, `scripts/`, `tests/`)는 수정하지 않는다. 실행 중 코드 결함이 드러나면 실행을 멈추고 원인과 수정안을 보고한 뒤 승인을 받아 수정한다.

**실행 환경**: 로컬 개발 컨테이너, RTX 4070 Ti 12GB, 베이스 `base_model/136M.pt`. 코드는 `local/task3`(소스는 `06b57fa` 이후 변경 없음).

**산출물 위치**

```
runs/karpov_136M/                         ← adapter.pt, checkpoint.pt, config.json, log.jsonl (git 제외)
runs/logs/train_karpov_136M.log           ← 학습의 표준 출력, 시도별 시작·종료 기록 (git 제외)
runs/logs/gpu_mem_karpov_136M.log         ← GPU 전체 사용량, 60초마다 (git 제외)
mydocs/test_result/task_m100_3_test.md    ← 학습 결과 정리 (3단계)
```

**진행 순서에 관한 메모**: 2026-10-13 작업지시자 지시로 사전 확인(1단계)을 마치고 학습(2단계)을 01:42:04에 이미 시작했다. 이 계획서는 학습이 도는 동안 작성한다. 1단계 보고서는 이 계획서 승인 후 작성한다.

---

## 1단계: 사전 확인

### 목표

학습을 시작해도 되는 상태인지 확인한다.

### 구현

소스 수정 없음. 아래를 확인한다 (2026-10-13 01:41 수행).

```bash
python -m pytest -q
sha256sum data/processed/karpov/base_q_136M.npz
nvidia-smi
bash -n runs/logs/train_player.sh
```

### 검증

- `pytest` 111건 통과
- 캐시의 SHA-256이 Task #2 1단계 보고서의 값(`4606d36323b43a79e43fe3c0ac9fc1bde7170bb840afceb8ea3ec1440c7ed8a3`)과 같다
- GPU에 다른 학습·캐시 프로세스가 없다
- `runs/karpov_136M/`과 `runs/logs/*karpov*`가 없다 (이전 실행의 잔여물 없이 처음부터 시작)
- 소스가 탈 학습 때와 같다 (`git diff 06b57fa -- src scripts tests`가 비어 있다)

### 보고서

`mydocs/working/task_m100_3_stage1.md` — 확인 결과와 학습 시작 시각, 초반 기록(기준선, 처리량, 최대 GPU 메모리).

---

## 2단계: 카르포프 어댑터 학습

### 목표

카르포프의 어댑터를 3 epoch(9,039 갱신) 끝까지 학습한다.

### 구현

```bash
setsid nohup runs/logs/train_player.sh karpov > runs/logs/train_karpov_136M.log 2>&1 < /dev/null &
```

스크립트가 실행하는 명령:

```bash
python -u -m src.train.train --player karpov --base base_model/136M.pt \
    --out runs/karpov_136M --eval-every 1000 --resume
```

- 평가는 1000 갱신마다 9회와 종료 시 1회, 합계 10회. 체크포인트는 200 갱신마다와 평가 때마다.
- 오류로 끝나면 스크립트가 20초 뒤 마지막 체크포인트에서 재개한다 (최대 10회, 진전 없이 연속 3회 실패하면 중단).
- 학습 중 `max_memory_gib`와 `positions_per_sec`를 확인한다. 처리량이 급격히 떨어지면(메모리 초과) 멈추고 보고한다.
- 중단이 있으면 시도별 시작·종료 시각, 중단 step, 오류 메시지, 그때의 GPU 사용량을 기록한다.
- 재시도 한도 안에 끝나지 못하면 멈춘 상태로 두고 보고한다. 한도나 설정을 임의로 바꾸지 않는다.

### 검증

- 로그에 `=== karpov 완료` 줄이 있고 종료 코드가 0이다
- `runs/karpov_136M/`에 `adapter.pt`, `checkpoint.pt`, `config.json`, `log.jsonl`이 있다
- `log.jsonl`의 마지막 `eval`의 step이 `config.json`의 `total_steps`(9,039)와 같다

### 보고서

`mydocs/working/task_m100_3_stage2.md` — 실행 경과(시작·종료 시각, 시도 횟수, 중단 기록), 소요 시간, 처리량, 최대 GPU 메모리와 GPU 전체 사용량의 범위.

---

## 3단계: 학습 검증과 결과 정리

### 목표

학습 결과가 온전한지 확인하고, 베이스 기준선 대비 val 지표 변화를 문서로 남긴다.

### 구현

**검증 스크립트** (스크래치패드에서 실행, 저장소에 넣지 않는다): `log.jsonl`·`config.json`을 읽어 아래 검증 항목을 확인하고 결과 문서의 표를 출력한다.

**`mydocs/test_result/task_m100_3_test.md`** — 신규. `task_m100_2_test.md`의 탈 부분과 같은 형식:

- 실행 명령, GPU, `config.json`
- `baseline` 줄, 모든 `eval` 줄, 마지막 `train` 줄 몇 개 (로그 원문)
- 베이스 기준선 대비 최종 val 지표 표: `imitation`, `top1`, `top3`, `regret`, `regret_top1`, `band_violation`, `strength`
- `adapter.pt`의 크기와 SHA-256
- 이 지표로 말할 수 있는 것과 없는 것

### 검증

- `log.jsonl`의 train 줄에 빠지거나 겹친 step이 없다 (20 간격, 마지막 줄 포함)
- 손실·기울기 노름에 NaN·무한대가 없다
- 저장된 `adapter.pt`를 새로 올린 베이스에 `load_adapter`로 붙여 val 5,213국면 전체를 다시 평가한 지표가 `log.jsonl`의 마지막 `eval` 값과 일치한다
- 카르포프와 탈의 `config.json`이 `player`, `out`, 데이터 크기에 따른 값(`total_steps`, `train_positions`, `val_positions`)을 빼고 같다
- 결과 문서의 표 수치가 `log.jsonl`의 값과 일치한다 (스크립트 출력을 그대로 쓴다)
- `pytest` 통과, `git status`에 미커밋 파일 없음

### 보고서

`mydocs/working/task_m100_3_stage3.md`, 이어서 최종 결과보고서 `mydocs/report/task_m100_3_report.md`.

---

## 검증 (전체)

- `pytest` 통과
- `runs/karpov_136M/adapter.pt`가 있고 새로 올린 베이스에 붙여 평가할 수 있다
- 탈과 같은 설정, 같은 코드로 학습되었다
- `mydocs/test_result/task_m100_3_test.md`의 수치가 로그와 일치한다

## 일정과 예상 소요 시간

| 단계 | 내용 | 시간 |
|------|------|------|
| 1단계 | 사전 확인 | 완료 (2026-10-13 01:41) |
| 2단계 | 학습 9,039 갱신 + 평가 10회 | 약 3.9시간 (추정, 중단이 없을 때). 01:42:04 시작이므로 05:35 전후 종료 예상 |
| 3단계 | 검증(재평가 약 4분 포함)·문서 | 실행 대기 거의 없음 |

## 예상 단계: 3단계
