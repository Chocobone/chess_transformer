# Task #5: 학습 재개 시 데이터 순서 불일치 수정, 학습 기록 정정 — 구현 계획서

## 공통

- 수정 방안은 **A**다 (작업지시자 결정, 2026-10-15). worker 시드용 생성기를 따로 두고, 데이터 순서용 `generator`는 `RandomSampler`만 쓴다.
- GPU 실행은 하지 않는다. 검증은 CPU 테스트와 CPU 시뮬레이션으로 한다.
- 소스 수정은 2단계에서만 한다 (`src/train/train.py`, `tests/test_train.py`).
- 측정·시뮬레이션 스크립트는 스크래치패드에서 실행하고 저장소에 넣지 않는다. 쓴 스크립트의 핵심은 보고서에 옮겨 적는다.
- 기존 문서는 원문을 지우지 않고 "정정 (2026-10-15, Task #5): …"를 붙인다.

**산출물 위치**

```
src/train/train.py, tests/test_train.py                ← 2단계
mydocs/troubleshootings/resume_data_order.md           ← 증상, 원인, 수정, 영향 추정 (1단계에 초안, 3단계에 완성)
mydocs/working/task_m100_5_stage{1,2,3}.md             ← 단계별 보고서
```

---

## 1단계: 기록 대조와 영향 추정

### 목표

세 기사 학습의 중단·재개 사실을 로그에서 정확히 뽑고, 수정 전 코드가 실제로 낸 데이터 순서를 다시 만들어 어댑터가 받은 영향을 국면 단위로 센다.

### 구현

1. **중단·재개 사실 추출**: `runs/logs/train_{tal,karpov,smyslov}_136M.log`에서 시도마다 시작·종료 시각, 종료 코드, 재개 step과 그 epoch, 실패한 시도의 마지막 train 줄 step, 잃은 갱신 수(마지막 train 줄 − 재개 step, 이후 기록되지 않은 최대 19개는 범위로 표시)를 표로 만든다. 겹치는 step의 lr·loss·grad_norm 비교표도 함께 만든다.
2. **수정 전 순서 시뮬레이션**: 데이터셋 길이(탈·카르포프·스미슬로프 train 국면 수), 배치 32, seed 0, 3 epoch로 수정 전 코드가 생성기에서 난수를 뽑는 순서를 그대로 흉내 낸다.
   - 반복자를 만들 때 worker 시드 1개, epoch마다 순열 1개를 뽑는다. `persistent_workers=True`라 한 프로세스 안에서는 반복자를 처음 한 번만 만든다.
   - 재개하면 체크포인트에 저장된 "그 epoch를 시작할 때의 생성기 상태"를 복원하고 반복자를 새로 만들며, 끝낸 배치 수만큼 건너뛴다. 카르포프처럼 두 번 재개한 경우도 시도 순서대로 따라간다.
   - 흉내가 맞는지 먼저 확인한다: 소형 데이터에서 실제 `DataLoader(num_workers=2, persistent_workers=True)`와 순서를 대조한다. 또 재개한 시도의 순서가 끊기지 않은 실행과 다르다는 점이 로그(겹치는 step의 loss 차이)와 맞는지 본다. 배치 내용까지는 로그로 대조할 수 없으므로, 이 부분은 한계로 적는다.
3. **영향 집계**: 기사마다 끊기지 않은 학습(국면마다 3회)과 비교해 아래를 센다.
   - 실제 학습 횟수별 국면 수 (2회, 3회, 4회 …)
   - 재개한 epoch 안에서 두 번 학습된 국면, 한 번도 학습되지 않은 국면의 수와 비율
   - 3 epoch 전체의 갱신 횟수는 계획과 같다는 것 (재개는 갱신 수를 바꾸지 않는다)
4. 결과로 트러블슈팅 문서 초안을 쓴다.

### 완료 기준

- 세 기사의 중단·재개 표와 영향 집계표가 나온다. 수치는 스크립트 출력에서 옮긴다.
- 흉내와 실제 `DataLoader`의 순서가 소형 데이터에서 같다.

---

## 2단계: 재개 순서 수정과 테스트

### 목표

데이터 순열이 재개 여부와 `--num-workers`에 따라 달라지지 않게 한다. 테스트를 먼저 넣고, 지금 코드에서 실패하는지 확인한 뒤 고친다.

### 테스트 추가 (`tests/test_train.py`)

1. `test_resume_matches_uninterrupted`의 기존 경우들(`--num-workers 0`)은 그대로 둔다. `--num-workers 2`로 epoch 경계 직후와 둘째 epoch 중간에서 죽는 경우(예: `[9]`, `[12]`, `[6, 12]`)를 추가한다. 비교 기준은 `--num-workers 0`으로 끊기지 않고 돈 실행이다.
2. `num_workers` 0과 2의 끊기지 않은 실행이 로그와 어댑터 모두 같은지 확인한다.
3. 옛 순서 방식의 체크포인트(아래 표시가 없는 것)로 끝나지 않은 학습을 재개하면 거부하는지 확인한다.

1·2는 지금 코드에서 실패해야 한다. 실패를 확인한 뒤 수정한다.

### 수정 (`src/train/train.py`)

```python
generator = torch.Generator().manual_seed(args.seed)          # 데이터 순서 (RandomSampler만 사용)
worker_seeds = torch.Generator().manual_seed(args.seed)       # DataLoader의 worker 시드. 순서에 영향 없음
train_sampler = ResumableBatchSampler(train_set, args.batch_positions, generator)
train_loader = DataLoader(..., generator=worker_seeds, ...)
```

- 데이터셋과 collate에는 난수를 쓰는 곳이 없으므로 worker 시드는 결과에 영향이 없다. 그래서 `worker_seeds`의 상태는 체크포인트에 넣지 않는다.
- 체크포인트에 `"data_order": 2`를 넣는다. 이 표시가 없는 체크포인트로 끝나지 않은 학습을 재개하려 하면 "수정 전 순서 방식의 체크포인트다"로 거부한다. 이미 끝난 학습(`step >= total_steps`)은 지금처럼 갱신 없이 끝낸다.
- `--num-workers`는 계속 `RESUME_FREE_ARGS`에 둔다. 수정 뒤에는 실제로 결과에 영향이 없다.
- 이 수정으로 epoch 0부터 순서가 수정 전과 달라진다. 수정 전 어댑터의 배치 순서는 재현할 수 없게 된다 (승인된 단점).
- `ResumableBatchSampler`의 docstring과 체크포인트 관련 주석에서 "`DataLoader(shuffle=True, generator=...)`와 같은 순서"처럼 더는 맞지 않는 설명을 고친다.

### 완료 기준

- 새 테스트가 수정 전 코드에서 실패하고 수정 뒤 통과한다.
- 전체 `pytest`가 통과한다 (기존 111건 + 추가분).
- 바뀐 것이 데이터 순열뿐임을 확인한다: `num_workers=0` 소형 실행에서 수정 전·후의 baseline 줄과 첫 배치의 국면 수 등 순서와 무관한 값이 같다.

---

## 3단계: 기존 기록 정정, 문서 완성

### 목표

틀린 기술을 바로잡고, 트러블슈팅 문서를 완성한다.

### 정정 대상

| 문서 | 정정 내용 |
|------|------|
| `mydocs/working/task_m100_4_stage2.md`, `mydocs/report/task_m100_4_report.md`, `mydocs/test_result/task_m100_4_test.md`, `mydocs/orders/20261014.md` | 스미슬로프 중단 지점 "step 5,400~5,420, 잃은 갱신 20개 이내" → 1단계 결과. 잃은 시간 "약 4분"이 맞는지 로그의 elapsed로 다시 계산 |
| `mydocs/working/task_m100_2_stage2.md`, `mydocs/report/task_m100_2_report.md`, `mydocs/test_result/task_m100_2_test.md`, `mydocs/orders/20261012.md` | 탈 중단 지점 "step 4800~4820" → 1단계 결과 |
| `mydocs/manual/training_guide.md` | "재개한 학습의 결과는 끊기지 않은 학습과 같다"에 수정 전 코드에서는 epoch 1 이후에 성립하지 않았다는 점과 수정 내용을 덧붙인다. 옛 체크포인트를 거부한다는 점을 적는다 |
| `mydocs/report/task_m100_2_report.md` 등 재개 동일성을 주장한 곳 | 같은 취지로 정정 |
| Task #3 (카르포프) 보고서·테스트 결과 | 재개 구간의 순서가 달랐다는 점과 영향 추정을 덧붙인다 |

- 대상은 실제 정정 전에 `grep`으로 다시 훑어 빠진 곳이 없게 한다. 위 표에 없는 곳이 나오면 단계 보고서에 적는다.
- `mydocs/troubleshootings/resume_data_order.md`를 완성한다: 증상, 원인, 확인 방법, 수정, 세 어댑터가 받은 영향, 남은 한계 (CUDA 오류 원인 미규명, 영향 실측 안 함).

### 완료 기준

- 정정 대상 목록과 실제 정정 내역을 단계 보고서에 표로 남긴다.
- `grep`으로 다시 찾았을 때 정정 표시 없는 틀린 기술이 남아 있지 않다.

---

## 위험과 대응

| 위험 | 대응 |
|------|------|
| `num_workers=2` 테스트가 CPU에서 느리거나 불안정함 | 추가 경우를 3개 안팎으로 줄이고 소요 시간을 보고한다. 문제가 되면 단계 보고서에서 다른 방식을 제안한다 |
| 시뮬레이션이 실제 수정 전 순서와 다름 | 소형 데이터에서 실제 `DataLoader`와 대조한 뒤에만 집계한다. 맞지 않으면 집계하지 않고 보고한다 |
| 정정 범위가 넓어 누락 | `grep` 목록을 단계 보고서에 그대로 남긴다 |
