# Task #5 2단계 완료 보고서: 재개 순서 수정과 테스트

## 요약

- 방안 A로 수정했다. worker 시드용 생성기를 따로 두고, 데이터 순서용 생성기는 샘플러만 쓴다.
- 체크포인트에 `"data_order": 2` 표시를 넣었다. 표시가 없는 체크포인트로 끝나지 않은 학습을 재개하면 거부한다 (작업지시자 결정).
- 테스트 5건을 먼저 추가했다. 수정 전 코드에서는 5건 모두 실패했고, 수정 뒤에는 전체 116건(기존 111 + 추가 5)이 통과했다.
- 수정 전·후의 소형 학습을 비교했다. 바뀐 것은 배치 순서(와 그에 따른 loss 등)뿐이다.

## 1. 테스트 (`tests/test_train.py`)

기존 `test_resume_matches_uninterrupted`의 본문을 두 도우미로 나눴다. `resume_after_crashes`는 지정한 갱신에서 죽였다가 끝까지 재개하고, `assert_same_adapter`는 두 실행의 어댑터 가중치가 같은지 본다. 기존 8개 경우(`--num-workers 0`)는 그대로다.

| 추가 테스트 | 내용 | 수정 전 코드 | 수정 뒤 |
|------|------|------|------|
| `test_num_workers_does_not_change_result` | `--num-workers 2`로 끊기지 않고 돈 학습이 `0`과 로그·어댑터 모두 같다 | 실패 (step 9, epoch 1 첫 갱신부터 다름) | 통과 |
| `test_resume_with_workers_matches_uninterrupted[9]` | `--num-workers 2`, epoch 경계(체크포인트 8)에서 재개 | 실패 (step 9부터 다름) | 통과 |
| `…[12]` | 둘째 epoch 중간(체크포인트 9)에서 재개 | 실패 | 통과 |
| `…[6, 12]` | 두 번 재개 | 실패 | 통과 |
| `test_resume_rejects_checkpoint_of_old_data_order` | 표시를 지운 체크포인트(step 3)로 재개하면 `ValueError`, 끝난 학습(step 16)은 표시가 없어도 갱신 없이 끝나고 로그가 그대로 | 실패 (`data_order` 키 없음) | 통과 |

- 비교 기준은 기존 `uninterrupted` 픽스처(`--num-workers 0`, 끊기지 않은 학습)다. 수정 뒤에는 `num_workers`와 재개 여부에 관계없이 같은 결과가 나와야 하므로 기준 하나로 충분하다.
- 처음 작성한 거부 테스트는 끝난 학습의 파일을 다른 폴더로 복사해 썼다. 그런데 체크포인트 설정의 `out`이 달라 설정 불일치로 거부되는 문제가 있었다. 그 자리에서 끝까지 돈 학습을 쓰도록 고쳤다.
- 전체 `pytest` 시간은 33~34초다 (추가 전 약 21초가 `test_train.py`). `num_workers=2` 테스트에서 fork 관련 `DeprecationWarning`이 나오지만 결과에는 영향이 없다.

## 2. 수정 (`src/train/train.py`)

```diff
 class ResumableBatchSampler(Sampler):
-    """`DataLoader(shuffle=True, generator=...)`와 같은 순서로 배치를 낸다.
+    """`generator`로 epoch마다 섞은 순서로 배치를 낸다. 이 생성기는 순서에만 쓴다.

+# 체크포인트의 데이터 순서 방식. 2: 순서용 생성기를 샘플러만 쓴다 (Task #5).
+# 표시가 없는 체크포인트는 DataLoader가 같은 생성기에서 worker 시드를 뽑던 방식이라,
+# num_workers > 0에서 epoch 1 이후에 재개하면 순서가 끊기지 않은 학습과 달랐다.
+DATA_ORDER = 2

     generator = torch.Generator().manual_seed(args.seed)
     train_sampler = ResumableBatchSampler(train_set, args.batch_positions, generator)
+    # DataLoader는 반복자를 만들 때마다 자기 생성기에서 worker 시드를 뽑는다. ...
+    worker_seeds = torch.Generator().manual_seed(args.seed)
     train_loader = DataLoader(
         train_set, batch_sampler=train_sampler, collate_fn=collate_fn,
-        num_workers=args.num_workers, generator=generator, persistent_workers=args.num_workers > 0,
+        num_workers=args.num_workers, generator=worker_seeds, persistent_workers=args.num_workers > 0,
     )

         resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
+        if resumed.get("data_order") != DATA_ORDER and resumed["step"] < total_steps:
+            raise ValueError(f"{checkpoint_path}는 수정 전 데이터 순서 방식의 체크포인트다. ...")

-            "step": step, **position, "metrics": metrics, ...
+            "step": step, "data_order": DATA_ORDER, **position, "metrics": metrics,
```

- 데이터 처리(`PositionDataset`, `collate_fn`)에는 난수가 없다. 그래서 worker 시드는 결과에 영향이 없고, `worker_seeds`의 상태는 체크포인트에 넣지 않았다.
- `RESUME_FREE_ARGS`는 바꾸지 않았다. `--num-workers`는 이제 실제로 결과에 영향이 없다 (`test_num_workers_does_not_change_result`).
- 모듈 docstring의 "결과는 끊기지 않은 학습과 같다"는 수정 뒤에 맞으므로 그대로 두었다.

## 3. 바뀐 것이 데이터 순서뿐인지 확인

테스트용 소형 데이터에서 같은 학습(`--num-workers 0`, 2 epoch, 16 갱신)을 수정 전·후 코드로 한 번씩 돌려 비교했다. 스크립트는 스크래치패드에 두었고, 테스트 픽스처를 재사용했다.

| 항목 | 수정 전·후 |
|------|------|
| baseline 줄 (학습 전 val 지표) | 같음 |
| train 줄의 step·epoch·lr | 같음 |
| epoch별 학습한 국면 집합 (모두 1회) | 같음 |
| epoch별 배치 순서 | 다름 (epoch 0부터) |
| loss 등 학습 수치 | 다름 (순서가 달라서) |

수정 전에는 `num_workers=0`에서도 epoch마다 worker 시드를 순서용 생성기에서 뽑았다. 그래서 수정 뒤에는 `num_workers=0`도 epoch 0부터 순서가 바뀐다. 이는 승인된 단점(수정 전 어댑터의 배치 순서를 재현할 수 없음)과 같은 내용이다.

## 4. 문서

`mydocs/troubleshootings/resume_data_order.md`에 "수정" 절을 채웠다. 기존 기록 정정과 학습 가이드 수정은 3단계에서 한다.

## 다음 단계

3단계: `grep`으로 정정 대상을 다시 훑은 뒤 기존 기록(Task #2·#3·#4, 학습 가이드)을 정정하고, 트러블슈팅 문서를 완성한다.
