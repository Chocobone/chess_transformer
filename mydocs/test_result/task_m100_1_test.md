# Task #1 테스트 결과: 베이스 모델 PyTorch 이식 검증

## 실행 환경

- 실행 일자: 2026-10-07
- 실행 위치: 로컬 개발 컨테이너 (GPU 서버 아님). 실행 시점에 컨테이너에 GPU가 연결되어 있지 않아 **원본 JAX와 PyTorch 모두 CPU**로 실행했다.
- PyTorch 쪽: Python 3.13.16, torch 2.14.1+cpu
- 원본 쪽: google-deepmind/searchless_chess 커밋 `90ae0e6`, jax 0.11.2, dm-haiku 0.0.17, orbax-checkpoint 0.12.7 (세션 임시 가상환경)
- 체크포인트: 공개 9M / 136M / 270M, step 6,400,000, `params` (EMA 아님 — 원본 엔진과 같다)

## 절차

크기마다 아래를 실행했다 (136M 예시).

```
python scripts/export_haiku_params.py --checkpoint <체크포인트>/136M --out base_model/136M_params.npz
PYTHONPATH=<searchless_chess의 부모 폴더> python scripts/dump_jax_reference.py \
    --params base_model/136M_params.npz --size 136M --out base_model/reference_136M.npz
python -m src.model.convert --params base_model/136M_params.npz --size 136M --out base_model/136M.pt
python scripts/verify_port.py --base base_model/136M.pt --reference base_model/reference_136M.npz
```

비교 대상은 `scripts/reference_fens.txt`의 53개 국면, 합법수 1,536개다. 기준 파일의 입력 토큰과 합법수 순서는 원본 토크나이저·행동 표·엔진 코드로 만든 것이다.

## 결과 요약

| 항목 | 9M | 136M | 270M |
|------|-----|------|------|
| 체크포인트 텐서 수 / 파라미터 수 | 94 / 8,954,240 | 94 / 136,479,872 | 182 / 270,730,368 |
| 변환 시 키 누락·잉여 | 0 | 0 | 0 |
| 합법수 순서 일치 | 일치 | 일치 | 일치 |
| 입력 토큰 일치 | 일치 | 일치 | 일치 |
| 구간 로그확률 최대 절대 오차 | 4.578e-05 | 6.485e-05 | 4.768e-05 |
| 구간 확률 최대 절대 오차 | 4.143e-06 | 8.047e-06 | 7.629e-06 |
| 승률 최대 절대 오차 | 1.580e-06 | 1.073e-06 | 1.490e-06 |
| 최선수 일치 | 53/53 | 53/53 | 53/53 |
| 판정 (허용치 1e-4) | 통과 | 통과 | 통과 |

## 추가 대조 (원본 모듈과 직접 비교)

- 행동 표: `MOVE_TO_ACTION`, `ACTION_TO_MOVE` 1,968개 전부가 원본 `utils`와 동일
- 토크나이저: 세 기사 전처리 데이터의 282,910개 국면 전부에서 원본 `tokenizer.tokenize`와 동일 (불일치 0건)

## 스크립트 출력 원문

### 9M

```
국면 53개, 합법수 1536개
합법수 순서 일치: True
입력 토큰 일치: True
구간 로그확률 최대 절대 오차: 4.578e-05
구간 확률 최대 절대 오차: 4.143e-06
승률 최대 절대 오차: 1.580e-06
최선수 일치: 53/53
결과: 통과
```

### 136M

```
국면 53개, 합법수 1536개
합법수 순서 일치: True
입력 토큰 일치: True
구간 로그확률 최대 절대 오차: 6.485e-05
구간 확률 최대 절대 오차: 8.047e-06
승률 최대 절대 오차: 1.073e-06
최선수 일치: 53/53
결과: 통과
```

### 270M

```
국면 53개, 합법수 1536개
합법수 순서 일치: True
입력 토큰 일치: True
구간 로그확률 최대 절대 오차: 4.768e-05
구간 확률 최대 절대 오차: 7.629e-06
승률 최대 절대 오차: 1.490e-06
최선수 일치: 53/53
결과: 통과
```

## GPU 재검증 (PyTorch 쪽만 GPU)

컨테이너 재생성으로 GPU가 연결된 뒤 같은 기준 파일(`base_model/reference_*.npz`, 원본 JAX CPU 출력)에 대해 PyTorch 모델을 GPU에서 다시 실행했다.

- 실행 일자: 2026-10-07
- 환경: NVIDIA GeForce RTX 4070 Ti (12GB), torch 2.14.1+cu130
- 명령: `python scripts/verify_port.py --base base_model/{size}.pt --reference base_model/reference_{size}.npz --device cuda`

| 항목 | 9M | 136M | 270M |
|------|-----|------|------|
| 합법수 순서·입력 토큰 일치 | 일치 | 일치 | 일치 |
| 구간 로그확률 최대 절대 오차 | 6.771e-05 | 8.011e-05 | 5.102e-05 |
| 구간 확률 최대 절대 오차 | 5.245e-06 | 9.120e-06 | 6.557e-06 |
| 승률 최대 절대 오차 | 1.669e-06 | 1.281e-06 | 1.609e-06 |
| 최선수 일치 | 53/53 | 53/53 | 53/53 |
| 판정 (허용치 1e-4) | 통과 | 통과 | 통과 |

로그확률 오차는 CPU보다 조금 크지만(최대 8.0e-05) 허용치 안이다.

## 한계

- 원본 JAX 기준 출력은 CPU에서 만든 것이다. PyTorch 쪽은 CPU와 GPU 양쪽에서 이 기준과 비교했다.
- 비교한 국면은 53개다. 토큰화와 행동 번호는 전수 대조했지만 모델 출력은 이 53개 국면에서만 대조했다.
