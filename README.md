# 체스 기사 기풍 주입 — Transformer 체스엔진 LoRA 파인튜닝

탐색 없이 한 번의 순전파로 수를 고르는 Transformer 체스 모델(DeepMind *Grandmaster-Level Chess Without Search*, 2024)에, 세 기사의 기풍을 각각 주입하는 연구 저장소다. 대상은 **미하일 탈 · 아나톨리 카르포프 · 바실리 스미슬로프**다. 기력은 유지하면서 기풍을 넣고, 세 기풍의 차이를 같은 척도에서 비교하는 것이 목표다.

- 연구 제안서: [`docs/캡스톤 제안서 — Transformer 체스엔진 기반 기풍 스타일 주입.md`](docs/캡스톤%20제안서%20—%20Transformer%20체스엔진%20기반%20기풍%20스타일%20주입.md)
- 기술 스택: Python 3.11+, PyTorch, python-chess

## 현재 상태 (2026-10-15, M100)

| 단계 | 상태 |
|---|---|
| 기보 전처리, 베이스 모델 PyTorch 이식, LoRA 학습 코드 | 완료 (Task #1) |
| 136M 베이스 위 기사별 LoRA 어댑터 학습 (탈·카르포프·스미슬로프) | 완료 (Task #2~#4) |
| 학습 재개 시 데이터 순서 결함 수정, 학습 기록 정정 | 완료 (Task #5) |
| 세 어댑터 결과 종합, 교차 평가, 실행 가이드 갱신 | 완료 (Task #6) |
| 스타일 특성 벡터 f(s,a), 방향 벡터 w_p (제안서 4.1~4.2) | 예정 |
| α 스윕, 지문 거리·혼동 행렬·대국 성적 평가 (제안서 6장) | 예정 |
| 270M 베이스 학습 | 예정 |

지금 어댑터는 **모방 항과 기력 항만으로 학습한 것(α = 0)**이다. 스타일 항은 스타일 특성 벡터가 생긴 뒤에 들어간다.

## 방법

```
기사 기보(PGN) ──전처리──▶ 국면·기사의 수 ──┐
                                            ├─▶ LoRA 어댑터 학습 ──▶ 기사별 어댑터
베이스 모델(동결) ──q(s,a) 캐시──────────────┘
```

- **베이스**: searchless_chess의 action-value Transformer를 PyTorch로 이식했다 (`src/model/`). (국면, 수)를 받아 128개 승률 구간 분포를 내고, 그 기댓값을 q(s, a)로 쓴다. 9M·136M·270M 공개 체크포인트에서 원본 JAX 출력과 일치함을 확인했다.
- **정책**: π(a|s) = softmax(q(s, a) / T), T = 0.02. 합법수 전체에 대해 계산한다.
- **어댑터**: 어텐션 `q k v o`에 LoRA(rank 16, alpha 32)를 넣는다. 136M에서 학습 파라미터는 1,048,576개다. 베이스는 동결한다.
- **손실** (제안서 4.3): L = L_imit + α·L_style + λ·L_strength
  - L_imit: 기사가 둔 수의 교차엔트로피
  - L_strength: Σ π(a|s)·max(0, regret(s,a) − τ). regret은 동결 베이스의 q 기준이다 (λ = 1, τ = 0.02).
  - L_style: 스타일 특성 벡터가 생긴 뒤 추가한다 (지금 α = 0).
- **데이터**: 고전 대국만 쓴다(속기·다면기·블라인드 등 제외). 기사가 둔 수의 직전 국면이 학습 표본이다. val은 국 단위 5%(seed 0)다.

## 결과 요약 (136M, α = 0)

상세는 [`mydocs/tech/adapters_136M_summary.md`](mydocs/tech/adapters_136M_summary.md)에 있다.

| | 탈 | 카르포프 | 스미슬로프 |
|---|---|---|---|
| 국면 train / val | 79,493 / 3,880 | 96,393 / 5,213 | 93,178 / 4,753 |
| val imitation (기준선 → 어댑터) | 1.437 → 1.066 (−25.8%) | 1.494 → 1.152 (−22.9%) | 1.450 → 1.094 (−24.6%) |
| val top1 (기사 수를 1순위로 맞힌 비율) | 60.6% → 65.4% | 55.9% → 60.6% | 56.4% → 63.3% |
| regret_top1 (최선수가 잃는 승률) | 0 → 0.0029 | 0 → 0.0027 | 0 → 0.0029 |
| 학습 시간 (RTX 4070 Ti 12GB) | 3시간 8분 | 7시간 18분* | 3시간 37분 |

\* 학습 중 평가가 GPU 메모리를 넘쳐 느려졌다. 대책(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)을 쓰면 3시간대다.

**교차 평가:** 각 어댑터를 세 기사의 val에 평가했다 (imitation, 낮을수록 좋음).

| 어댑터 \ val | 탈 | 카르포프 | 스미슬로프 |
|---|---|---|---|
| 베이스 | 1.437 | 1.494 | 1.450 |
| 탈 | **1.066** | 1.230 | 1.190 |
| 카르포프 | 1.174 | **1.152** | 1.198 |
| 스미슬로프 | 1.157 | 1.218 | **1.094** |

- 세 val 모두에서 자기 기사의 어댑터가 가장 좋다. 대국 단위 부트스트랩 95% 구간에서도, 오프닝을 뺀 ply 20 이상에서도 같다.
- 다만 개선의 72~79%는 세 어댑터에 공통이다. 기사 특이 몫은 21~28%이고, ply 20 이상에서는 11~22%다.
- 기력 쪽 지표 중 나빠진 것은 `regret_top1`뿐이다. regret 기댓값은 기준선보다 좋아졌다.
- 이 결과는 모방 지표다. 기풍이 분리됐다는 증거(지문 거리·혼동 행렬)는 스타일 특성 벡터 이후에 다룬다.

## 빠른 시작

전체 절차와 옵션, 실측 시간, 자주 나는 오류는 [`mydocs/manual/training_guide.md`](mydocs/manual/training_guide.md)에 있다.

```bash
# 1. 환경 (torch는 CUDA 버전에 맞는 휠을 먼저 설치)
pip install -r requirements.txt
pytest                                   # 116건 통과

# 2. 베이스 가중치: base_model/{136M,270M}.pt 를 준비한다 (가이드 2절. 원본은 base_model/download.sh)
python scripts/verify_port.py --base base_model/136M.pt --reference base_model/reference_136M.npz

# 3. 전처리와 베이스 q 캐시 (기사마다)
python -m src.data.preprocess --player tal
python scripts/cache_base_q.py --player tal --base base_model/136M.pt

# 4. 학습 (중단되면 같은 명령으로 이어서 돈다)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m src.train.train --player tal --base base_model/136M.pt \
    --out runs/tal_136M --eval-every 1000 --resume
```

학습된 어댑터 사용:

```python
from src.model.lora import load_adapter
from src.model.transformer import load_base

model = load_base("base_model/136M.pt", "cuda")
load_adapter(model, "runs/tal_136M/adapter.pt")
model.eval()
```

`base_model/*.pt`, `data/`, `runs/`는 git에 올리지 않는다.

## 저장소 구조

```
├── src/
│   ├── players.py            기사별 설정 (PGN 경로, 이름 식별)
│   ├── data/                 PGN 전처리, 학습 데이터셋 (국면 + 베이스 q 캐시)
│   ├── model/                토크나이저, 행동 사전, Transformer 이식, 가중치 변환, LoRA
│   └── train/                정책, 손실, 학습 스크립트 (평가·체크포인트·재개)
├── scripts/                  베이스 q 캐시, 이식 검증, JAX 가중치 추출
├── tests/                    pytest (전처리, 모델 이식, LoRA, 손실, 학습·재개)
├── player_pgn/               탈·카르포프·스미슬로프 기보
├── base_model/               베이스 체크포인트 (download.sh만 git에 있음)
├── docker/                   로컬 GPU 개발 컨테이너
├── docs/                     연구 제안서, 참고 논문
└── mydocs/                   개발 문서 (아래)
```

## 문서

| 문서 | 내용 |
|---|---|
| [`mydocs/manual/training_guide.md`](mydocs/manual/training_guide.md) | 학습 실행 가이드 |
| [`mydocs/tech/adapters_136M_summary.md`](mydocs/tech/adapters_136M_summary.md) | 세 어댑터 종합: 지표, 곡선, 조건 차이, 교차 평가, 한계 |
| [`mydocs/troubleshootings/resume_data_order.md`](mydocs/troubleshootings/resume_data_order.md) | 학습 재개 시 데이터 순서 결함 (Task #5) |
| `mydocs/report/task_m100_{N}_report.md` | 타스크별 최종 결과보고서 |
| `mydocs/test_result/task_m100_{N}_test.md` | 학습·평가 실행 기록 |
| `mydocs/plans/`, `mydocs/working/` | 타스크별 계획서, 단계별 보고서 |
| `mydocs/orders/yyyymmdd.md` | 날짜별 작업일지 |

### 타스크 이력 (M100)

| 이슈 | 타스크 | 보고서 |
|---|---|---|
| #1 | 기보 전처리, 베이스 이식, LoRA 학습 코드 | [`task_m100_1_report.md`](mydocs/report/task_m100_1_report.md) |
| #2 | 탈 136M 어댑터 학습 (재개 기능 추가) | [`task_m100_2_report.md`](mydocs/report/task_m100_2_report.md) |
| #3 | 카르포프 136M 어댑터 학습 | [`task_m100_3_report.md`](mydocs/report/task_m100_3_report.md) |
| #4 | 스미슬로프 136M 어댑터 학습 (평가 메모리 대책) | [`task_m100_4_report.md`](mydocs/report/task_m100_4_report.md) |
| #5 | 학습 재개 데이터 순서 수정, 기록 정정 | [`task_m100_5_report.md`](mydocs/report/task_m100_5_report.md) |
| #6 | 세 어댑터 결과 종합, 교차 평가, 가이드 갱신 | [`task_m100_6_report.md`](mydocs/report/task_m100_6_report.md) |

## 알려진 한계

- 세 어댑터는 Task #5 수정 전 코드로 학습되었다. 중단 뒤 재개한 구간의 데이터 순서가 끊기지 않은 학습과 달랐다. 재개한 epoch에서 국면의 0.8~12.5%가 두 번 학습되고, 같은 수가 한 번도 학습되지 않았다. 성능 영향은 실측하지 않았다.
- 로컬 GPU(WSL2)에서 학습 도중 원인 불명의 CUDA 오류(device-side assert)가 났다 (탈 3회, 스미슬로프 1회). 가중치는 오염되지 않고 자동 재개로 이어진다. 원인 조사는 보류 중이다.
- 결과는 136M 베이스, seed 하나에서 나왔다.

## 개발 방식

이 저장소는 **하이퍼-워터폴(Hyper-Waterfall)** 방법론으로 개발한다. 방법론 설명은 [`mydocs/manual/hyper_waterfall.md`](mydocs/manual/hyper_waterfall.md)에, Claude Code용 지시는 [`CLAUDE.md`](CLAUDE.md)에 있다.

- 절차: GitHub 이슈 → `local/task{N}` 브랜치 → 수행 계획서 → 구현 계획서 → 단계별 구현·보고서 → 최종 보고서. 단계마다 작업지시자 승인을 받는다.
- 브랜치: `main`(릴리즈), `devel`(통합, push), `local/devel`·`local/task{N}`(로컬 전용)
- 커밋 메시지: `Task #N: 내용`
- 문서는 모두 한국어로 쓴다. 기존 기록은 지우지 않고 "정정 (날짜, Task #N)" 표시로 고친다.

## 참고

- Ruoss et al., *Grandmaster-Level Chess Without Search*, 2024 — [`docs/Deepmind chess_transformer.pdf`](docs/Deepmind%20chess_transformer.pdf)
- 베이스 체크포인트: google-deepmind/searchless_chess (`base_model/download.sh`)
