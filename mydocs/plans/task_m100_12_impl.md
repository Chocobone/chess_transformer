# Task #12: 특성 없는 평가 (A1~A3, B2·B3·B10, C3) — 구현 계획서

> **상태 (2026-11-12)**: 승인 (해석 7건 포함)
> 수행 계획서: `mydocs/plans/task_m100_12.md` (승인. 결정 4건 권장안)

## 공통

- **기존 파일은 고치지 않는다.** `src/`에는 새 패키지 `src/eval/`만 더한다. 기존 함수(`policy_from_q`, `PositionDataset`, `collate_fn`, `split_by_moves`, `load_base`, `load_adapter`, `evaluate`)는 불러 쓴다
- **덤프(GPU, torch)와 분석(CPU, numpy만)을 파일로 나눈다.** 분석 모듈은 torch를 불러오지 않는다. 그래서 torch가 없는 이 PC의 호스트 Python에서도 돌릴 수 있다
- 결과물은 git 제외 경로 `runs/gpu/dump/`(덤프)와 `runs/gpu/judge/`(분석 결과)에 둔다
- 로컬 테스트는 Task #11과 같이 개발 컨테이너 이미지(`chess-transformer-dev`)를 GPU 없이 `docker run --rm`으로 돌린다. 테스트 데이터는 `tests/test_train.py`의 `workspace` fixture(소형 무작위 베이스·데이터·q 캐시)를 쓴다
- 단계마다 `mydocs/working/task_m100_12_stage{N}.md`를 쓰고, 승인 뒤 그 단계 소스와 함께 `local/task12`에 커밋한다

## 모델과 표기

| 모델 ID | 정체 | 쓰는 곳 |
|---|---|---|
| `base` | 베이스 136M, 캐시의 베이스 q | A1~A3·C3의 비교 대상 |
| `R1`, `R2`, `R8` | 탈·스미슬로프·카르포프 α = 0, seed 1 | 판정 대상 |
| `R1s0`, `R2s0`, `R8s0` | 같은 기사 seed 0 | B3 잡음 바닥, 나머지는 참고 |

- 기사 순서는 `src/players.py`의 `PLAYERS` 순서(탈, 카르포프, 스미슬로프)로 고정한다. 표도 이 순서로 쓴다
- 기사별 어댑터: 탈 `R1`, 카르포프 `R8`, 스미슬로프 `R2`

---

## 1단계: 덤프 도구

### 목표

모델 하나를 세 기사 val 전체(S_val)에서 평가해, 국면마다 합법수 전체의 log π를 파일로 남긴다. 모델과 관계없는 국면 정보도 한 번 남긴다.

### 구현

**`src/eval/__init__.py`** — 빈 패키지

**`src/eval/dump.py`** (torch 사용) — 덤프와 학습 기록 대조

```python
def dump_positions(base: Path, out: Path, data_root: Path | None = None) -> dict
    # → {out}/positions.npz, {out}/base.npz
def dump_model(run_dir: Path, out: Path, device: str, num_workers: int = 2, ...) -> dict
    # → {out}/{모델 ID}.npz, {out}/{모델 ID}.json (기록 대조 결과, 환경, SHA-256, 시간)
# python -m src.eval.dump positions --base base_model/136M.pt --out runs/gpu/dump
# python -m src.eval.dump model runs/gpu/R1 --out runs/gpu/dump
```

`positions.npz` (모델과 무관, 세 기사 val을 `PLAYERS` 순서로 이어 붙임. 국면 13,846개, 합법수 M개)

| 배열 | 모양 | 내용 |
|---|---|---|
| `player` | [N] int8 | 기사 번호 (0 탈, 1 카르포프, 2 스미슬로프) |
| `game` | [N] int32 | 기사 안의 `game_id`. 대국 키는 (player, game) |
| `ply` | [N] int16 | 전처리 기록의 ply |
| `offsets` | [N + 1] int64 | 국면 i의 합법수는 [offsets[i], offsets[i+1]). 순서는 캐시와 같다(행동 번호 순) |
| `target` | [N] int16 | 기사가 둔 수가 합법수 중 몇 번째인지 |
| `regret` | [M] float32 | 베이스 기준 regret = max q_base − q_base (학습 `evaluate`와 같은 식) |
| `new` | [N] bool | 새 국면(2.4절): ply ≥ 20이고, FEN 앞 4필드가 세 기사 어느 train에도 없음 |

`base.npz`, `{모델 ID}.npz`

| 배열 | 모양 | 내용 |
|---|---|---|
| `log_pi` | [M] float32 | log π(a\|s), temperature는 학습 `config.json` 값(0.02). 베이스는 캐시 q로 CPU에서 계산한다 |

- 기사 val마다 학습과 같은 로더로 순회한다: `batch_positions`(32), 섞지 않음, `eval_max_moves`(256)로 `split_by_moves`, float32(`eval_amp` false). 학습 중 평가와 같은 묶음이라 자기 기사 val에서는 기록이 그대로 재현되어야 한다
- 모델 출력 `q`에 `policy_from_q`를 그대로 적용하고 mask 위치만 펴서 저장한다
- 설정값(temperature, batch, eval_max_moves, amp)은 학습 폴더 `config.json`에서 읽는다. 6개 모델의 값이 다르면 오류로 멈춘다
- **학습 기록 대조** (`{모델 ID}.json`): 덤프에서 자기 기사 val의 7개 지표(imitation, top1, top3, regret, regret_top1, band_violation, strength)를 다시 계산해 `log.jsonl` 마지막 eval과 비교한다. 규칙은 `gpu_train/eval_adapter.py`와 같다(top1·top3 정확히, 나머지 |차이| < 5 × 10⁻⁵). `base.npz`는 같은 방식으로 baseline 줄과 비교한다
- 이미 있는 덤프는 SHA-256과 대조가 통과했으면 건너뛴다(다시 제출해도 이어서 된다)

**`gpu_train/dump.sbatch`, `gpu_train/submit_dump.sh`** — `eval.sbatch`·`submit_eval.sh`와 같은 틀

- 순서: 입력 확인 → 기록(호스트, GPU, git HEAD, torch) → `positions` → 폴더마다 `model` → 대조 판정 표
- 하나가 실패해도 나머지를 덤프한다. 종료 코드 0 모두 통과, 1 실패·오류, 2 입력 확인 실패
- 노드 고정은 `site.env`의 `SBATCH_EXTRA=--nodelist=moana-u1`로 한다 (결정 2). 스크립트는 바꾸지 않는다
- 작업 이름 `dump`, 로그 `runs/gpu/logs/dump_{작업ID}.out`

**`tests/test_eval_dump.py`**

- `workspace` fixture로 소형 학습 2개를 돌린 뒤 덤프한다
- 덤프에서 다시 계산한 7개 지표가 `evaluate`의 값과 같다 (모델, 베이스)
- `positions.npz`의 배열 길이·offsets·target이 캐시와 맞는다. `new`가 손으로 만든 train FEN 예제와 맞는다
- 다른 어댑터를 넣으면 기록 대조가 실패한다. 설정이 다른 모델은 거부한다
- `dump.sbatch`를 Slurm 밖에서 돌려 판정 표와 종료 코드를 확인한다 (`test_eval_adapter.py`의 방식)

### 검증

- 새 테스트 통과, 전체 pytest 회귀 없음 (컨테이너, CPU)
- `git diff --stat -- src/`에 새 파일만 있다

---

## 2단계: 지표, 부트스트랩, 판정

### 목표

덤프만 읽어 지표·구간·판정을 내는 numpy 코드와 테스트를 만든다.

### 구현

**`src/eval/stats.py`** — 대국 단위 부트스트랩

```python
def resample_weights(groups: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> np.ndarray
    # [n_boot, G] 대국별 뽑힌 횟수
def ratio_ci(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> Interval
    # 대국별 합 num[G], den[G] → 값(원표본), 구간(2.5, 97.5 백분위)
```

- 국면 평균 지표는 "뽑힌 대국들의 국면 합 / 국면 수"(비율 추정)로 계산한다. 국면 수가 다른 대국을 국면 수만큼 가중한다. 원표본 값은 전체 국면 평균과 같다
- 쌍대: 같은 모집단에서 비교하는 값은 **같은 뽑기 행렬**을 쓴다. 차이는 반복마다 계산한 뒤 백분위를 낸다 (2.2절)
- 모집단과 뽑기
  - 기사 val (A1~A3, C3, B1, B10): 그 기사의 대국 117~129개
  - S_val (B2·B3): **기사별 층화**. 반복마다 기사마다 그 기사 대국 수만큼 따로 뽑아 합친다. 세 기사의 비율이 반복마다 같게 유지된다 (해석 1)
  - 뽑기 행렬은 모집단마다 `default_rng(0)`으로 새로 만든다. 계산 순서와 관계없이 같은 결과가 나온다
- 백분위는 `np.percentile` 기본(선형 보간)이다

**`src/eval/metrics.py`** — 지표 (기준 문서의 절과 식을 docstring에 적는다)

| 함수 | 계산 |
|---|---|
| `per_position(log_pi, offsets, target, regret)` | 국면별 top1 적중, top3 적중, −log π(a_p), 1순위 수 â(argmax, 동률은 앞 번호), regret_top1, regret, band_violation, strength |
| `a1_a2(...)` | Δtop1, Δimitation(어댑터 − 베이스). 전체, ply ≥ 20, 새 국면(A2). top3 차이는 보고 |
| `a3(...)` | D = {â_어댑터 ≠ â_베이스}. r = (n[어댑터 적중] − n[베이스 적중]) / \|D\|를 대국별 합으로 부트스트랩. \|D\|/N, D와 전체의 ply 구간 분포(0~9, 10~19, 20~39, 40 이상), McNemar p(정확 이항, 참고) |
| `js(log_p, log_q, offsets)` | 국면별 JS 발산, 밑 2, float64. 0·log 0 = 0 |
| `b2_b3(...)` | 15쌍(6개 모델)의 d와 δ. n_p = d(Rs0, R). M(p, q) = d(p, q) − max(n_p, n_q)를 반복마다 계산 |
| `b10(...)` | 대국 점수 L_m(g) = Σ_{ply ≥ 20} log π_m(a_p). 3지선다·쌍별 2지선다 재현율과 구간, 행 정규화 3×3, 열 합계. 전체 구간(ply 제한 없음) 보고. k = 2, 5, 10 |
| `b1(...)` | 3×3 교차 평가(imitation, top1). 기사 특이 몫 = 1 − (다른 두 어댑터 개선 평균) / (자기 어댑터 개선). 전체, ply ≥ 20 (Task #6 방식) |
| `b11(...)` | 세 어댑터의 1순위가 모두 같지는 않은 국면에서, 기사 p val의 어댑터 m 적중률 3×3 |
| `c3(...)` | regret_top1 평균과 구간. regret, band_violation, strength 보고 |

**`src/eval/judge.py`** — 판정과 보고

```python
# python -m src.eval.judge runs/gpu/dump --out runs/gpu/judge
#   → results.json (모든 값·구간·판정), report.md (표)
```

| 판정 | 규칙 (기준 문서) |
|---|---|
| 지표 | 통과 조건 / 실패 조건 / 그 밖 판정 보류 (2.5절, 각 지표 표) |
| A1 | top1·imitation 둘 다 통과해야 통과 (3.3절) |
| 특성 없는 A | A1~A3가 세 기사 모두 통과면 "부분 통과 (A5·A6 판정 보류)". 실패 지표가 있으면 실패 (3.5절) |
| B3 | 세 쌍 모두 M 구간 하한 > 0 통과, 어느 쌍이든 상한 < 0 실패 (4.3절) |
| B10 | 3지선다: 세 기사 재현율 구간 하한 > 1/3. 쌍별: 세 쌍 모두 두 기사 재현율 하한 > 1/2 (4.3절) |
| 특성 없는 층 | B3 통과 + B10 3지선다·쌍별 통과면 통과. B3 또는 B10 실패면 실패. 그 밖 보류 (4.7절) |
| C3 | 구간 상한 < 0.02 통과, 하한 > 0.02 실패. "단독 판정 불가"로 표시한다 (5.4절) |

- 기사별 결과를 항상 표로 낸다 (2.5절). seed 0 모델의 A·B10·C3 값은 "참고" 열로 붙이고 판정하지 않는다
- 덤프의 `{모델 ID}.json` 대조가 통과하지 않은 모델이 있으면 판정을 내지 않고 멈춘다

**해석 (결과를 보기 전에 정한다. 승인 대상)**

1. **S_val 부트스트랩은 기사별 층화로 한다.** 기준 문서는 "S_val의 대국 단위 부트스트랩"이라고만 적었다. 층화하면 반복마다 기사 구성이 같아 n_p와 d(p, q)의 비교가 기사 비율 변동에 흔들리지 않는다
2. **B10 동률**: L_m(g)가 같은 어댑터가 둘 이상이면 오답으로 센다(보수적). float64 합이라 실제로 생길 가능성은 낮고, 생기면 그 수를 보고한다
3. **B10 묶음 k > 1**: 기사 p의 대국을 무작위로 k개씩 나누고 남는 대국은 버린다. 묶음 점수는 대국 점수의 합이다. 1,000번 나눠 재현율을 평균한다(seed 0). 구간은 내지 않는다(보고용)
4. **1순위 동률**: â는 log π가 가장 큰 합법수 중 행동 번호가 가장 작은 것이다. 학습 `evaluate`(topk)와 동률 처리가 다를 수 있어, 덤프 대조에서 top1이 다르면 동률 국면 수를 함께 기록한다
5. **ply**: 전처리 기록의 `ply`(대국 시작부터 0으로 센 반수)를 쓴다. "ply 20 이상"은 `ply >= 20`이다
6. **A3 McNemar**: D 중 어댑터만 적중(b)과 베이스만 적중(c)으로 정확 이항 검정(양측)을 한다. 참고값이다
7. **기사 특이 몫(B1)**: 개선은 베이스 대비 imitation 감소다. 구간은 내지 않는다 (Task #6과 같다)

**`tests/test_eval_metrics.py`**

- 손으로 계산한 작은 예제(국면 3~6개, 대국 2~3개)로 각 지표의 원표본 값을 확인한다
- JS: 같은 분포 0, 서로 겹치지 않는 분포 1, 대칭
- 부트스트랩: 대국이 하나뿐이면 구간 폭 0, seed가 같으면 같은 구간, 비율 추정의 원표본 값 = 국면 평균
- B10: 대각 행렬이 되는 예제에서 재현율 1, 한 어댑터로 몰리는 예제에서 열 합계와 3지선다 실패
- 판정: 통과·실패·보류 경계(구간이 0을 포함, 1/3에 걸침)
- `judge`를 1단계 테스트의 소형 덤프에 돌려 `results.json`과 `report.md`가 나온다

### 검증

- 새 테스트 통과, 전체 pytest 회귀 없음
- 분석 모듈이 torch 없이 불러와진다 (테스트에서 `sys.modules` 확인)

---

## 3단계: 배포와 서버 실행 안내

### 목표

서버가 받을 수 있게 `devel`에 반영하고, 실행 방법을 문서로 남긴다.

### 구현

1. **`gpu_train/README.md`**: 덤프 절 추가 (노드 고정, 제출, 확인, 회수)
2. `local/task12` → `local/devel` → `devel` merge, `origin/devel` push (Task #11 3단계와 같은 선반영)
3. 단계 보고서에 서버 실행 명령을 적는다

```bash
git pull                                              # devel
# site.env: SBATCH_EXTRA=--nodelist=moana-u1 (A2000 고정, 결정 2)
SBATCH_DRY_RUN=1 bash gpu_train/submit_dump.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8 runs/gpu/R1s0 runs/gpu/R2s0 runs/gpu/R8s0
bash gpu_train/submit_dump.sh runs/gpu/R1 runs/gpu/R2 runs/gpu/R8 runs/gpu/R1s0 runs/gpu/R2s0 runs/gpu/R8s0
# 끝나면 (약 4.2시간)
tar -czf runs/gpu/collect/dump_$(date +%Y%m%d_%H%M%S).tar.gz runs/gpu/dump runs/gpu/logs/dump_*.out
```

- 서버에 R1s0·R2s0·R8s0 폴더가 있어야 한다 (학습한 서버이므로 있다)
- 시간 제한 `SBATCH_TIME`은 5시간 이상이 필요하다. 짧으면 같은 명령으로 다시 제출한다(끝난 덤프는 건너뛴다)

### 검증

- `SBATCH_DRY_RUN=1`로 명령이 나오는지 로컬에서 본다(site.env 예시 사용)
- `origin/devel`이 이 단계 커밋을 가리킨다

---

## 4단계: 덤프 회수, 대조, 판정

### 목표

서버 덤프를 받아 학습 기록과 대조하고, 지표와 판정을 낸다.

### 구현

1. 받은 묶음을 `runs/gpu/collect/`에 두고 `runs/gpu/dump/`에 배치한다
2. 확인
   - 7개 덤프의 대조 결과(`{모델 ID}.json`, `base.json`)가 모두 통과
   - 덤프 노드가 모두 같은 GPU 기종 (`dump_*.out`의 기록)
   - `positions.npz`의 국면 수 13,846(3,880 + 5,213 + 4,753), 대국 수 368(117 + 122 + 129)
3. `python -m src.eval.judge runs/gpu/dump --out runs/gpu/judge`를 실행한다(호스트 Python 또는 컨테이너)

### 검증

- 대조 통과. 실패하면 판정을 내지 않고 차이를 기록해 보고한다
- `results.json`의 원표본 값 일부를 독립 스크래치패드 스크립트로 다시 계산해 같다 (A1 Δtop1, B3 n_p, B10 재현율, C3)

---

## 5단계: 결과 정리

### 구현

1. **`mydocs/test_result/task_m100_12_test.md`**
   - 덤프 실행 기록(작업, 노드, GPU, 시간, 대조 결과)
   - 지표별 값·구간·판정 표 (기사별, 전체와 ply ≥ 20)
   - 판정 요약: 특성 없는 A, 특성 없는 층 B, C3
   - Task #6 기준선(기존 어댑터)과 나란히 둘 수 있는 값(B1, A1)은 비교 표로 둔다
2. 기준 문서는 바꾸지 않는다 (결정 4)

### 검증

- test 문서의 수치가 `results.json`과 같다 (스크립트 대조)

---

## 단계 요약

| 단계 | 산출물 | 커밋 |
|---|---|---|
| 1 | `src/eval/dump.py`, `gpu_train/dump.sbatch`, `submit_dump.sh`, `tests/test_eval_dump.py` | 소스 + 보고서 |
| 2 | `src/eval/stats.py`, `metrics.py`, `judge.py`, `tests/test_eval_metrics.py` | 소스 + 보고서 |
| 3 | `gpu_train/README.md` 덤프 절, `devel` 반영 | 문서 + 보고서 |
| 4 | 덤프 회수·대조·판정 (`runs/` git 제외) | 보고서 |
| 5 | `task_m100_12_test.md` | 문서 + 보고서 |

## 예상 단계: 5단계
