# Task #11 단계1 완료 보고서: 배치와 무결성 확인

## 수행 내용

1. **묶음 원본 이동**: 저장소 루트의 `R1_20261109_135827.tar.gz`, `R2_20261109_135827.tar.gz`, `R8_20261109_135829.tar.gz`, `smoke_20261109_115519.tar.gz`를 `runs/gpu/collect/`로 옮겼다. 같은 이름의 파일은 없었다.
2. **배치**: 풀어 둔 `runs/gpu/collect/{R1,R2,R8}/runs/gpu/` 아래 내용을 서버와 같은 경로인 `runs/gpu/`로 복사했다. 이름이 겹치는 파일은 없어서 덮어쓴 것도 없다.

   ```
   runs/gpu/
   ├── R1/  R2/  R8/          adapter.pt, config.json, DONE, log.jsonl
   ├── logs/                  gpu_{R}.log, {R}_{작업ID}.out
   ├── meta/                  {R}_{작업ID}.txt
   └── collect/               묶음 원본 4개와 풀어 둔 폴더 (R1, R2, R8, smoke)
   ```

3. **확인**: 스크래치패드 스크립트(커밋하지 않음)로 아래 항목을 확인했다.
   - 각 항목은 `SUMMARY.txt`, `config.json`, `log.jsonl`, `adapter.pt`, seed 0 폴더 `runs/{tal,smyslov,karpov}_136M/`을 직접 읽어 대조했다.
   - 호스트에 torch가 없다. 그래서 `adapter.pt` 안의 `settings`는 `data.pkl`만 풀어서 읽었다. 텐서는 불러오지 않았다.

## 확인 결과

| 항목 | 기준 | R1 (탈) | R2 (스미슬로프) | R8 (카르포프) |
|---|---|---|---|---|
| `adapter.pt` 크기 | `SUMMARY.txt`와 같다 | 4,213,206 B ✅ | 4,213,206 B ✅ | 4,213,206 B ✅ |
| SHA-256 | `SUMMARY.txt` = 배치본 = 묶음 안의 원본 | ✅ `cc46b856…0e2c565` | ✅ | ✅ |
| `DONE` | 있다 | 11-06 00:05:53, 작업 153217 ✅ | 11-06 09:25:38, 작업 153218 ✅ | 11-06 19:29:52, 작업 153219 ✅ |
| 마지막 eval step | `total_steps`와 같다 | 7,455 = 7,455 ✅ | 8,736 = 8,736 ✅ | 9,039 = 9,039 ✅ |
| eval 줄 수 | 1,000 step마다 + 마지막 | 8 | 9 | 10 |
| 학습·val 국면 수 | seed 0과 같다 | 79,493 / 3,880 ✅ | 93,178 / 4,753 ✅ | 96,393 / 5,213 ✅ |
| seed 0과 다른 `config.json` 키 | `seed`, `out`뿐이다 | `seed` 0→1, `out` ✅ | 같음 ✅ | 같음 ✅ |
| 어댑터 `settings` | 8.1절: rank 16, alpha 32, dropout 0, q·k·v·o | ✅ | ✅ | ✅ |
| α, γ | 0, 0 | ✅ | ✅ | ✅ |
| baseline top1·top3 | seed 0과 정확히 같다 | ✅ | ✅ | ✅ |
| baseline 소수 값 최대 차이 | (참고) | 7.3 × 10⁻¹⁰ | 7.1 × 10⁻⁹ | 1.5 × 10⁻⁹ |

- `adapter.pt`의 최상위 키는 `settings`, `state_dict` 두 개다.
- `config.json`의 공통 설정은 8.1절과 같다. 베이스 136M, 학습률 1e-4, 3 epoch, 배치 32국면, temperature 0.02, τ = 0.02, λ = 1, γ = 0, α = 0, seed 1이다.
- 세 학습의 SHA-256 전체 값은 3단계 test 문서에 적는다.

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `mydocs/working/task_m100_11_stage1.md` | 이 보고서 |
| `mydocs/orders/20261110.md` | 1단계 진행 기록 |

- 소스 변경은 없다.
- 배치한 파일은 `runs/` 아래에 있어 git에서 제외된다. `git status`에는 `runs/` 아래 파일이 나오지 않는다. 나오는 것은 #11 범위 밖의 `src/play/README.md`뿐이다.

## 발견 사항

- seed 0 폴더 `runs/{player}_136M/`은 읽기만 했다.
- 이번 확인은 수행 계획서 "이미 확인한 것"의 내용과 모두 같다.

## 검증

- 세 학습 모두 위 확인 항목 통과: ✅
- `git status`에 `runs/` 아래 파일이 나오지 않음 (git 제외): ✅
- `git diff --stat -- src/` 비어 있음: ✅

## 다음 단계

2단계에서 `gpu_train/eval_adapter.py`와 테스트를 작성하고, 로컬 GPU에서 세 어댑터를 재평가한다.

- **작업지시자께 요청**: Docker Desktop을 켜 주시면 됩니다. 그러면 컨테이너를 띄우고 `fetch_inputs.sh`로 입력을 받는다. Hugging Face 토큰이 컨테이너 안에 없으면 `hf auth login`이 필요하다.

## 커밋

(승인 후 기록)
