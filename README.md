# Hyper-Waterfall 프로젝트 템플릿

> rhwp 프로젝트에서 검증된 Hyper-Waterfall 방법론을 새 프로젝트에 바로 적용하기 위한 템플릿이다.

## 이 템플릿이 포함하는 것

```
hyper-waterfall/
├── CLAUDE.md                               ← Claude Code 지시서 (수정 후 사용)
├── README.md                               ← 이 파일
└── mydocs/
    ├── manual/
    │   └── hyper_waterfall.md              ← 방법론 설명서
    ├── orders/
    │   └── YYYYMMDD_template.md            ← 오늘 할일 템플릿
    ├── plans/
    │   ├── task_MILESTONE_NUM_template.md       ← 수행 계획서 템플릿
    │   └── task_MILESTONE_NUM_impl_template.md  ← 구현 계획서 템플릿
    ├── working/
    │   └── task_MILESTONE_NUM_stageN_template.md ← 단계별 완료보고서 템플릿
    ├── report/
    │   └── task_MILESTONE_NUM_report_template.md ← 최종 결과보고서 템플릿
    ├── feedback/
    │   └── feedback_template.md            ← 피드백 문서 템플릿
    ├── tech/
    │   └── tech_investigation_template.md  ← 기술 조사 템플릿
    └── troubleshootings/
        └── troubleshooting_template.md     ← 트러블슈팅 템플릿
```

---

## 새 프로젝트에 적용하는 방법

### 1단계: 파일 복사

`hyper-waterfall/` 안의 파일들을 새 프로젝트 루트에 복사한다.

```bash
# 예시: my-project에 적용
cp -r hyper-waterfall/CLAUDE.md my-project/
cp -r hyper-waterfall/mydocs my-project/
```

### 2단계: CLAUDE.md 커스터마이징

`CLAUDE.md`를 열고 `TODO` 항목을 채운다:

- **프로젝트 개요**: 목표와 기술 스택 기술
- **빌드 및 실행**: 프로젝트별 빌드/테스트/실행 명령 기술

```markdown
## 프로젝트 개요

**목표**: 내 프로젝트 목표
**기술 스택**: TypeScript, Node.js, PostgreSQL
```

### 3단계: GitHub Issues 설정

하이퍼-워터폴은 GitHub Issues를 타스크 번호로 사용한다.

```bash
# 첫 번째 타스크 등록 예시
gh issue create --title "기능명: 설명" --body "상세 내용"
```

마일스톤도 설정한다:

```bash
gh api repos/{owner}/{repo}/milestones \
  --method POST \
  --field title="v1.0.0" \
  --field description="첫 번째 릴리즈"
```

### 4단계: 브랜치 구조 설정

```bash
git checkout -b devel
git checkout -b local/devel
```

---

## 첫 번째 타스크 시작 방법

### 1. 오늘 할일 작성

`mydocs/orders/YYYYMMDD_template.md`를 복사하여 오늘 날짜로 만든다:

```bash
cp mydocs/orders/YYYYMMDD_template.md mydocs/orders/20260427.md
```

파일을 열고 타스크 정보를 채운다.

### 2. 타스크 브랜치 생성

```bash
git checkout local/devel
git checkout -b local/task1   # 이슈 번호에 맞게
```

### 3. Claude Code에 수행 계획서 작성 요청

Claude Code에게 다음과 같이 요청한다:

```
GitHub 이슈 #1의 수행 계획서를 작성해줘.
이슈 내용: [이슈 내용]
마일스톤: m100
```

Claude Code가 `mydocs/plans/task_m100_1.md`를 생성한다.

### 4. 계획서 검토 후 승인

계획서를 읽고 방향이 맞으면 "승인" 또는 "다음 단계 진행해줘"라고 응답한다.
방향이 틀리면 구체적인 피드백을 준다.

### 5. 단계별 구현 → 보고서 → 승인 반복

각 단계가 완료될 때마다:
- Claude Code가 `mydocs/working/task_m100_1_stage1.md` 작성
- 검토 후 승인하면 다음 단계 진행

### 6. 최종 보고서 작성 및 이슈 클로즈

모든 단계 완료 후:
- Claude Code가 `mydocs/report/task_m100_1_report.md` 작성
- 검토 후 승인하면 이슈 클로즈

---

## 파일명 규칙 요약

| 문서 | 파일명 | 위치 |
|------|--------|------|
| 오늘 할일 | `20260427.md` | `mydocs/orders/` |
| 수행 계획서 | `task_m100_1.md` | `mydocs/plans/` |
| 구현 계획서 | `task_m100_1_impl.md` | `mydocs/plans/` |
| 단계별 보고서 | `task_m100_1_stage1.md` | `mydocs/working/` |
| 최종 보고서 | `task_m100_1_report.md` | `mydocs/report/` |
| 피드백 | `feedback_기능명.md` | `mydocs/feedback/` |
| 기술 조사 | `기술명.md` | `mydocs/tech/` |
| 트러블슈팅 | `문제명.md` | `mydocs/troubleshootings/` |

**마일스톤 표기**: `m100` = v1.0.0, `m200` = v2.0.0, `m050` = v0.5.0

---

## 핵심 원칙 (반드시 숙지)

1. **소스 수정 전 반드시 승인** — 아무리 명백한 버그라도 먼저 보고하고 승인받는다
2. **단계 건너뛰기 금지** — 이슈 → 브랜치 → 할일 → 계획서 → 구현 순서 준수
3. **AI가 작업 종료를 결정하지 않는다** — 작업지시자가 결정한다
4. **문서는 타스크 브랜치에서 커밋** — merge 전 `git status`로 확인
5. **최종 보고서는 `report/` 폴더** — `working/`에 두지 않는다

---

## 방법론 상세

`mydocs/manual/hyper_waterfall.md` 참조.

rhwp 프로젝트 실전 적용 사례: https://github.com/edwardkim/rhwp
