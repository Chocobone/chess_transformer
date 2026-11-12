"""평가 기준 문서(`mydocs/tech/evaluation_criteria.md`)의 지표를 계산한다.

- `dump`: 모델 출력(합법수별 log π)을 파일로 남긴다 (torch, GPU)
- 나머지 모듈: 덤프만 읽어 지표·구간·판정을 낸다 (numpy만 쓴다)
"""
