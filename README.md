# 매크로 지표 상관관계 분석 앱

소비자물가지수(CPI), 주식 인덱스, 고용률, 금리, 달러인덱스, 가상화폐 공포탐욕지수, 유가, 통화량, 금값, 국채가격, 주요 경기지표를 웹에서 수집해 다음을 제공합니다.

- 기간 선택 기반 시계열 그래프
- 지표 간 상관관계 히트맵
- 전월/전년/YTD 비교표
- 주택/주식/가상화폐/금/저축 자산군의 다음달 기대수익률(단순 회귀)

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 소스

- FRED (St. Louis Fed)
- Yahoo Finance
- alternative.me Fear & Greed Index API

> 투자 예측은 연구/학습용 참고 정보입니다.
