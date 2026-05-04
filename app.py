import datetime as dt
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="매크로 상관관계 분석기", layout="wide")


@dataclass
class SeriesConfig:
    name: str
    source: str
    key: str
    transform: str = "level"


def fetch_fred(series_id: str, start: dt.date, end: dt.date) -> pd.Series:
    s = web.DataReader(series_id, "fred", start, end)[series_id]
    s.name = series_id
    return s


def fetch_yf(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    df = yf.download(ticker, start=start, end=end + dt.timedelta(days=1), progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"]
    s.name = ticker
    return s


def fetch_fear_greed(start: dt.date, end: dt.date) -> pd.Series:
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    records = []
    for row in data:
        d = dt.datetime.utcfromtimestamp(int(row["timestamp"]))
        if start <= d.date() <= end:
            records.append((d, float(row["value"])))
    if not records:
        return pd.Series(dtype=float, name="Crypto Fear & Greed")
    s = pd.Series(dict(records)).sort_index()
    s.name = "Crypto Fear & Greed"
    return s


def transform_series(s: pd.Series, mode: str) -> pd.Series:
    if mode == "pct_change":
        return s.pct_change() * 100
    if mode == "yoy":
        return s.pct_change(12) * 100
    return s


def monthly_resample(s: pd.Series) -> pd.Series:
    return s.resample("M").last()


def prepare_data(configs: List[SeriesConfig], start: dt.date, end: dt.date) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    for c in configs:
        try:
            if c.source == "fred":
                raw = fetch_fred(c.key, start, end)
            elif c.source == "yfinance":
                raw = fetch_yf(c.key, start, end)
            elif c.source == "feargreed":
                raw = fetch_fear_greed(start, end)
            else:
                continue
            out[c.name] = transform_series(monthly_resample(raw), c.transform)
        except Exception as e:
            st.warning(f"{c.name} 데이터 조회 실패: {e}")

    if not out:
        return pd.DataFrame()
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()


def forecast_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    target_map = {
        "주택": "ITB",
        "주식": "SPY",
        "가상화폐": "BTC-USD",
        "금": "GLD",
        "저축": "SHY",
    }
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 10)
    macro = df.copy().dropna(how="all").ffill().dropna()
    if len(macro) < 24:
        return pd.DataFrame()

    rows = []
    X = macro.values
    for asset, ticker in target_map.items():
        y = monthly_resample(fetch_yf(ticker, start, end)).pct_change().shift(-1) * 100
        y = y.reindex(macro.index)
        valid = ~y.isna() & np.isfinite(X).all(axis=1)
        if valid.sum() < 24:
            continue
        model = LinearRegression().fit(X[valid], y[valid])
        pred = float(model.predict([X[-1]])[0])
        rows.append({"투자항목": asset, "예상수익률(다음달, %)": pred, "설명력(R²)": float(model.score(X[valid], y[valid]))})
    return pd.DataFrame(rows).sort_values("예상수익률(다음달, %)", ascending=False)



def build_mock_data(configs: List[SeriesConfig], start: dt.date, end: dt.date) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="M")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    for i, c in enumerate(configs):
        base = np.sin(np.linspace(0, 8, len(idx)) + i * 0.2) * 2
        noise = rng.normal(0, 0.8, len(idx))
        trend = np.linspace(-1, 1, len(idx)) * (0.2 + i * 0.03)
        df[c.name] = base + noise + trend
    return df

def diagnose_korean(row: pd.Series) -> str:
    r = row["예상수익률(다음달, %)"]
    fit = row["설명력(R²)"]
    if r >= 2:
        tone = "상대적으로 긍정"
    elif r >= 0:
        tone = "중립~완만한 긍정"
    else:
        tone = "보수적 접근 필요"
    return f"{row['투자항목']}: 예상수익률 {r:.2f}% / 모델설명력 {fit:.2f} → {tone}"


st.title("거시경제 지표 상관관계/추이 분석 및 투자 예측 앱")
st.caption("모든 진단 문구는 한글로 표시됩니다. 필요 시 샘플데이터 모드로 즉시 데모 실행이 가능합니다.")

all_configs = [
    SeriesConfig("소비자물가지수(CPI)", "fred", "CPIAUCSL", "yoy"),
    SeriesConfig("주식인덱스(S&P500)", "yfinance", "^GSPC", "pct_change"),
    SeriesConfig("고용률(고용/인구비율)", "fred", "EMRATIO", "level"),
    SeriesConfig("정책금리(Fed Funds)", "fred", "FEDFUNDS", "level"),
    SeriesConfig("달러인덱스(DXY)", "yfinance", "DX-Y.NYB", "pct_change"),
    SeriesConfig("가상화폐 공포탐욕지수", "feargreed", "FNG", "level"),
    SeriesConfig("유가(WTI)", "fred", "DCOILWTICO", "pct_change"),
    SeriesConfig("통화량(M2)", "fred", "M2SL", "yoy"),
    SeriesConfig("금값(Gold)", "fred", "GOLDAMGBD228NLBM", "pct_change"),
    SeriesConfig("국채가격(20Y ETF)", "yfinance", "TLT", "pct_change"),
    SeriesConfig("실업률", "fred", "UNRATE", "level"),
    SeriesConfig("ISM 제조업 PMI", "fred", "NAPM", "level"),
]

with st.sidebar:
    st.header("Market Indicators")
    end_date = st.date_input("종료일", value=dt.date.today())
    start_date = st.date_input("시작일", value=end_date - dt.timedelta(days=365 * 8))
    st.write("표시할 지표 선택")
    selected_names = [c.name for c in all_configs if st.checkbox(c.name, value=True)]
    use_mock = st.toggle("샘플데이터 모드(오프라인 데모)", value=False)
    run = st.button("RunAnalysis", type="primary")

if start_date >= end_date:
    st.error("시작일은 종료일보다 과거여야 합니다.")
    st.stop()

selected_configs = [c for c in all_configs if c.name in selected_names]
if not selected_configs:
    st.warning("최소 1개 이상의 지표를 선택해 주세요.")
    st.stop()

if run:
    with st.spinner("데이터 수집/분석 중..."):
        data = build_mock_data(selected_configs, start_date, end_date) if use_mock else prepare_data(selected_configs, start_date, end_date)

    if data.empty:
        st.error("데이터를 가져오지 못했습니다. 샘플데이터 모드를 켜고 다시 시도해 보세요.")
        st.stop()

    st.subheader("지표 추이")
    fig = px.line(data, x=data.index, y=data.columns)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("지표 상관관계")
    corr = data.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1), use_container_width=True)

    st.subheader("전월/전년/YTD 요약")
    summary = pd.DataFrame(index=data.columns)
    summary["전월(%)"] = (data.iloc[-1] / data.iloc[-2] - 1) * 100
    summary["전년(%)"] = (data.iloc[-1] / data.iloc[-13] - 1) * 100 if len(data) > 13 else np.nan
    this_year = data[data.index.year == data.index[-1].year]
    last_year = data[data.index.year == data.index[-1].year - 1]
    if len(this_year) > 0 and len(last_year) > 0:
        summary["YTD(%)"] = (this_year.iloc[-1] / last_year.iloc[-1] - 1) * 100
    st.dataframe(summary.round(2), use_container_width=True)

    st.subheader("RunAnalysis 결과: 투자항목별 간단 예측")
    pred = forecast_asset_returns(data)
    if pred.empty:
        st.info("예측 가능한 데이터가 부족합니다.")
    else:
        st.dataframe(pred.round(3), use_container_width=True)
        st.markdown("#### 간단 진단 (한글)")
        for _, row in pred.iterrows():
            st.write("- " + diagnose_korean(row))
else:
    st.info("왼쪽에서 지표를 선택하고 RunAnalysis를 눌러 분석을 실행하세요.")

st.caption("주의: 본 결과는 투자 참고용 통계 분석이며, 투자 자문이 아닙니다.")
