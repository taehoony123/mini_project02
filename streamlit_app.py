from pathlib import Path
import html
from dataclasses import dataclass
from textwrap import dedent

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


st.set_page_config(
    page_title="Daegu Apartment Market Dashboard",
    page_icon="APT",
    layout="wide",
)

DEFAULT_DATA_PATHS = [
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_20230101~20231231.xlsx"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_20240101~20251231.xlsx"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_20250101~20251231.xlsx"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_20260101~20260616.xlsx"),
]


@st.cache_data(show_spinner=False)
def load_apartment_data(cache_version=0):
    sources = [p for p in DEFAULT_DATA_PATHS if p.exists()]
    if not sources:
        return pd.DataFrame()

    raw_parts = [pd.read_excel(source, skiprows=12) for source in sources]
    raw = pd.concat(raw_parts, ignore_index=True)
    raw = raw.dropna(how="all").copy()

    df = raw.rename(
        columns={
            "시군구": "address",
            "단지명": "complex_name",
            "전용면적(㎡)": "area_m2",
            "계약년월": "contract_ym",
            "계약일": "contract_day",
            "거래금액(만원)": "price_10k_krw",
            "층": "floor",
            "건축년도": "built_year",
            "거래유형": "deal_type",
            "도로명": "road_name",
        }
    )

    required = [
        "address",
        "complex_name",
        "area_m2",
        "contract_ym",
        "contract_day",
        "price_10k_krw",
        "floor",
        "built_year",
        "deal_type",
    ]
    df = df[[c for c in required if c in df.columns]].copy()

    df["price_10k_krw"] = (
        df["price_10k_krw"].astype(str).str.replace(",", "", regex=False)
    )
    numeric_cols = ["area_m2", "contract_ym", "contract_day", "price_10k_krw", "floor", "built_year"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["address", "area_m2", "contract_ym", "price_10k_krw", "built_year"])
    df = df.drop_duplicates(
        subset=[
            "address",
            "complex_name",
            "area_m2",
            "contract_ym",
            "contract_day",
            "price_10k_krw",
            "floor",
            "built_year",
            "deal_type",
        ]
    )
    df["contract_ym"] = df["contract_ym"].astype(int)
    df["contract_day"] = df["contract_day"].fillna(1).astype(int).clip(1, 31)
    df["contract_date"] = pd.to_datetime(
        df["contract_ym"].astype(str) + df["contract_day"].astype(str).str.zfill(2),
        format="%Y%m%d",
        errors="coerce",
    )
    df["month"] = pd.to_datetime(df["contract_ym"].astype(str), format="%Y%m", errors="coerce")
    df["district"] = df["address"].astype(str).str.extract(r"대구광역시\s+(\S+)")
    df["neighborhood"] = df["address"].astype(str).str.extract(r"대구광역시\s+\S+\s+(\S+)")
    df["price_per_m2"] = df["price_10k_krw"] / df["area_m2"]
    df["price_per_pyeong"] = df["price_10k_krw"] / (df["area_m2"] / 3.3058)
    df["area_pyeong"] = df["area_m2"] / 3.3058
    df["building_age"] = 2026 - df["built_year"]
    df["area_group"] = pd.cut(
        df["area_m2"],
        bins=[0, 60, 85, 135, float("inf")],
        labels=["소형(60㎡ 이하)", "중형(60-85㎡)", "대형(85-135㎡)", "초대형(135㎡ 초과)"],
        right=False,
    )
    df["age_group"] = pd.cut(
        df["building_age"],
        bins=[-1, 5, 10, 20, 30, float("inf")],
        labels=["5년 이하", "6-10년", "11-20년", "21-30년", "31년 이상"],
    )
    df["pyeong_group"] = pd.cut(
        df["area_pyeong"],
        bins=[0, 10, 20, 30, 40, 50, float("inf")],
        labels=["10평 미만", "10~20평", "20~30평", "30~40평", "40~50평", "50평 이상"],
        right=False,
    )
    return df.dropna(subset=["district", "month", "price_per_m2", "price_per_pyeong"])


def format_price_uk(value):
    if pd.isna(value):
        return "-"
    return f"{value / 10000:.2f}억"


PYEONG_DIVISOR = 3.305785
RECENT_MONTHS = 12
MODEL_CACHE_VERSION = 3
DATA_CACHE_VERSION = 2


@dataclass(frozen=True)
class TrainedPriceModel:
    name: str
    model: TransformedTargetRegressor
    features: list[str]
    mae: float
    mape: float
    r2: float
    train_rows: int
    test_rows: int
    test_period: str
    band_metrics: pd.DataFrame


@dataclass(frozen=True)
class PriceModelBundle:
    with_complex: TrainedPriceModel
    without_complex: TrainedPriceModel
    last_contract_month: pd.Period


def prediction_dataset(df):
    pred_df = df.rename(
        columns={
            "address": "시군구",
            "complex_name": "단지명",
            "area_m2": "전용면적(㎡)",
            "contract_ym": "계약년월",
            "contract_day": "계약일",
            "price_10k_krw": "거래금액(만원)",
            "floor": "층",
            "built_year": "건축년도",
            "district": "구",
            "neighborhood": "법정동",
            "contract_date": "계약일자",
            "area_pyeong": "평",
            "price_per_pyeong": "평당가(만원)",
        }
    ).copy()
    pred_df["계약년도"] = (pred_df["계약년월"] // 100).astype(int)
    pred_df["계약월"] = (pred_df["계약년월"] % 100).astype(int)
    pred_df["계약월순번"] = pred_df["계약년도"] * 12 + pred_df["계약월"]
    pred_df["건물연식"] = (pred_df["계약년도"] - pred_df["건축년도"]).clip(lower=0)
    return pred_df.dropna(
        subset=["시군구", "단지명", "전용면적(㎡)", "계약년월", "거래금액(만원)", "건축년도", "구", "법정동"]
    ).copy()


def make_price_model(features, categorical_features):
    numeric_features = [col for col in features if col not in categorical_features]
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_features),
            ("num", numeric_pipe, numeric_features),
        ],
        remainder="drop",
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=450,
        learning_rate=0.045,
        max_leaf_nodes=45,
        min_samples_leaf=18,
        l2_regularization=0.03,
        random_state=42,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("regressor", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def train_one_price_model(df, name, features, categorical_features):
    model_df = df[features + ["거래금액(만원)"]].dropna().copy()
    max_year = int(model_df["계약년도"].max())
    train_df = model_df[model_df["계약년도"] < max_year]
    test_df = model_df[model_df["계약년도"] == max_year]
    test_period = f"{max_year}년"

    if len(train_df) < 1000 or len(test_df) < 200:
        model_df = model_df.sort_values("계약월순번")
        split_at = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:split_at]
        test_df = model_df.iloc[split_at:]
        test_period = "최근 20%"

    eval_model = make_price_model(features, categorical_features)
    eval_model.fit(train_df[features], train_df["거래금액(만원)"])
    pred = eval_model.predict(test_df[features])
    actual = test_df["거래금액(만원)"]

    eval_result = test_df[["거래금액(만원)"]].copy()
    eval_result["예측금액(만원)"] = pred
    eval_result["절대오차(만원)"] = (eval_result["거래금액(만원)"] - eval_result["예측금액(만원)"]).abs()
    eval_result["오차율"] = eval_result["절대오차(만원)"] / eval_result["거래금액(만원)"]
    eval_result["가격대"] = pd.cut(
        eval_result["거래금액(만원)"],
        bins=[0, 10000, 30000, 60000, 100000, np.inf],
        labels=["1억 이하", "1-3억", "3-6억", "6-10억", "10억 초과"],
    )
    band_metrics = (
        eval_result.groupby("가격대", observed=False)
        .agg(건수=("거래금액(만원)", "size"), MAE_만원=("절대오차(만원)", "mean"), MAPE=("오차율", "mean"))
        .reset_index()
    )
    band_metrics = band_metrics[band_metrics["건수"] > 0].copy()

    final_model = make_price_model(features, categorical_features)
    final_model.fit(model_df[features], model_df["거래금액(만원)"])
    return TrainedPriceModel(
        name=name,
        model=final_model,
        features=features,
        mae=mean_absolute_error(actual, pred),
        mape=float(np.mean(np.abs((actual - pred) / actual))),
        r2=r2_score(actual, pred),
        train_rows=len(train_df),
        test_rows=len(test_df),
        test_period=test_period,
        band_metrics=band_metrics,
    )


@st.cache_resource(show_spinner="가격 예측 모델을 학습하는 중입니다...")
def train_price_models(df, cache_version):
    shared_features = [
        "시군구",
        "구",
        "법정동",
        "전용면적(㎡)",
        "층",
        "건축년도",
        "건물연식",
        "계약월순번",
        "계약월",
        "계약년도",
    ]
    with_complex_features = ["단지명"] + shared_features
    return PriceModelBundle(
        with_complex=train_one_price_model(
            df,
            "단지명 사용 모델",
            with_complex_features,
            ["단지명", "시군구", "구", "법정동"],
        ),
        without_complex=train_one_price_model(
            df,
            "단지명 미사용 모델",
            shared_features,
            ["시군구", "구", "법정동"],
        ),
        last_contract_month=pd.Period(str(int(df["계약년월"].max())), freq="M"),
    )


def recent_trade_weight(match_count):
    if match_count >= 30:
        return 0.3
    if match_count >= 15:
        return 0.2
    if match_count >= 5:
        return 0.1
    return 0.0


def blend_with_recent_median(model_price, matches):
    if matches.empty:
        return model_price, None, 0.0
    weight = recent_trade_weight(len(matches))
    if weight == 0:
        return model_price, None, weight
    recent_median = float(matches["거래금액(만원)"].median())
    adjusted_price = model_price * (1 - weight) + recent_median * weight
    return adjusted_price, recent_median, weight


def recent_matches(df, sigungu, complex_name, area, built_year, last_month):
    start_month = (last_month - RECENT_MONTHS + 1).strftime("%Y%m")
    filtered = df[df["계약년월"] >= int(start_month)].copy()
    if sigungu:
        filtered = filtered[filtered["시군구"] == sigungu]
    if complex_name != "전체/모름":
        filtered = filtered[filtered["단지명"] == complex_name]

    area_tolerance = max(3.0, area * 0.08)
    year_tolerance = 7
    narrowed = filtered[
        (filtered["전용면적(㎡)"].between(area - area_tolerance, area + area_tolerance))
        & (filtered["건축년도"].between(built_year - year_tolerance, built_year + year_tolerance))
    ]
    if len(narrowed) < 5:
        narrowed = filtered[filtered["전용면적(㎡)"].between(area - area_tolerance, area + area_tolerance)]
    if len(narrowed) < 5:
        narrowed = filtered

    columns = ["계약일자", "시군구", "단지명", "전용면적(㎡)", "평", "층", "건축년도", "거래금액(만원)", "평당가(만원)"]
    return narrowed[columns].sort_values("계약일자", ascending=False)


def build_prediction_row(sigungu, complex_name, area, floor, built_year, contract_year, contract_month):
    parts = str(sigungu).split()
    gu = parts[1] if len(parts) > 1 else ""
    dong = parts[2] if len(parts) > 2 else ""
    return pd.DataFrame(
        [
            {
                "시군구": sigungu,
                "구": gu,
                "법정동": dong,
                "단지명": complex_name,
                "전용면적(㎡)": area,
                "평": area / PYEONG_DIVISOR,
                "층": floor,
                "건축년도": built_year,
                "건물연식": max(0, contract_year - built_year),
                "계약월순번": contract_year * 12 + contract_month,
                "계약년도": contract_year,
                "계약월": contract_month,
            }
        ]
    )


def render_soft_table(df, max_rows=300):
    view = df.head(max_rows).copy()
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    note = ""
    if len(df) > max_rows:
        note = f'<div class="soft-table-note">상위 {max_rows:,}건만 표시 중 · 전체 {len(df):,}건은 다운로드로 확인</div>'
    return f"""
    <div class="soft-table-wrap">
      <table class="soft-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      {note}
    </div>
    """


def render_dong_rank_panel(df, selected_district, max_rows=10):
    rank_df = (
        df.groupby("neighborhood", as_index=False)
        .agg(
            avg_price=("price_10k_krw", "mean"),
            median_pyeong=("price_per_pyeong", "median"),
            deals=("price_10k_krw", "count"),
        )
        .query("deals >= 3")
        .sort_values("avg_price", ascending=False)
        .head(max_rows)
    )
    if rank_df.empty:
        return dedent("""
        <div class="dong-rank-card">
          <div class="dong-rank-header"><strong>동별 평균 매매가</strong><span>거래 3건 이상</span></div>
          <div class="analysis-note">선택한 조건에서 동별 순위를 만들 거래가 부족합니다.</div>
        </div>
        """).strip()

    max_price = rank_df["avg_price"].max()
    rows = []
    for idx, row in enumerate(rank_df.itertuples(index=False), start=1):
        width = 0 if not max_price else row.avg_price / max_price * 100
        rows.append(
            dedent(f"""
            <div class="dong-rank-row">
              <div class="dong-rank-no">{idx}</div>
              <div class="dong-rank-name">{html.escape(str(row.neighborhood))}</div>
              <div class="dong-rank-value">{row.avg_price:,.0f}</div>
              <div class="dong-rank-bar"><div class="dong-rank-fill" style="width:{width:.1f}%"></div></div>
            </div>
            """).strip()
        )
    return dedent(f"""
    <div class="dong-rank-card">
      <div class="dong-rank-header">
        <strong>{html.escape(selected_district)} 동별 평균 매매가</strong>
        <span>단위: 만원 · 거래 3건 이상</span>
      </div>
      <div class="dong-rank-list">
        {"".join(rows)}
      </div>
    </div>
    """).strip()


def render_prediction_result_card(
    adjusted_price,
    predicted_price,
    recent_median_price,
    recent_weight,
    adjusted_pyeong_price,
    area,
    floor,
    built_year,
    contract_year,
    contract_month,
    gu,
    dong,
    complex_name,
    active_model,
):
    model_weight = 1 - recent_weight
    model_width = max(model_weight * 100, 2 if model_weight > 0 else 0)
    recent_width = max(recent_weight * 100, 2 if recent_weight > 0 else 0)
    correction_gap = adjusted_price - predicted_price
    correction_rate = correction_gap / predicted_price * 100 if predicted_price else 0
    correction_class = "prediction-up" if correction_gap >= 0 else "prediction-down"
    correction_sign = "+" if correction_gap >= 0 else ""
    recent_label = format_price_uk(recent_median_price) if recent_median_price is not None else "반영 없음"
    complex_label = complex_name if complex_name != "전체/모름" else "단지명 미사용"
    basis_note = (
        f"모델 예측가 {model_weight:.0%}와 최근 1년 유사거래 중위가 {recent_weight:.0%}를 섞어 보정했습니다."
        if recent_median_price is not None
        else "최근 유사 거래가 5건 미만이라 모델 예측가를 그대로 사용했습니다."
    )
    location = f"{gu} {dong}".strip()

    return dedent(f"""
    <div class="prediction-result-card">
      <div class="prediction-result-head">
        <div>
          <span>최종 보정 예상가</span>
          <strong>{format_price_uk(adjusted_price)}</strong>
          <p>예상 평당가 {adjusted_pyeong_price:,.0f}만원/평</p>
        </div>
        <div class="prediction-badge">{html.escape(active_model.name)}</div>
      </div>
      <div class="prediction-context">
        <span>{html.escape(location)}</span>
        <span>{html.escape(complex_label)}</span>
        <span>{area:,.1f}㎡ · {area / PYEONG_DIVISOR:,.1f}평</span>
        <span>{built_year}년식 · {floor}층</span>
        <span>{contract_year}-{contract_month:02d} 기준</span>
      </div>
      <div class="prediction-mini-grid">
        <div><span>모델 예측가</span><b>{format_price_uk(predicted_price)}</b></div>
        <div><span>최근 유사거래 중위가</span><b>{recent_label}</b></div>
        <div><span>모델 대비 보정폭</span><b class="{correction_class}">{correction_sign}{correction_rate:.1f}%</b></div>
        <div><span>시간검증 오차</span><b>{active_model.mae:,.0f}만원</b><small>MAPE {active_model.mape:.1%}</small></div>
      </div>
      <div class="prediction-blend">
        <div class="prediction-blend-label">
          <span>가격 산출 비중</span>
          <b>모델 {model_weight:.0%} · 최근거래 {recent_weight:.0%}</b>
        </div>
        <div class="prediction-blend-bar">
          <div class="prediction-model-bar" style="width:{model_width:.1f}%"></div>
          <div class="prediction-recent-bar" style="width:{recent_width:.1f}%"></div>
        </div>
      </div>
      <p class="prediction-note">{basis_note}</p>
    </div>
    """).strip()


def soft_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(fill="#FFFFFF", strokeOpacity=0)
        .configure_axis(
            labelColor="#1F2937",
            titleColor="#1F2937",
            gridColor="#E2E8F0",
            domainColor="#CBD5E1",
            tickColor="#CBD5E1",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(
            labelColor="#1F2937",
            titleColor="#1F2937",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_title(color="#1F2937", fontSize=15)
    )


def chart_district_price(df):
    summary = (
        df.groupby("district", as_index=False)
        .agg(
            median_price=("price_10k_krw", "median"),
            median_pyeong=("price_per_pyeong", "median"),
            deals=("price_10k_krw", "count"),
        )
        .sort_values("median_pyeong", ascending=False)
    )
    chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=12, cornerRadiusTopRight=12, color="#2563EB")
        .encode(
            x=alt.X("district:N", sort="-y", title="구"),
            y=alt.Y("median_pyeong:Q", title="중위 평당가(만원/평)"),
            tooltip=[
                alt.Tooltip("district:N", title="구"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
            ],
        )
        .properties(height=360)
    )
    return soft_chart(chart)


def chart_district_volume(df):
    volume = (
        df.groupby("district", as_index=False)
        .agg(
            deals=("price_10k_krw", "count"),
            median_pyeong=("price_per_pyeong", "median"),
        )
        .sort_values("deals", ascending=False)
    )
    chart = (
        alt.Chart(volume)
        .mark_bar(cornerRadiusTopLeft=12, cornerRadiusTopRight=12, color="#8E88FF")
        .encode(
            x=alt.X("district:N", sort="-y", title="지역구"),
            y=alt.Y("deals:Q", title="거래건수"),
            tooltip=[
                alt.Tooltip("district:N", title="지역구"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
            ],
        )
        .properties(title="지역구별 거래량", height=320)
    )
    return soft_chart(chart)


def chart_area_group_price(df):
    area_order = ["소형(60㎡ 이하)", "중형(60-85㎡)", "대형(85-135㎡)", "초대형(135㎡ 초과)"]
    summary = (
        df.dropna(subset=["area_group"])
        .groupby("area_group", observed=True, as_index=False)
        .agg(
            median_pyeong=("price_per_pyeong", "median"),
            median_price=("price_10k_krw", "median"),
            deals=("price_10k_krw", "count"),
        )
    )
    summary["area_group"] = summary["area_group"].astype(str)
    chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=12, cornerRadiusTopRight=12, color="#FF9FB7")
        .encode(
            x=alt.X("area_group:N", sort=area_order, title="전용면적 구간"),
            y=alt.Y("median_pyeong:Q", title="중위 평당가(만원/평)"),
            tooltip=[
                alt.Tooltip("area_group:N", title="면적 구간"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
            ],
        )
        .properties(title="전용면적 구간별 시세", height=320)
    )
    return soft_chart(chart)


def chart_monthly_trend(df):
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            deals=("price_10k_krw", "count"),
            median_pyeong=("price_per_pyeong", "median"),
            median_price=("price_10k_krw", "median"),
        )
        .sort_values("month")
    )
    bars = (
        alt.Chart(monthly)
        .mark_bar(color="#93C5FD", opacity=0.86, cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
        .encode(
            x=alt.X(
                "month:T",
                title="계약월",
                axis=alt.Axis(
                    format="%m월",
                    labelAngle=0,
                    labelExpr="month(datum.value) == 0 ? timeFormat(datum.value, '%Y년 1월') : timeFormat(datum.value, '%m월')",
                ),
            ),
            y=alt.Y("deals:Q", title="거래량"),
            tooltip=[
                alt.Tooltip("month:T", title="월", format="%Y-%m"),
                alt.Tooltip("deals:Q", title="거래량", format=","),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
            ],
        )
    )
    line = (
        alt.Chart(monthly)
        .mark_line(color="#2563EB", point={"filled": True, "size": 90}, strokeWidth=3)
        .encode(
            x=alt.X("month:T"),
            y=alt.Y("median_pyeong:Q", title="중위 평당가(만원/평)"),
            tooltip=[
                alt.Tooltip("month:T", title="월", format="%Y-%m"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
            ],
        )
    )
    return soft_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=360))


def chart_district_month_heatmap(df):
    heat = (
        df.groupby(["district", "month"], as_index=False)
        .agg(median_pyeong=("price_per_pyeong", "median"), deals=("price_10k_krw", "count"))
    )
    chart = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X(
                "month:T",
                title="계약월",
                axis=alt.Axis(
                    format="%m월",
                    labelAngle=0,
                    labelExpr="month(datum.value) == 0 ? timeFormat(datum.value, '%Y년 1월') : timeFormat(datum.value, '%m월')",
                ),
            ),
            y=alt.Y("district:N", title="구", sort="-x"),
            color=alt.Color(
                "median_pyeong:Q",
                title="중위 평당가",
                scale=alt.Scale(range=["#DBEAFE", "#93C5FD", "#2563EB", "#1E40AF"]),
            ),
            tooltip=[
                alt.Tooltip("district:N", title="구"),
                alt.Tooltip("month:T", title="월", format="%Y-%m"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
            ],
        )
        .properties(height=330)
    )
    return soft_chart(chart)


def area_price_heatmap_figure(df, selected_district):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    area_bins = [0, 40, 60, 85, 102, 135, 165, float("inf")]
    area_labels = ["40㎡ 미만", "40~60㎡", "60~85㎡", "85~102㎡", "102~135㎡", "135~165㎡", "165㎡ 이상"]
    heat_df = df.copy()
    heat_df["area_band"] = pd.cut(
        heat_df["area_m2"],
        bins=area_bins,
        labels=area_labels,
        right=False,
    )

    index_col = "neighborhood" if selected_district != "전체" else "district"
    index_label = "동" if selected_district != "전체" else "지역구"
    title_prefix = selected_district if selected_district != "전체" else "대구 전체"

    if selected_district != "전체":
        top_index = (
            heat_df.groupby(index_col)["price_10k_krw"]
            .count()
            .sort_values(ascending=False)
            .head(14)
            .index
        )
        heat_df = heat_df[heat_df[index_col].isin(top_index)]

    pivot = heat_df.pivot_table(
        index=index_col,
        columns="area_band",
        values="price_per_pyeong",
        aggfunc="median",
        observed=False,
    ).reindex(columns=area_labels)
    pivot = pivot.dropna(how="all")
    pivot = pivot.loc[pivot.median(axis=1).sort_values(ascending=False).index]

    fig_height = max(4.8, min(8.5, 1.0 + len(pivot) * 0.42))
    fig, ax = plt.subplots(figsize=(11.6, fig_height))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    sns.heatmap(
        pivot,
        annot=True,
        fmt=",.0f",
        cmap="Blues",
        mask=pivot.isna(),
        linewidths=1.4,
        linecolor="#E2E8F0",
        cbar_kws={"label": "중위 평당가(만원/평)", "shrink": 0.82},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(
        f"{title_prefix} {index_label} × 전용면적 구간별 중위 평당가",
        fontsize=15,
        fontweight="bold",
        color="#1F2937",
        pad=16,
    )
    ax.set_xlabel("전용면적 구간", fontsize=12, color="#1F2937", labelpad=10)
    ax.set_ylabel(index_label, fontsize=12, color="#1F2937", labelpad=10)
    ax.tick_params(axis="x", colors="#1F2937", labelrotation=0, labelsize=10)
    ax.tick_params(axis="y", colors="#1F2937", labelrotation=0, labelsize=10)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.label.set_color("#1F2937")
    colorbar.ax.tick_params(colors="#1F2937")
    fig.tight_layout()
    return fig

def chart_focus_monthly_price(df, selected_district):
    focus = df.copy()
    title_prefix = "대구 전체" if selected_district == "전체" else selected_district
    monthly = (
        focus.groupby("month", as_index=False)
        .agg(
            avg_pyeong=("price_per_pyeong", "mean"),
            median_pyeong=("price_per_pyeong", "median"),
            deals=("price_10k_krw", "count"),
        )
        .sort_values("month")
    )
    line_data = monthly.melt(
        id_vars=["month", "deals"],
        value_vars=["avg_pyeong", "median_pyeong"],
        var_name="metric",
        value_name="price_per_pyeong",
    )
    line_data["metric"] = line_data["metric"].map(
        {"avg_pyeong": "평균 평당가", "median_pyeong": "중위 평당가"}
    )
    chart = (
        alt.Chart(line_data)
        .mark_line(point={"filled": True, "size": 92}, strokeWidth=3)
        .encode(
            x=alt.X(
                "month:T",
                title="계약월",
                axis=alt.Axis(
                    format="%m월",
                    labelAngle=0,
                    labelExpr="month(datum.value) == 0 ? timeFormat(datum.value, '%Y년 1월') : timeFormat(datum.value, '%m월')",
                ),
            ),
            y=alt.Y("price_per_pyeong:Q", title="평당가(만원/평)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "metric:N",
                title=None,
                scale=alt.Scale(range=["#2563EB", "#60A5FA"]),
            ),
            tooltip=[
                alt.Tooltip("month:T", title="월", format="%Y-%m"),
                alt.Tooltip("metric:N", title="지표"),
                alt.Tooltip("price_per_pyeong:Q", title="평당가", format=",.0f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
            ],
        )
        .properties(title=f"{title_prefix} 월별 시세변동 추이", height=330)
    )
    return soft_chart(chart)


def chart_neighborhood_price_compare(df, selected_district):
    focus = df.copy()
    title_prefix = "대구 전체" if selected_district == "전체" else selected_district
    neighborhood = (
        focus.groupby("neighborhood", as_index=False)
        .agg(
            median_pyeong=("price_per_pyeong", "median"),
            avg_pyeong=("price_per_pyeong", "mean"),
            deals=("price_10k_krw", "count"),
            median_price=("price_10k_krw", "median"),
        )
        .query("deals >= 3")
        .sort_values("median_pyeong", ascending=False)
        .head(18)
    )
    avg_row = pd.DataFrame(
        {
            "neighborhood": ["평균"],
            "median_pyeong": [focus["price_per_pyeong"].median()],
            "avg_pyeong": [focus["price_per_pyeong"].mean()],
            "deals": [len(focus)],
            "median_price": [focus["price_10k_krw"].median()],
        }
    )
    plot_df = pd.concat([avg_row, neighborhood], ignore_index=True)
    plot_df["kind"] = plot_df["neighborhood"].eq("평균").map({True: "지역 평균", False: "동"})
    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
        .encode(
            x=alt.X("neighborhood:N", title="동", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("median_pyeong:Q", title="중위 평당가(만원/평)"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(domain=["지역 평균", "동"], range=["#93C5FD", "#2563EB"]),
            ),
            tooltip=[
                alt.Tooltip("neighborhood:N", title="지역"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("avg_pyeong:Q", title="평균 평당가", format=",.0f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
            ],
        )
        .properties(title=f"{title_prefix} 동별 면적당 가격 비교", height=360)
    )
    return soft_chart(chart)


def chart_neighborhood_volume_top5(df, selected_district):
    title_prefix = selected_district if selected_district != "전체" else "대구 전체"
    volume = (
        df.groupby("neighborhood", as_index=False)
        .agg(
            deals=("price_10k_krw", "count"),
            median_price=("price_10k_krw", "median"),
            median_pyeong=("price_per_pyeong", "median"),
        )
        .sort_values("deals", ascending=False)
    )
    if volume.empty:
        return soft_chart(alt.Chart(pd.DataFrame({"label": [], "deals": []})).mark_bar())

    top5 = volume.head(5).copy()
    avg_row = pd.DataFrame(
        {
            "neighborhood": ["동 평균"],
            "deals": [volume["deals"].mean()],
            "median_price": [df["price_10k_krw"].median()],
            "median_pyeong": [df["price_per_pyeong"].median()],
        }
    )
    plot_df = pd.concat([avg_row, top5], ignore_index=True)
    plot_df["kind"] = plot_df["neighborhood"].eq("동 평균").map({True: "평균", False: "TOP 5"})
    plot_df["sort_order"] = range(len(plot_df))

    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopRight=12, cornerRadiusBottomRight=12)
        .encode(
            y=alt.Y("neighborhood:N", sort=plot_df["neighborhood"].tolist(), title="동"),
            x=alt.X("deals:Q", title="거래건수"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(domain=["평균", "TOP 5"], range=["#93C5FD", "#2563EB"]),
            ),
            tooltip=[
                alt.Tooltip("neighborhood:N", title="동"),
                alt.Tooltip("deals:Q", title="거래건수", format=",.1f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
            ],
        )
        .properties(title=f"{title_prefix} 거래량 TOP 5 동과 평균 비교", height=330)
    )
    return soft_chart(chart)


def chart_neighborhood_volume_price_scatter(df, selected_district):
    title_prefix = selected_district if selected_district != "전체" else "대구 전체"
    scatter_df = (
        df.groupby("neighborhood", as_index=False)
        .agg(
            deals=("price_10k_krw", "count"),
            median_pyeong=("price_per_pyeong", "median"),
            median_price=("price_10k_krw", "median"),
            avg_area=("area_m2", "mean"),
        )
        .query("deals >= 3")
        .copy()
    )
    if len(scatter_df) < 3:
        return soft_chart(alt.Chart(pd.DataFrame({"deals": [], "median_pyeong": []})).mark_circle())

    x = scatter_df["deals"].to_numpy(dtype=float)
    y = scatter_df["median_pyeong"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    r2 = corr ** 2
    x_line = np.linspace(x.min(), x.max(), 80)
    reg_df = pd.DataFrame({"deals": x_line, "fit": slope * x_line + intercept})

    points = (
        alt.Chart(scatter_df)
        .mark_circle(size=110, opacity=0.86, stroke="#FFFFFF", strokeWidth=1.2)
        .encode(
            x=alt.X("deals:Q", title="동별 거래건수"),
            y=alt.Y("median_pyeong:Q", title="중위 평당가(만원/평)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "neighborhood:N",
                title="동",
                scale=alt.Scale(scheme="tableau20"),
                legend=alt.Legend(columns=2, symbolLimit=18),
            ),
            tooltip=[
                alt.Tooltip("neighborhood:N", title="동"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
                alt.Tooltip("median_pyeong:Q", title="중위 평당가", format=",.0f"),
                alt.Tooltip("median_price:Q", title="중위 거래가", format=",.0f"),
                alt.Tooltip("avg_area:Q", title="평균 전용면적", format=",.1f"),
            ],
        )
    )
    line = (
        alt.Chart(reg_df)
        .mark_line(color="#2563EB", strokeWidth=3)
        .encode(x="deals:Q", y="fit:Q")
    )
    annotation = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [x.min()],
                    "y": [y.max()],
                    "label": [f"y={slope:.2f}x+{intercept:.0f}  |  r={corr:.3f}, R²={r2:.3f}"],
                }
            )
        )
        .mark_text(align="left", baseline="top", dx=8, dy=8, color="#1F2937", fontSize=14, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )
    chart = (points + line + annotation).properties(
        title=f"{title_prefix} 동별 거래량과 중위 평당가 상관관계",
        height=420,
    )
    return soft_chart(chart)


def chart_district_trend_correlation(df, selected_district):
    monthly = (
        df.groupby(["district", "month"], as_index=False)
        .agg(median_pyeong=("price_per_pyeong", "median"))
    )
    overall = (
        df.groupby("month", as_index=False)
        .agg(overall_pyeong=("price_per_pyeong", "median"))
    )
    pivot = monthly.pivot(index="month", columns="district", values="median_pyeong")
    joined = pivot.join(overall.set_index("month"), how="inner")
    rows = []
    for district in pivot.columns:
        pair = joined[[district, "overall_pyeong"]].dropna()
        corr = pair[district].corr(pair["overall_pyeong"]) if len(pair) >= 3 else None
        rows.append(
            {
                "district": district,
                "correlation": corr,
                "months": len(pair),
                "selected": "선택 지역" if district == selected_district else "지역구",
            }
        )
    corr_df = pd.DataFrame(rows).dropna(subset=["correlation"]).sort_values("correlation", ascending=False)
    chart = (
        alt.Chart(corr_df)
        .mark_bar(cornerRadiusTopRight=12, cornerRadiusBottomRight=12)
        .encode(
            y=alt.Y("district:N", sort="-x", title="지역구"),
            x=alt.X("correlation:Q", title="대구 전체 월별 시세와의 상관계수", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color(
                "selected:N",
                title=None,
                scale=alt.Scale(domain=["선택 지역", "지역구"], range=["#93C5FD", "#2563EB"]),
            ),
            tooltip=[
                alt.Tooltip("district:N", title="지역구"),
                alt.Tooltip("correlation:Q", title="상관계수", format=".3f"),
                alt.Tooltip("months:Q", title="비교 월수", format=","),
            ],
        )
        .properties(title="대구 전체 시세변동과 지역구별 시세변동 상관계수", height=360)
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6B7280", strokeDash=[4, 4]).encode(x="x:Q")
    return soft_chart(chart + rule)


def chart_selected_district_vs_daegu_scatter(df, selected_district):
    if selected_district == "전체":
        empty = pd.DataFrame({"overall_pyeong": [], "district_pyeong": []})
        return soft_chart(alt.Chart(empty).mark_circle())

    overall = (
        df.groupby("month", as_index=False)
        .agg(overall_pyeong=("price_per_pyeong", "median"))
    )
    district = (
        df[df["district"] == selected_district]
        .groupby("month", as_index=False)
        .agg(
            district_pyeong=("price_per_pyeong", "median"),
            deals=("price_10k_krw", "count"),
        )
    )
    compare = overall.merge(district, on="month", how="inner").dropna()
    if len(compare) < 3:
        return soft_chart(alt.Chart(pd.DataFrame({"overall_pyeong": [], "district_pyeong": []})).mark_circle())

    x = compare["overall_pyeong"].to_numpy(dtype=float)
    y = compare["district_pyeong"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    r2 = corr ** 2
    x_line = np.linspace(x.min(), x.max(), 80)
    reg_df = pd.DataFrame({"overall_pyeong": x_line, "fit": slope * x_line + intercept})
    compare["month_label"] = compare["month"].dt.strftime("%Y-%m")

    points = (
        alt.Chart(compare)
        .mark_circle(size=125, opacity=0.88, color="#2563EB", stroke="#FFFFFF", strokeWidth=1.2)
        .encode(
            x=alt.X("overall_pyeong:Q", title="대구 전체 중위 평당가(만원/평)", scale=alt.Scale(zero=False)),
            y=alt.Y("district_pyeong:Q", title=f"{selected_district} 중위 평당가(만원/평)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("month_label:N", title="계약월"),
                alt.Tooltip("overall_pyeong:Q", title="대구 전체", format=",.0f"),
                alt.Tooltip("district_pyeong:Q", title=selected_district, format=",.0f"),
                alt.Tooltip("deals:Q", title="지역구 거래건수", format=","),
            ],
        )
    )
    line = (
        alt.Chart(reg_df)
        .mark_line(color="#2563EB", strokeWidth=3)
        .encode(x="overall_pyeong:Q", y="fit:Q")
    )
    annotation = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [x.min()],
                    "y": [y.max()],
                    "label": [f"y={slope:.2f}x+{intercept:.0f}  |  r={corr:.3f}, R²={r2:.3f}"],
                }
            )
        )
        .mark_text(align="left", baseline="top", dx=8, dy=8, color="#1F2937", fontSize=14, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )
    chart = (points + line + annotation).properties(
        title=f"대구 전체와 {selected_district} 월별 시세변화 상관관계",
        height=410,
    )
    return soft_chart(chart)


def chart_district_market_influence(df, selected_district):
    overall = (
        df.groupby("month", as_index=False)
        .agg(overall_pyeong=("price_per_pyeong", "median"))
        .sort_values("month")
    )
    overall["overall_change"] = overall["overall_pyeong"].pct_change()

    district_monthly = (
        df.groupby(["district", "month"], as_index=False)
        .agg(
            district_pyeong=("price_per_pyeong", "median"),
            monthly_deals=("price_10k_krw", "count"),
        )
        .sort_values(["district", "month"])
    )
    district_monthly["district_change"] = district_monthly.groupby("district")["district_pyeong"].pct_change()
    joined = district_monthly.merge(overall[["month", "overall_change"]], on="month", how="inner")

    total_deals = len(df)
    rows = []
    for district, group in joined.groupby("district"):
        pair = group[["district_change", "overall_change"]].dropna()
        corr = pair["district_change"].corr(pair["overall_change"]) if len(pair) >= 3 else None
        deals = int(df.loc[df["district"] == district, "price_10k_krw"].count())
        share = deals / total_deals if total_deals else 0
        rows.append(
            {
                "district": district,
                "change_correlation": corr,
                "deal_share": share * 100,
                "deals": deals,
                "impact_score": (corr if pd.notna(corr) else 0) * share * 100,
                "selected": "선택 지역" if district == selected_district else "지역구",
            }
        )

    plot_df = pd.DataFrame(rows).dropna(subset=["change_correlation"])
    if plot_df.empty:
        return soft_chart(alt.Chart(pd.DataFrame({"deal_share": [], "change_correlation": []})).mark_circle())

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.88, stroke="#FFFFFF", strokeWidth=1.4)
        .encode(
            x=alt.X("deal_share:Q", title="거래량 비중(%)"),
            y=alt.Y(
                "change_correlation:Q",
                title="대구 전체 월별 시세변화와의 상관계수",
                scale=alt.Scale(domain=[-1, 1]),
            ),
            size=alt.Size("deals:Q", title="거래건수", scale=alt.Scale(range=[160, 1200])),
            color=alt.Color(
                "selected:N",
                title=None,
                scale=alt.Scale(domain=["선택 지역", "지역구"], range=["#93C5FD", "#2563EB"]),
            ),
            tooltip=[
                alt.Tooltip("district:N", title="지역구"),
                alt.Tooltip("change_correlation:Q", title="시세변화 상관계수", format=".3f"),
                alt.Tooltip("deal_share:Q", title="거래량 비중", format=".2f"),
                alt.Tooltip("deals:Q", title="거래건수", format=","),
                alt.Tooltip("impact_score:Q", title="영향도 참고값", format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(plot_df)
        .mark_text(dx=9, dy=-7, color="#1F2937", fontSize=12, fontWeight="bold")
        .encode(
            x="deal_share:Q",
            y="change_correlation:Q",
            text="district:N",
        )
    )
    hline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#6B7280", strokeDash=[4, 4]).encode(y="y:Q")
    chart = (scatter + labels + hline).properties(
        title="지역구별 거래량 비중과 대구 전체 시세변화 유사도",
        height=430,
    )
    return soft_chart(chart)


def district_summary_cards(df):
    summary = (
        df.groupby("district", as_index=False)
        .agg(
            median_pyeong=("price_per_pyeong", "median"),
            median_price=("price_10k_krw", "median"),
            deals=("price_10k_krw", "count"),
        )
        .sort_values("median_pyeong", ascending=False)
    )
    return summary


st.markdown(
    """
    <style>
    :root {
        --soft-bg: #F5F9FF;
        --soft-surface: #FFFFFF;
        --soft-surface-blue: #F8FBFF;
        --soft-border: #DDE8F7;
        --soft-text: #1F2937;
        --soft-muted: #64748B;
        --soft-accent: #2563EB;
        --soft-accent-dark: #1D4ED8;
        --soft-light-shadow: rgba(255, 255, 255, .88);
        --soft-dark-shadow: rgba(37, 99, 235, .10);
        --soft-shadow-raised: 0 18px 38px rgba(37, 99, 235, .12);
        --soft-shadow-small: 0 10px 24px rgba(37, 99, 235, .08);
        --soft-shadow-inset: inset 0 0 0 1px var(--soft-border);
        --soft-radius-lg: 32px;
        --soft-radius-md: 24px;
        --soft-radius-sm: 18px;
        --soft-transition: 300ms ease;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--soft-bg);
        color: var(--soft-text);
    }
    .block-container {
        padding-top: 1.4rem;
        max-width: 1420px;
        color: var(--soft-text);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: var(--soft-bg);
        border-right: 0;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--soft-bg);
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] div[data-testid="stFileUploader"],
    [data-testid="stSidebar"] div[data-baseweb="select"],
    [data-testid="stSidebar"] div[data-testid="stSlider"] {
        color: var(--soft-text);
    }
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
        background: var(--soft-surface);
        border: 0;
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-inset);
    }
    .market-filter-title {
        margin-top: 18px;
        color: var(--soft-text);
        font-size: 1.08rem;
        font-weight: 900;
    }
    .market-filter-help {
        margin: 6px 0 12px;
        color: var(--soft-muted);
        font-size: .9rem;
        line-height: 1.45;
        font-weight: 700;
    }
    div[data-testid="stSlider"] {
        padding: 16px 18px 12px;
        border-radius: var(--soft-radius-md);
        background: var(--soft-surface);
        box-shadow: var(--soft-shadow-inset);
        margin-bottom: 14px;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 10px;
    }
    div[data-testid="stSlider"] p {
        color: var(--soft-muted);
        font-size: .78rem;
        font-weight: 800;
    }
    .area-range-card {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 10px;
        align-items: center;
        padding: 14px;
        border-radius: var(--soft-radius-md);
        background: var(--soft-surface);
        border: 1px solid var(--soft-border);
        box-shadow: var(--soft-shadow-small);
    }
    .area-range-card div {
        display: grid;
        gap: 2px;
        text-align: center;
    }
    .area-range-card strong {
        color: var(--soft-accent);
        font-size: 1.05rem;
    }
    .area-range-card span {
        color: var(--soft-muted);
        font-size: .82rem;
        font-weight: 800;
    }
    .area-range-card b {
        color: var(--soft-muted);
        font-size: 1rem;
    }
    .filter-summary {
        margin: 8px 0 20px;
        color: var(--soft-muted);
        font-weight: 800;
        font-size: .92rem;
    }
    h1, h2, h3, p, label, span {
        color: var(--soft-text);
    }
    .hero {
        min-height: 310px;
        border-radius: var(--soft-radius-lg);
        background: var(--soft-surface);
        color: var(--soft-text);
        padding: 48px 54px;
        display: grid;
        grid-template-columns: 1.2fr .8fr;
        gap: 28px;
        overflow: hidden;
        margin-bottom: 28px;
        box-shadow: var(--soft-shadow-raised);
        position: relative;
        transition: transform var(--soft-transition), box-shadow var(--soft-transition);
    }
    .hero:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 44px rgba(37, 99, 235, .14);
    }
    .intro-hero {
        min-height: min(68vh, 620px);
        align-items: center;
        margin: 0 0 18px;
        padding: 76px 68px;
    }
    .intro-hero h1 {
        font-size: clamp(48px, 6vw, 86px);
        line-height: 1.12;
    }
    .intro-hero p {
        margin-top: 34px;
        font-size: clamp(22px, 2.2vw, 34px);
        line-height: 1.85;
    }
    .intro-action {
        margin-top: 42px;
        max-width: 260px;
    }
    .dashboard-frame {
        display: grid;
        grid-template-columns: 190px minmax(0, 1fr);
        gap: 24px;
        align-items: start;
    }
    .nav-panel {
        position: sticky;
        top: 18px;
        padding: 18px;
        border-radius: var(--soft-radius-md);
        background: var(--soft-surface);
        box-shadow: var(--soft-shadow-small);
    }
    .nav-panel h2 {
        margin: 0 0 4px;
        color: var(--soft-text);
        font-size: 1.02rem;
        font-weight: 950;
        letter-spacing: 0;
    }
    .nav-panel p {
        margin: 0 0 16px;
        color: var(--soft-muted);
        font-size: .78rem;
        line-height: 1.45;
        font-weight: 750;
    }
    .page-kicker {
        margin: 0 0 6px;
        color: var(--soft-accent);
        font-size: .82rem;
        font-weight: 900;
    }
    .page-title {
        margin: 0 0 18px;
        color: var(--soft-text);
        font-size: clamp(28px, 3vw, 44px);
        line-height: 1.18;
        font-weight: 950;
        letter-spacing: 0;
    }
    .hero:after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 270px;
        height: 270px;
        border-radius: 50%;
        background: rgba(37, 99, 235, .08);
        box-shadow: var(--soft-shadow-inset);
    }
    .hero h1 {
        font-size: 46px;
        line-height: 1.12;
        letter-spacing: 0;
        margin: 0 0 22px;
        font-weight: 800;
    }
    .hero p {
        font-size: 21px;
        line-height: 1.8;
        margin: 0;
        font-weight: 650;
    }
    .hero-visual {
        position: relative;
        min-height: 230px;
    }
    .building {
        position: absolute;
        right: 78px;
        top: 20px;
        width: 150px;
        height: 215px;
        background: linear-gradient(135deg, #FFFFFF 0%, #DBEAFE 46%, #93C5FD 100%);
        box-shadow: var(--soft-shadow-raised);
        transform: skewY(-7deg);
        border: 4px solid rgba(255, 255, 255, .42);
        border-radius: 18px;
    }
    .building:before {
        content: "";
        position: absolute;
        top: 18px;
        left: 20px;
        right: 20px;
        bottom: 20px;
        background:
          repeating-linear-gradient(90deg, transparent 0 16px, rgba(255,255,255,.82) 17px 28px),
          repeating-linear-gradient(0deg, transparent 0 22px, rgba(255,255,255,.7) 23px 33px);
        opacity: .95;
    }
    .pin {
        position: absolute;
        right: 28px;
        bottom: 20px;
        width: 78px;
        height: 78px;
        border-radius: 50% 50% 50% 0;
        background: linear-gradient(135deg, #60A5FA 0%, var(--soft-accent) 72%);
        transform: rotate(-45deg);
        box-shadow: var(--soft-shadow-small);
    }
    .pin:after {
        content: "";
        position: absolute;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: var(--soft-bg);
        border: 8px solid rgba(255,255,255,.45);
        left: 14px;
        top: 14px;
    }
    .section-title {
        color: var(--soft-text);
        font-size: 1.25rem;
        font-weight: 800;
        margin: .5rem 0 .8rem;
    }
    div[data-testid="stMetric"] {
        background: var(--soft-surface);
        border: 0;
        border-radius: var(--soft-radius-md);
        padding: 16px 18px;
        box-shadow: var(--soft-shadow-small);
        transition: transform var(--soft-transition), box-shadow var(--soft-transition);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--soft-shadow-raised);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--soft-muted);
        font-weight: 700;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--soft-text);
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stAltairChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stImage"],
    div[data-testid="stPyplot"] {
        background: var(--soft-surface);
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-small);
        padding: 12px;
    }
    div[data-testid="stAltairChart"] svg {
        border-radius: var(--soft-radius-sm);
    }
    .district-list {
        display: grid;
        gap: 12px;
        max-height: 360px;
        overflow-y: auto;
        padding: 6px 8px 8px 2px;
    }
    .district-row {
        display: grid;
        grid-template-columns: 72px 1fr;
        gap: 12px;
        align-items: center;
        background: var(--soft-surface);
        border-radius: 20px;
        padding: 13px 15px;
        box-shadow: var(--soft-shadow-small);
        transition: transform var(--soft-transition), box-shadow var(--soft-transition);
    }
    .district-row:hover {
        transform: translateY(-2px);
        box-shadow: var(--soft-shadow-raised);
    }
    .district-name {
        color: var(--soft-accent);
        font-weight: 850;
        font-size: 1.05rem;
    }
    .district-values {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
    }
    .district-stat {
        color: var(--soft-text);
        font-weight: 800;
        font-size: .94rem;
        line-height: 1.25;
    }
    .district-stat span {
        display: block;
        color: var(--soft-muted);
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin: 12px 0 22px;
    }
    .insight-card {
        padding: 20px 22px;
        border-radius: var(--soft-radius-md);
        background: var(--soft-surface);
        box-shadow: var(--soft-shadow-small);
        color: var(--soft-text);
        min-height: 118px;
    }
    .insight-card span {
        display: block;
        color: var(--soft-muted);
        font-size: .88rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .insight-card strong {
        display: block;
        font-size: 1.18rem;
        margin-bottom: 8px;
    }
    .insight-card p {
        margin: 0;
        color: var(--soft-muted);
        line-height: 1.55;
        font-size: .94rem;
    }
    .insight-metrics {
        display: grid;
        gap: 7px;
        margin-top: 8px;
    }
    .insight-metric {
        color: var(--soft-muted);
        font-size: .95rem;
        font-weight: 750;
        line-height: 1.45;
    }
    .insight-card .metric-up {
        display: inline;
        margin: 0;
        color: var(--soft-accent);
        font-weight: 900;
        white-space: nowrap;
    }
    .dong-rank-card {
        background: var(--soft-surface);
        border: 1px solid var(--soft-border);
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-small);
        padding: 20px 22px;
        margin-bottom: 18px;
    }
    .dong-rank-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
    }
    .dong-rank-header strong {
        color: var(--soft-text);
        font-size: 1.18rem;
        font-weight: 950;
    }
    .dong-rank-header span {
        color: var(--soft-muted);
        font-size: .82rem;
        font-weight: 800;
    }
    .dong-rank-list {
        display: grid;
        gap: 11px;
    }
    .dong-rank-row {
        display: grid;
        grid-template-columns: 38px minmax(74px, .8fr) 82px minmax(120px, 1fr);
        gap: 12px;
        align-items: center;
    }
    .dong-rank-no {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        display: inline-grid;
        place-items: center;
        background: var(--soft-accent);
        color: #FFFFFF;
        font-weight: 900;
        font-size: .92rem;
    }
    .dong-rank-row:nth-child(n+6) .dong-rank-no {
        background: #93C5FD;
    }
    .dong-rank-name {
        color: var(--soft-text);
        font-weight: 850;
        white-space: nowrap;
    }
    .dong-rank-value {
        color: var(--soft-text);
        font-weight: 900;
        text-align: right;
        white-space: nowrap;
    }
    .dong-rank-bar {
        height: 14px;
        border-radius: 999px;
        background: #EAF2FF;
        overflow: hidden;
    }
    .dong-rank-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #60A5FA, var(--soft-accent));
    }
    .region-summary-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid var(--soft-border);
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-small);
        padding: 20px 22px;
        min-height: 100%;
    }
    .region-summary-card strong {
        display: block;
        color: var(--soft-text);
        font-size: 1.7rem;
        font-weight: 950;
        margin: 8px 0 14px;
    }
    .region-summary-card span {
        color: var(--soft-muted);
        font-size: .86rem;
        font-weight: 850;
    }
    .region-summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 18px;
    }
    .region-summary-item {
        border-radius: 18px;
        background: #F8FBFF;
        border: 1px solid var(--soft-border);
        padding: 12px;
    }
    .region-summary-item b {
        display: block;
        color: var(--soft-accent);
        font-size: 1.05rem;
        margin-top: 4px;
    }
    .prediction-result-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid var(--soft-border);
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-small);
        padding: 24px;
        margin-bottom: 18px;
    }
    .prediction-result-head {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-start;
        margin-bottom: 18px;
    }
    .prediction-result-head span {
        color: var(--soft-muted);
        font-weight: 850;
        font-size: .9rem;
    }
    .prediction-result-head strong {
        display: block;
        color: var(--soft-text);
        font-size: clamp(2.25rem, 4vw, 4rem);
        line-height: 1.05;
        font-weight: 950;
        margin: 8px 0 6px;
        letter-spacing: 0;
    }
    .prediction-result-head p {
        margin: 0;
        color: var(--soft-accent);
        font-weight: 900;
    }
    .prediction-badge {
        flex: 0 0 auto;
        padding: 9px 12px;
        border-radius: 999px;
        color: var(--soft-accent);
        background: rgba(37, 99, 235, .08);
        border: 1px solid rgba(37, 99, 235, .16);
        font-size: .78rem;
        font-weight: 900;
        white-space: nowrap;
    }
    .prediction-context {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 18px;
    }
    .prediction-context span {
        padding: 8px 10px;
        border-radius: 999px;
        background: #EFF6FF;
        color: var(--soft-text);
        border: 1px solid var(--soft-border);
        font-size: .82rem;
        font-weight: 850;
    }
    .prediction-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 18px;
    }
    .prediction-mini-grid div {
        min-height: 86px;
        padding: 14px;
        border-radius: 18px;
        background: #F8FBFF;
        border: 1px solid var(--soft-border);
    }
    .prediction-mini-grid span {
        display: block;
        color: var(--soft-muted);
        font-size: .78rem;
        font-weight: 850;
        margin-bottom: 7px;
    }
    .prediction-mini-grid b {
        display: block;
        color: var(--soft-text);
        font-size: 1.1rem;
        font-weight: 950;
    }
    .prediction-mini-grid small {
        display: block;
        color: var(--soft-muted);
        margin-top: 3px;
        font-weight: 800;
    }
    .prediction-up {
        color: #DC2626 !important;
    }
    .prediction-down {
        color: var(--soft-accent) !important;
    }
    .prediction-blend {
        padding: 14px;
        border-radius: 18px;
        background: #F8FBFF;
        border: 1px solid var(--soft-border);
    }
    .prediction-blend-label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--soft-muted);
        font-size: .82rem;
        font-weight: 850;
        margin-bottom: 10px;
    }
    .prediction-blend-label b {
        color: var(--soft-text);
    }
    .prediction-blend-bar {
        display: flex;
        height: 16px;
        border-radius: 999px;
        overflow: hidden;
        background: #E5EAF3;
    }
    .prediction-model-bar {
        background: linear-gradient(90deg, var(--soft-accent), #60A5FA);
    }
    .prediction-recent-bar {
        background: linear-gradient(90deg, #93C5FD, #BFDBFE);
    }
    .prediction-note {
        margin: 14px 0 0;
        color: var(--soft-muted);
        line-height: 1.55;
        font-weight: 750;
    }
    .analysis-note {
        margin: 8px 0 24px;
        padding: 18px 20px;
        border-radius: var(--soft-radius-md);
        background: var(--soft-surface);
        box-shadow: var(--soft-shadow-inset);
        color: var(--soft-muted);
        line-height: 1.6;
        font-weight: 700;
    }
    .soft-table-wrap {
        background: var(--soft-surface);
        border-radius: var(--soft-radius-md);
        border: 1px solid var(--soft-border);
        box-shadow: var(--soft-shadow-small);
        padding: 14px;
        overflow: auto;
        max-height: 540px;
    }
    .soft-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
        color: var(--soft-text);
        font-size: .92rem;
    }
    .soft-table thead th {
        color: var(--soft-muted);
        font-weight: 850;
        text-align: left;
        padding: 9px 12px;
        white-space: nowrap;
    }
    .soft-table tbody tr {
        background: #F8FBFF;
        box-shadow: inset 0 -1px 0 var(--soft-border);
    }
    .soft-table tbody td {
        padding: 11px 12px;
        white-space: nowrap;
        border: 0;
    }
    .soft-table tbody td:first-child {
        border-radius: 16px 0 0 16px;
        color: var(--soft-accent);
        font-weight: 800;
    }
    .soft-table tbody td:last-child {
        border-radius: 0 16px 16px 0;
    }
    .soft-table-note {
        color: var(--soft-muted);
        font-size: .86rem;
        margin-top: 10px;
        font-weight: 700;
    }
    @media (max-width: 820px) {
        .district-row { grid-template-columns: 1fr; }
        .district-values { grid-template-columns: 1fr; }
    }
    div[role="radiogroup"] {
        background: var(--soft-surface);
        border-radius: var(--soft-radius-md);
        padding: 12px;
        box-shadow: var(--soft-shadow-inset);
    }
    div[role="radiogroup"] label {
        background: var(--soft-surface);
        border-radius: 16px;
        padding: 8px 10px;
        margin: 5px 0;
        box-shadow: var(--soft-shadow-small);
        transition: transform var(--soft-transition), box-shadow var(--soft-transition), color var(--soft-transition);
    }
    div[role="radiogroup"] label:hover {
        transform: translateX(3px);
        color: var(--soft-accent);
    }
    div[role="radiogroup"] label:has(input:checked) {
        color: var(--soft-accent);
        box-shadow: var(--soft-shadow-inset);
        font-weight: 800;
    }
    div[data-baseweb="select"] > div {
        background: var(--soft-surface);
        border: 0;
        border-radius: var(--soft-radius-sm);
        box-shadow: var(--soft-shadow-inset);
        color: var(--soft-text);
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] * {
        background: var(--soft-surface) !important;
        color: var(--soft-text) !important;
        border-color: transparent !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"] {
        border: 0 !important;
        border-radius: var(--soft-radius-sm) !important;
        box-shadow: 0 16px 34px rgba(37, 99, 235, .14) !important;
        overflow: hidden !important;
    }
    ul[role="listbox"],
    div[role="listbox"],
    ul[role="listbox"] *,
    div[role="listbox"] * {
        background: var(--soft-surface) !important;
        color: var(--soft-text) !important;
    }
    ul[role="listbox"],
    div[role="listbox"] {
        border-radius: var(--soft-radius-sm) !important;
        padding: 8px !important;
    }
    li[role="option"],
    div[role="option"],
    li[role="option"] *,
    div[role="option"] * {
        background: transparent !important;
        color: var(--soft-text) !important;
    }
    li[role="option"],
    div[role="option"] {
        border-radius: 14px !important;
        margin: 3px 0 !important;
        font-weight: 750 !important;
    }
    li[role="option"]:hover,
    div[role="option"]:hover,
    li[aria-selected="true"],
    div[aria-selected="true"],
    li[role="option"]:hover *,
    div[role="option"]:hover *,
    li[aria-selected="true"] *,
    div[aria-selected="true"] *,
    li[aria-highlighted="true"],
    div[aria-highlighted="true"],
    li[data-highlighted="true"],
    div[data-highlighted="true"] {
        background: rgba(37, 99, 235, .10) !important;
        color: var(--soft-accent) !important;
    }
    div[data-testid="stExpander"] {
        background: var(--soft-surface);
        border: 0;
        border-radius: var(--soft-radius-md);
        box-shadow: var(--soft-shadow-small);
        overflow: hidden;
    }
    div[data-testid="stExpander"] details {
        border: 0;
    }
    div[data-testid="stExpander"] summary {
        color: var(--soft-text);
        font-weight: 800;
    }
    div[data-testid="stNumberInput"] input {
        background: var(--soft-surface);
        color: var(--soft-text);
        border: 0;
        border-radius: 16px;
        box-shadow: var(--soft-shadow-inset);
    }
    div[data-testid="stNumberInput"] button {
        color: var(--soft-accent) !important;
    }
    div[data-baseweb="tag"] {
        background: rgba(37, 99, 235, .10) !important;
        color: var(--soft-accent) !important;
        border-radius: 14px !important;
    }
    button, div[data-testid="stDownloadButton"] button, div[data-testid="stFileUploader"] button {
        background: var(--soft-surface) !important;
        color: var(--soft-text) !important;
        border: 0 !important;
        border-radius: var(--soft-radius-sm) !important;
        box-shadow: var(--soft-shadow-small) !important;
        transition: transform var(--soft-transition), box-shadow var(--soft-transition), color var(--soft-transition) !important;
    }
    button:hover, div[data-testid="stDownloadButton"] button:hover, div[data-testid="stFileUploader"] button:hover {
        color: var(--soft-accent) !important;
        transform: translateY(-2px);
    }
    button:active, div[data-testid="stDownloadButton"] button:active {
        box-shadow: var(--soft-shadow-inset) !important;
        transform: translateY(1px);
    }
    div[data-testid="stTabs"] button {
        border-radius: var(--soft-radius-sm) !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--soft-accent) !important;
        box-shadow: var(--soft-shadow-inset) !important;
    }
    .stSlider [data-baseweb="slider"] > div {
        color: var(--soft-accent);
    }
    @media (max-width: 820px) {
        .hero { grid-template-columns: 1fr; padding: 34px 28px; }
        .intro-hero { min-height: auto; padding: 46px 30px; }
        .hero h1 { font-size: 34px; }
        .hero p { font-size: 17px; }
        .hero-visual { display: none; }
        .dashboard-frame { grid-template-columns: 1fr; }
        .nav-panel { position: static; }
        .insight-grid { grid-template-columns: 1fr; }
        .district-values { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

data_session_key = f"apartment_data_v{DATA_CACHE_VERSION}"
if data_session_key not in st.session_state:
    st.session_state[data_session_key] = load_apartment_data(DATA_CACHE_VERSION)
df = st.session_state[data_session_key]

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 기본 실거래가 엑셀 파일 경로를 확인하세요.")
    st.stop()

districts = sorted(df["district"].dropna().unique().tolist())

if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "대구 전체 분석"
if st.session_state.active_page == "전체 시장 요약":
    st.session_state.active_page = "대구 전체 분석"

if not st.session_state.app_started:
    st.markdown(
        """
        <section class="hero intro-hero">
          <div>
            <h1>대구 아파트<br>매매 실거래가 분석</h1>
            <p>최근 실거래 데이터를 바탕으로<br>지역구별 시세와 월별 흐름을 한눈에 확인하세요.</p>
          </div>
          <div class="hero-visual">
            <div class="building"></div>
            <div class="pin"></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    start_col, _ = st.columns([0.18, 0.82])
    with start_col:
        if st.button("시작하기", type="primary", use_container_width=True):
            st.session_state.app_started = True
            st.session_state.active_page = "대구 전체 분석"
            st.rerun()
    st.stop()

nav_col, content_col = st.columns([0.18, 0.82], gap="large")
with nav_col:
    st.markdown(
        """
        <div class="nav-panel">
          <h2>분석 메뉴</h2>
          <p>보고 싶은 화면을 선택하세요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_page = st.radio(
        "페이지 선택",
        ["대구 전체 분석", "지역구 분석", "매매가 예측"],
        index=["대구 전체 분석", "지역구 분석", "매매가 예측"].index(st.session_state.active_page),
        key="active_page",
        label_visibility="collapsed",
    )
    if st.button("처음 화면", use_container_width=True):
        st.session_state.app_started = False
        st.rerun()

with content_col:
    if active_page != "매매가 예측":
        max_area_m2 = float(max(250, df["area_m2"].max()))
        st.markdown(
            """
            <div class="market-filter-title">전용면적 범위</div>
            <div class="market-filter-help">시장 요약과 지역구 분석에 적용할 전용면적 조건을 선택하세요.</div>
            """,
            unsafe_allow_html=True,
        )
        filter_left, filter_right = st.columns([0.68, 0.32])
        with filter_left:
            area_min_m2, area_max_m2 = st.slider(
                "전용면적 범위",
                min_value=0,
                max_value=int(np.ceil(max_area_m2)),
                value=(0, int(np.ceil(max_area_m2))),
                step=1,
                format="%d㎡",
                label_visibility="collapsed",
            )
        with filter_right:
            st.markdown(
                f"""
                <div class="area-range-card">
                  <div><strong>{area_min_m2:.0f}㎡</strong><span>{area_min_m2 / 3.3058:.1f}평</span></div>
                  <b>~</b>
                  <div><strong>{area_max_m2:.0f}㎡</strong><span>{area_max_m2 / 3.3058:.1f}평</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        area_filtered = df[
            df["area_m2"].between(area_min_m2, area_max_m2, inclusive="both")
        ].copy()
        filtered = area_filtered.copy()

        if filtered.empty:
            st.warning("선택한 전용면적 조건에 해당하는 실거래 데이터가 없습니다. 필터를 다시 선택해 주세요.")
            st.stop()

        st.caption("대상: 대구광역시 아파트 매매 실거래가 · 기준: 계약일 · 단위: 거래금액 만원")
        st.markdown(
            f"**현재 분석 지역:** `대구 전체` · **전용면적 조건:** `{area_min_m2:.0f}㎡ ~ {area_max_m2:.0f}㎡`"
        )

    if active_page == "대구 전체 분석":
        st.markdown('<div class="page-kicker">Market Summary</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="page-title">대구 전체 분석</h1>', unsafe_allow_html=True)

        market_summary = district_summary_cards(area_filtered)
        price_leader = market_summary.iloc[0] if not market_summary.empty else None
        price_runner_up = market_summary.iloc[1] if len(market_summary) > 1 else None
        overall_median_pyeong = area_filtered["price_per_pyeong"].median()
        price_gap_vs_overall = (
            (price_leader["median_pyeong"] - overall_median_pyeong) / overall_median_pyeong * 100
            if price_leader is not None and overall_median_pyeong
            else 0
        )
        price_gap_vs_second = (
            (price_leader["median_pyeong"] - price_runner_up["median_pyeong"]) / price_runner_up["median_pyeong"] * 100
            if price_leader is not None and price_runner_up is not None and price_runner_up["median_pyeong"]
            else 0
        )
        price_leader_neighborhoods = (
            area_filtered[area_filtered["district"] == price_leader["district"]]
            .groupby("neighborhood", as_index=False)
            .agg(
                median_pyeong=("price_per_pyeong", "median"),
                deals=("price_10k_krw", "count"),
            )
            .query("deals >= 10")
            .sort_values("median_pyeong", ascending=False)
            if price_leader is not None
            else pd.DataFrame()
        )
        price_leader_neighborhood = (
            price_leader_neighborhoods.iloc[0]["neighborhood"]
            if not price_leader_neighborhoods.empty
            else ""
        )
        price_leader_label = (
            f"{price_leader['district']} {price_leader_neighborhood}"
            if price_leader is not None and price_leader_neighborhood
            else (price_leader["district"] if price_leader is not None else "-")
        )
        volume_by_district = (
            area_filtered.groupby("district", as_index=False)
            .agg(deals=("price_10k_krw", "count"))
            .sort_values("deals", ascending=False)
        )
        volume_leader = volume_by_district.iloc[0] if not volume_by_district.empty else None
        volume_runner_up = volume_by_district.iloc[1] if len(volume_by_district) > 1 else None
        total_deals = len(area_filtered)
        volume_share = volume_leader["deals"] / total_deals * 100 if volume_leader is not None and total_deals else 0
        volume_gap_vs_second = (
            (volume_leader["deals"] - volume_runner_up["deals"]) / volume_runner_up["deals"] * 100
            if volume_leader is not None and volume_runner_up is not None and volume_runner_up["deals"]
            else 0
        )
        monthly_market = (
            area_filtered.groupby("month", as_index=False)
            .agg(median_pyeong=("price_per_pyeong", "median"), deals=("price_10k_krw", "count"))
            .sort_values("month")
        )
        if len(monthly_market) >= 2:
            recent_change = (
                (monthly_market.iloc[-1]["median_pyeong"] - monthly_market.iloc[-2]["median_pyeong"])
                / monthly_market.iloc[-2]["median_pyeong"]
                * 100
            )
            recent_month = monthly_market.iloc[-1]["month"].strftime("%Y-%m")
        else:
            recent_change = 0
            recent_month = "-"

        metric_cols = st.columns(5)
        metric_cols[0].metric("거래 건수", f"{len(filtered):,}건")
        metric_cols[1].metric("중위 거래가", format_price_uk(filtered["price_10k_krw"].median()))
        metric_cols[2].metric("중위 평당가", f"{filtered['price_per_pyeong'].median():,.0f}만원/평")
        metric_cols[3].metric("평균 전용면적", f"{filtered['area_m2'].mean():.1f}㎡")
        metric_cols[4].metric("거래량 최다 지역구", volume_leader["district"] if volume_leader is not None else "-")

        st.markdown(
            f"""
            <div class="insight-grid">
              <div class="insight-card">
                <span>평당가 상위</span>
                <strong>{price_leader_label}</strong>
                <div class="insight-metrics">
                  <div class="insight-metric">대구 전체 대비 <span class="metric-up">{price_gap_vs_overall:+.1f}% ↑</span></div>
                  <div class="insight-metric">2위 지역 대비 <span class="metric-up">{price_gap_vs_second:+.1f}% ↑</span></div>
                </div>
              </div>
              <div class="insight-card">
                <span>거래 집중</span>
                <strong>{volume_leader['district'] if volume_leader is not None else '-'} 거래량 최다</strong>
                <div class="insight-metrics">
                  <div class="insight-metric">전체 거래 비중 <span class="metric-up">{volume_share:.1f}%</span></div>
                  <div class="insight-metric">2위 지역 대비 거래량 <span class="metric-up">{volume_gap_vs_second:+.1f}% ↑</span></div>
                </div>
              </div>
              <div class="insight-card">
                <span>최근 흐름</span>
                <strong>{recent_month} 전월 대비 {recent_change:+.1f}%</strong>
                <p>월별 중위 평당가 기준으로 최근 시장 흐름이 상승인지 하락인지 빠르게 확인합니다.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="analysis-note">대구 전체 시장을 먼저 보고, 지역구별 시세 수준과 거래량이 어디에 집중되는지 비교하는 화면입니다.</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown('<div class="section-title">대구 전체 월별 시세 흐름</div>', unsafe_allow_html=True)
            st.altair_chart(chart_focus_monthly_price(area_filtered, "전체"), width="stretch")
        with right:
            st.markdown('<div class="section-title">월별 거래량과 중위 평당가</div>', unsafe_allow_html=True)
            st.altair_chart(chart_monthly_trend(area_filtered), width="stretch")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">지역구별 중위 평당가</div>', unsafe_allow_html=True)
            st.altair_chart(chart_district_price(area_filtered), width="stretch")
        with right:
            st.markdown('<div class="section-title">지역구별 거래량</div>', unsafe_allow_html=True)
            st.altair_chart(chart_district_volume(area_filtered), width="stretch")

        st.markdown('<div class="section-title">전용면적 구간별 시세</div>', unsafe_allow_html=True)
        st.altair_chart(chart_area_group_price(area_filtered), width="stretch")

        st.markdown('<div class="section-title">지역구별 거래량 비중과 대구 전체 시세변화 유사도</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-note">x축은 거래량 비중, y축은 대구 전체 월별 시세변화와의 유사도입니다. 오른쪽 위에 가까울수록 거래가 많고 전체 흐름과 비슷하게 움직이는 지역구입니다.</div>', unsafe_allow_html=True)
        st.altair_chart(chart_district_market_influence(area_filtered, "전체"), width="stretch")

    elif active_page == "지역구 분석":
        st.markdown('<div class="page-kicker">District Detail</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="page-title">지역구 분석</h1>', unsafe_allow_html=True)
        selector_left, selector_right = st.columns([0.28, 0.72])
        with selector_left:
            selected_region = st.selectbox(
                "지역구 선택",
                districts,
                index=districts.index("수성구") if "수성구" in districts else 0,
            )
        region_filtered = area_filtered[area_filtered["district"] == selected_region].copy()

        if region_filtered.empty:
            st.warning("선택한 지역구와 전용면적 조건에 해당하는 거래 데이터가 없습니다.")
        else:
            top_dong = (
                region_filtered.groupby("neighborhood", as_index=False)
                .agg(median_pyeong=("price_per_pyeong", "median"), deals=("price_10k_krw", "count"))
                .query("deals >= 3")
                .sort_values("median_pyeong", ascending=False)
            )
            top_dong_name = top_dong.iloc[0]["neighborhood"] if not top_dong.empty else "-"
            top_dong_price = top_dong.iloc[0]["median_pyeong"] if not top_dong.empty else 0

            with selector_left:
                st.markdown(
                    f"""
                    <div class="region-summary-card">
                      <span>선택 지역구</span>
                      <strong>{selected_region}</strong>
                      <div class="region-summary-grid">
                        <div class="region-summary-item"><span>거래 건수</span><b>{len(region_filtered):,}건</b></div>
                        <div class="region-summary-item"><span>중위 거래가</span><b>{format_price_uk(region_filtered["price_10k_krw"].median())}</b></div>
                        <div class="region-summary-item"><span>중위 평당가</span><b>{region_filtered["price_per_pyeong"].median():,.0f}</b></div>
                        <div class="region-summary-item"><span>상위 동</span><b>{top_dong_name}</b></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with selector_right:
                st.markdown(render_dong_rank_panel(region_filtered, selected_region), unsafe_allow_html=True)

            st.markdown(f'<div class="analysis-note">{selected_region}를 선택하면 동별 평균 매매가 순위와 월별 시세 흐름, 동별 면적당 가격, 거래량, 면적 구간 히트맵을 한 화면에서 이어서 확인합니다.</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">월별 시세변동 추이</div>', unsafe_allow_html=True)
            st.altair_chart(chart_focus_monthly_price(region_filtered, selected_region), width="stretch")

            left, right = st.columns([1.1, 0.9])
            with left:
                st.markdown('<div class="section-title">동별 면적당 가격 비교</div>', unsafe_allow_html=True)
                st.altair_chart(chart_neighborhood_price_compare(region_filtered, selected_region), width="stretch")
            with right:
                st.markdown('<div class="section-title">거래량 TOP 5 동과 평균 비교</div>', unsafe_allow_html=True)
                st.altair_chart(chart_neighborhood_volume_top5(region_filtered, selected_region), width="stretch")

            st.markdown('<div class="section-title">동 × 전용면적 구간별 중위 평당가 히트맵</div>', unsafe_allow_html=True)
            st.pyplot(area_price_heatmap_figure(region_filtered, selected_region), width="stretch")

    else:
        st.markdown('<div class="page-kicker">Price Prediction</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="page-title">매매가 예측</h1>', unsafe_allow_html=True)

    if active_page != "매매가 예측":
        st.stop()

    st.markdown(
        '<div class="analysis-note">2023~2026년 실거래 데이터를 기반으로 가상의 아파트 매매가를 추정합니다. 단지명을 선택하면 단지명 사용 모델을, 전체/모름을 선택하면 단지명 미사용 모델을 사용합니다.</div>',
        unsafe_allow_html=True,
    )
    prediction_data_key = f"prediction_data_v{DATA_CACHE_VERSION}"
    if prediction_data_key not in st.session_state:
        st.session_state[prediction_data_key] = prediction_dataset(df)
    prediction_df = st.session_state[prediction_data_key]

    model_bundle_key = f"price_model_bundle_v{DATA_CACHE_VERSION}_{MODEL_CACHE_VERSION}"
    if model_bundle_key not in st.session_state:
        st.session_state[model_bundle_key] = train_price_models(prediction_df, MODEL_CACHE_VERSION)
    bundle = st.session_state[model_bundle_key]

    input_left, input_right = st.columns([0.58, 0.42])
    with input_left:
        st.markdown('<div class="section-title">예측 조건</div>', unsafe_allow_html=True)
        gu_options = sorted(prediction_df["구"].dropna().unique())
        default_gu_idx = gu_options.index("수성구") if "수성구" in gu_options else 0
        gu = st.selectbox("구", gu_options, index=default_gu_idx, key="predict_gu")

        dong_options = sorted(prediction_df.loc[prediction_df["구"] == gu, "법정동"].dropna().unique())
        dong = st.selectbox("동", dong_options, key="predict_dong")

        location_df = prediction_df[(prediction_df["구"] == gu) & (prediction_df["법정동"] == dong)]
        sigungu_options = sorted(location_df["시군구"].dropna().unique())
        sigungu = sigungu_options[0] if sigungu_options else ""

        complex_options = ["전체/모름"] + sorted(location_df["단지명"].dropna().unique())
        complex_name = st.selectbox("단지명", complex_options, key="predict_complex")

        area_min = float(max(10, np.floor(prediction_df["전용면적(㎡)"].quantile(0.01))))
        area_max = float(np.ceil(prediction_df["전용면적(㎡)"].quantile(0.99)))
        area = st.number_input(
            "전용면적(㎡)",
            min_value=area_min,
            max_value=area_max,
            value=min(84.0, area_max),
            step=0.1,
            format="%.1f",
            key="predict_area",
        )
        st.markdown(
            f'<div class="filter-summary">전용면적 {area:,.1f}㎡ · {area / PYEONG_DIVISOR:,.1f}평</div>',
            unsafe_allow_html=True,
        )

        year_options = list(range(int(prediction_df["계약년도"].min()), int(prediction_df["계약년도"].max()) + 2))
        col_year, col_month = st.columns(2)
        with col_year:
            contract_year = st.selectbox(
                "계약년도",
                year_options,
                index=year_options.index(int(prediction_df["계약년도"].max())),
                key="predict_year",
            )
        with col_month:
            contract_month = st.selectbox("계약월", list(range(1, 13)), index=5, key="predict_month")

        built_min = int(prediction_df["건축년도"].min())
        built_max = max(int(prediction_df["건축년도"].max()), int(contract_year))
        built_year = st.number_input(
            "건축년도",
            min_value=built_min,
            max_value=built_max,
            value=min(2015, built_max),
            step=1,
            key="predict_built",
        )

        floor_min = int(prediction_df["층"].dropna().quantile(0.01))
        floor_max = int(prediction_df["층"].dropna().quantile(0.99))
        floor = st.number_input(
            "층",
            min_value=floor_min,
            max_value=floor_max,
            value=min(10, floor_max),
            step=1,
            key="predict_floor",
        )

    prediction_input = build_prediction_row(
        sigungu=sigungu,
        complex_name=complex_name,
        area=area,
        floor=floor,
        built_year=built_year,
        contract_year=contract_year,
        contract_month=contract_month,
    )
    matches = recent_matches(
        prediction_df,
        sigungu=sigungu,
        complex_name=complex_name,
        area=area,
        built_year=built_year,
        last_month=bundle.last_contract_month,
    )
    active_model = bundle.without_complex if complex_name == "전체/모름" else bundle.with_complex
    predicted_price = float(active_model.model.predict(prediction_input[active_model.features])[0])
    adjusted_price, recent_median_price, recent_weight = blend_with_recent_median(predicted_price, matches)
    adjusted_pyeong_price = adjusted_price / (area / PYEONG_DIVISOR)

    with input_right:
        st.markdown('<div class="section-title">예측 결과</div>', unsafe_allow_html=True)
        st.markdown(
            render_prediction_result_card(
                adjusted_price=adjusted_price,
                predicted_price=predicted_price,
                recent_median_price=recent_median_price,
                recent_weight=recent_weight,
                adjusted_pyeong_price=adjusted_pyeong_price,
                area=area,
                floor=floor,
                built_year=built_year,
                contract_year=contract_year,
                contract_month=contract_month,
                gu=gu,
                dong=dong,
                complex_name=complex_name,
                active_model=active_model,
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">최근 1년 유사 거래</div>', unsafe_allow_html=True)
    summary_cols = st.columns(4)
    if matches.empty:
        st.warning("조건에 맞는 최근 1년 거래가 없습니다. 조건을 조금 넓혀보세요.")
    else:
        summary_cols[0].metric("거래 건수", f"{len(matches):,}건")
        summary_cols[1].metric("최근 1년 중위가", format_price_uk(matches["거래금액(만원)"].median()))
        summary_cols[2].metric("최근 1년 평균가", format_price_uk(matches["거래금액(만원)"].mean()))
        summary_cols[3].metric("최근거래 반영률", f"{recent_weight:.0%}")

        display = matches.head(300).copy()
        display["계약일자"] = display["계약일자"].dt.strftime("%Y-%m-%d")
        display["전용면적(㎡)"] = display["전용면적(㎡)"].map(lambda x: f"{x:,.2f}")
        display["평"] = display["평"].map(lambda x: f"{x:,.1f}")
        display["거래금액(만원)"] = display["거래금액(만원)"].map(format_price_uk)
        display["평당가(만원)"] = display["평당가(만원)"].map(lambda x: f"{x:,.0f}")
        display = display.rename(columns={"거래금액(만원)": "거래금액"})
        st.markdown(render_soft_table(display, max_rows=120), unsafe_allow_html=True)

    with st.expander("모델과 데이터 기준"):
        st.write(
            {
                "학습 행 수": f"{active_model.train_rows:,}",
                "검증 행 수": f"{active_model.test_rows:,}",
                "검증 기준": active_model.test_period,
                "전체 데이터 행 수": f"{len(prediction_df):,}",
                "데이터 기간": f"{int(prediction_df['계약년월'].min())} - {int(prediction_df['계약년월'].max())}",
                "최근 1년 기준": f"{(bundle.last_contract_month - RECENT_MONTHS + 1).strftime('%Y-%m')} - {bundle.last_contract_month.strftime('%Y-%m')}",
                "현재 사용 모델": active_model.name,
                "사용 변수": ", ".join(active_model.features),
                "단지명 사용 모델": f"MAE {bundle.with_complex.mae:,.0f}만원 / MAPE {bundle.with_complex.mape:.1%} / R² {bundle.with_complex.r2:.3f}",
                "단지명 미사용 모델": f"MAE {bundle.without_complex.mae:,.0f}만원 / MAPE {bundle.without_complex.mape:.1%} / R² {bundle.without_complex.r2:.3f}",
            }
        )
        band_display = active_model.band_metrics.copy()
        band_display["MAE_만원"] = band_display["MAE_만원"].map(lambda x: f"{x:,.0f}")
        band_display["MAPE"] = band_display["MAPE"].map(lambda x: f"{x:.1%}")
        band_display = band_display.rename(
            columns={
                "건수": "검증 건수",
                "MAE_만원": "MAE(만원)",
                "MAPE": "평균 오차율",
            }
        )
        st.caption("가격대별 시간검증 성능")
        st.markdown(render_soft_table(band_display, max_rows=20), unsafe_allow_html=True)
