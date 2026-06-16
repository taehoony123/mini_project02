from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_CSV_PATHS = [
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_23년.csv"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_24년.csv"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_25년.csv"),
    Path(r"C:\Users\Win11Pro\Downloads\아파트(매매)_실거래가_26년.csv"),
]

st.set_page_config(
    page_title="대구 아파트 실거래가 분석",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.86rem;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 750;
    }
    .market-hero {
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 18px 20px;
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        margin-bottom: 14px;
    }
    .market-title {
        font-size: 1.6rem;
        line-height: 1.25;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .market-subtitle {
        color: #64748b;
        margin-top: 6px;
        font-size: 0.95rem;
    }
    .insight-box {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 12px 14px;
        border-radius: 6px;
        color: #334155;
        line-height: 1.55;
    }
    .section-label {
        color: #334155;
        font-weight: 760;
        margin: 4px 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _find_header_line(file_obj_or_path) -> int:
    if isinstance(file_obj_or_path, (str, Path)):
        with open(file_obj_or_path, "rb") as f:
            raw = f.read()
    else:
        pos = file_obj_or_path.tell()
        raw = file_obj_or_path.read()
        file_obj_or_path.seek(pos)

    for encoding in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("cp949", errors="ignore")

    for i, line in enumerate(text.splitlines()):
        if '"NO","시군구"' in line or "NO,시군구" in line:
            return i
    return 0


@st.cache_data(show_spinner=False)
def read_raw_csv_from_path(path: str) -> pd.DataFrame:
    header_line = _find_header_line(path)
    for encoding in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=encoding, skiprows=header_line)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, encoding="cp949", skiprows=header_line, encoding_errors="ignore")
    df["자료파일"] = Path(path).name
    return df


@st.cache_data(show_spinner=False)
def load_data_from_paths(paths: tuple[str, ...]) -> pd.DataFrame:
    frames = [read_raw_csv_from_path(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    return prepare_data(df)


def read_raw_csv_from_upload(uploaded_file) -> pd.DataFrame:
    header_line = _find_header_line(uploaded_file)
    uploaded_file.seek(0)
    for encoding in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=header_line)
            break
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            continue
    else:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp949", skiprows=header_line, encoding_errors="ignore")
    df["자료파일"] = uploaded_file.name
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    money = df["거래금액(만원)"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["가격_만원"] = pd.to_numeric(money, errors="coerce")
    df["전용면적"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")
    df["층"] = pd.to_numeric(df["층"], errors="coerce")
    df["건축년도"] = pd.to_numeric(df["건축년도"], errors="coerce")
    df["계약년월"] = df["계약년월"].astype(str).str.extract(r"(\d{6})")[0]
    df["계약연도"] = pd.to_numeric(df["계약년월"].str[:4], errors="coerce")
    df["계약월"] = pd.to_numeric(df["계약년월"].str[4:6], errors="coerce")
    df["계약월일자"] = pd.to_datetime(df["계약년월"] + "01", format="%Y%m%d", errors="coerce")
    df["계약월표시"] = df["계약월일자"].dt.strftime("%Y-%m")

    address = df["시군구"].astype(str).str.split()
    df["구군"] = address.str[1]
    df["읍면동"] = address.apply(lambda x: " ".join(x[2:]) if len(x) > 2 else "")
    df["연식"] = df["계약연도"] - df["건축년도"]
    df["㎡당가격_만원"] = df["가격_만원"] / df["전용면적"]
    df["평수"] = df["전용면적"] / 3.305785
    df["평당가격_만원"] = df["가격_만원"] / df["평수"]
    df["가격_억원"] = df["가격_만원"] / 10000

    canceled = df["해제사유발생일"].notna() & (df["해제사유발생일"].astype(str).str.strip() != "-")
    df["해제거래"] = canceled

    df = df[
        df["가격_만원"].gt(0)
        & df["전용면적"].gt(0)
        & df["층"].notna()
        & df["건축년도"].notna()
        & df["계약월"].between(1, 12)
    ].copy()

    df["면적구간"] = pd.cut(
        df["전용면적"],
        bins=[0, 40, 60, 85, 102, 135, np.inf],
        labels=["~40㎡", "40~60㎡", "60~85㎡", "85~102㎡", "102~135㎡", "135㎡~"],
        right=False,
    )
    df["층구간"] = pd.cut(
        df["층"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["1~5층", "6~10층", "11~20층", "21층~"],
        right=True,
    )
    df["연식구간"] = pd.cut(
        df["연식"],
        bins=[-np.inf, 2, 5, 10, 20, 30, np.inf],
        labels=["0~2년", "3~5년", "6~10년", "11~20년", "21~30년", "31년~"],
        right=True,
    )
    return df


def ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "가격_만원" not in df.columns and "거래금액(만원)" in df.columns:
        money = df["거래금액(만원)"].astype(str).str.replace(",", "", regex=False).str.strip()
        df["가격_만원"] = pd.to_numeric(money, errors="coerce")
    if "전용면적" not in df.columns and "전용면적(㎡)" in df.columns:
        df["전용면적"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")
    if "㎡당가격_만원" not in df.columns and {"가격_만원", "전용면적"}.issubset(df.columns):
        df["㎡당가격_만원"] = df["가격_만원"] / df["전용면적"]
    if "평수" not in df.columns and "전용면적" in df.columns:
        df["평수"] = df["전용면적"] / 3.305785
    if "평당가격_만원" not in df.columns and {"가격_만원", "평수"}.issubset(df.columns):
        df["평당가격_만원"] = df["가격_만원"] / df["평수"]
    return df


def corr_ratio(categories: pd.Series, values: pd.Series) -> float:
    tmp = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if tmp.empty:
        return np.nan
    overall_mean = tmp["val"].mean()
    ss_between = tmp.groupby("cat", observed=True)["val"].agg(lambda x: len(x) * (x.mean() - overall_mean) ** 2).sum()
    ss_total = ((tmp["val"] - overall_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total else np.nan


def metric_card(label: str, value: str, help_text: str | None = None, delta: str | None = None):
    st.metric(label, value, delta=delta, help=help_text)


def format_manwon(v: float) -> str:
    return f"{v:,.0f}만원"


def format_pyeong(v: float) -> str:
    return f"{v:,.0f}만원/평"


def format_signed_price(v: float, formatter) -> str:
    prefix = "+" if v > 0 else ""
    return f"{prefix}{formatter(v)}"


def format_eok(v: float) -> str:
    return f"{v / 10000:,.2f}억"


DAEGU_DISTRICT_POSITIONS = {
    "군위군": (8.25, 8.45),
    "북구": (4.15, 6.05),
    "동구": (7.05, 5.45),
    "서구": (2.85, 4.6),
    "중구": (4.05, 4.3),
    "수성구": (5.9, 3.05),
    "남구": (4.15, 3.25),
    "달서구": (2.55, 3.05),
    "달성군": (2.65, 1.05),
}

DAEGU_DISTRICT_POLYGONS = {
    "군위군": [
        (7.05, 7.55), (8.05, 7.25), (9.05, 7.55), (9.45, 8.35),
        (9.15, 9.25), (8.25, 9.55), (7.35, 9.15), (6.95, 8.25),
    ],
    "북구": [
        (2.55, 5.15), (3.15, 6.35), (4.15, 7.25), (5.35, 7.05),
        (6.05, 6.25), (5.65, 5.25), (4.75, 4.9), (3.65, 5.0),
    ],
    "동구": [
        (5.35, 7.05), (6.85, 6.85), (8.25, 6.25), (8.9, 5.25),
        (8.35, 4.05), (7.35, 3.35), (6.3, 3.5), (5.25, 4.2),
        (4.75, 4.9), (5.65, 5.25), (6.05, 6.25),
    ],
    "서구": [
        (2.0, 4.05), (2.55, 5.15), (3.65, 5.0), (3.85, 4.35),
        (3.55, 3.65), (2.55, 3.45),
    ],
    "중구": [
        (3.85, 4.35), (4.55, 4.6), (5.0, 4.25), (4.75, 3.75),
        (4.05, 3.65), (3.55, 3.65),
    ],
    "수성구": [
        (4.75, 3.75), (5.25, 4.2), (6.3, 3.5), (7.35, 3.35),
        (7.0, 2.25), (6.15, 1.55), (5.0, 1.75), (4.45, 2.55),
        (4.05, 3.65),
    ],
    "남구": [
        (3.55, 3.65), (4.05, 3.65), (4.45, 2.55), (3.95, 2.2),
        (3.15, 2.45), (2.9, 3.2),
    ],
    "달서구": [
        (1.35, 2.35), (2.0, 4.05), (2.55, 3.45), (2.9, 3.2),
        (3.15, 2.45), (2.85, 1.65), (1.85, 1.55), (1.15, 1.95),
    ],
    "달성군": [
        (0.65, 0.75), (1.85, 1.55), (2.85, 1.65), (3.95, 2.2),
        (4.85, 1.45), (4.15, 0.55), (2.75, 0.25), (1.25, 0.35),
    ],
}


def polygon_path(points: list[tuple[float, float]]) -> str:
    start = points[0]
    rest = " ".join(f"L {x},{y}" for x, y in points[1:])
    return f"M {start[0]},{start[1]} {rest} Z"


def make_district_map(district_data: pd.DataFrame, selected_region: str) -> go.Figure:
    map_df = district_data.copy()
    map_df[["x", "y"]] = map_df["구군"].apply(
        lambda name: pd.Series(DAEGU_DISTRICT_POSITIONS.get(name, (np.nan, np.nan)))
    )
    map_df = map_df.dropna(subset=["x", "y"]).copy()
    map_df["선택"] = np.where(map_df["구군"] == selected_region, "선택 지역", "다른 지역")
    map_df["라벨"] = map_df.apply(lambda r: f"{r['구군']}<br>{r['시세지수']:.0f}", axis=1)
    color_low = max(60, map_df["시세지수"].quantile(0.05))
    color_high = min(180, map_df["시세지수"].quantile(0.95))
    if color_low == color_high:
        color_low -= 1
        color_high += 1

    fig = go.Figure()
    index_by_district = map_df.set_index("구군")["시세지수"].to_dict()
    row_by_district = map_df.set_index("구군").to_dict("index")
    for district, points in DAEGU_DISTRICT_POLYGONS.items():
        if district not in index_by_district:
            continue
        value = index_by_district[district]
        norm = float(np.clip((value - color_low) / (color_high - color_low), 0, 1))
        fill = px.colors.sample_colorscale("RdYlBu_r", norm)[0]
        is_selected = district == selected_region
        x_values = [p[0] for p in points] + [points[0][0]]
        y_values = [p[1] for p in points] + [points[0][1]]
        info = row_by_district[district]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color="#0f172a" if is_selected else "#ffffff", width=3 if is_selected else 1.6),
                opacity=0.92 if is_selected else 0.78,
                name=district,
                customdata=[[district]] * len(x_values),
                hovertemplate=(
                    f"<b>{district}</b><br>"
                    f"시세지수 {info['시세지수']:.1f}<br>"
                    f"중앙값 {info['median_price']:,.0f}만원<br>"
                    f"평당 {info['pyeong_median']:,.0f}만원<br>"
                    f"거래 {info['volume']:,.0f}건"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    marker_sizes = 16 + (map_df["volume"] / map_df["volume"].max()).pow(0.55) * 22
    fig.add_trace(
        go.Scatter(
            x=map_df["x"],
            y=map_df["y"],
            mode="markers+text",
            marker=dict(
                size=marker_sizes,
                color=map_df["시세지수"],
                colorscale="RdYlBu_r",
                cmin=color_low,
                cmax=color_high,
                colorbar=dict(title="시세<br>지수", thickness=12, len=0.68),
                line=dict(width=2, color="#ffffff"),
                opacity=0.96,
            ),
            text=map_df["라벨"],
            textposition="middle center",
            textfont=dict(size=12, color="#0f172a"),
            customdata=map_df[["구군"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "시세지수 %{marker.color:.1f}<br>"
                "거래량은 원 크기로 표시<extra></extra>"
            ),
            selected=dict(marker=dict(size=52, color="#f97316", opacity=1)),
            unselected=dict(marker=dict(opacity=0.9)),
            showlegend=False,
        )
    )
    if selected_region != "대구 전체" and selected_region in set(map_df["구군"]):
        selected = map_df[map_df["구군"] == selected_region].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[selected["x"]],
                y=[selected["y"]],
                mode="markers",
                marker=dict(size=58, color="rgba(0,0,0,0)", line=dict(width=4, color="#111827")),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(visible=False, range=[0.25, 9.7], fixedrange=True),
        yaxis=dict(visible=False, range=[0.0, 9.8], fixedrange=True, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f1f5f9",
        paper_bgcolor="#ffffff",
        dragmode="select",
    )
    fig.add_shape(type="rect", x0=0.25, x1=9.7, y0=0.0, y1=9.8, line=dict(color="#dbe4ee"), fillcolor="#f8fafc", layer="below")
    fig.add_annotation(x=6.2, y=7.35, text="군위군은 대구 북쪽 분리 지역", showarrow=False, font=dict(size=11, color="#64748b"))
    fig.add_annotation(x=5.25, y=0.85, text="달성군", showarrow=False, font=dict(size=11, color="#64748b"))
    return fig


st.markdown(
    """
    <div class="market-hero">
        <p class="market-title">대구 아파트 매매 시세판</p>
        <div class="market-subtitle">
            실거래가를 지역별 시세, 월별 거래 흐름, 면적·연식·층수 영향으로 나눠 직관적으로 비교합니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("데이터")
    uploaded = st.file_uploader("국토교통부 실거래가 CSV 업로드", type=["csv"], accept_multiple_files=True)
    include_canceled = st.toggle("해제 거래 포함", value=False)
    price_basis = st.radio("가격 기준", ["평당 가격", "총 매매가격", "㎡당 가격"], horizontal=False)

try:
    if uploaded:
        raw_parts = [read_raw_csv_from_upload(file) for file in uploaded]
        raw_df = prepare_data(pd.concat(raw_parts, ignore_index=True))
        source_label = ", ".join(file.name for file in uploaded)
    else:
        existing_paths = tuple(str(path) for path in DEFAULT_CSV_PATHS if path.exists())
        if not existing_paths:
            st.error("기본 CSV 파일을 찾지 못했습니다. 사이드바에서 CSV를 업로드해 주세요.")
            st.stop()
        raw_df = load_data_from_paths(existing_paths)
        source_label = f"기본 병합 데이터 {len(existing_paths)}개 파일"
except Exception as exc:
    st.error(f"CSV를 읽지 못했습니다: {exc}")
    st.stop()

df = ensure_price_columns(raw_df)
if not include_canceled:
    df = df[~df["해제거래"]].copy()

with st.sidebar:
    st.divider()
    st.header("필터")
    years = sorted(df["계약연도"].dropna().astype(int).unique())
    selected_years = st.multiselect("계약연도", years, default=years)
    districts = sorted(df["구군"].dropna().unique())
    selected_districts = st.multiselect("시군구", districts, default=districts)
    months = sorted(df["계약월"].dropna().astype(int).unique())
    selected_months = st.multiselect("계약월", months, default=months)
    area_options = [str(x) for x in df["면적구간"].dropna().cat.categories]
    selected_areas = st.multiselect("면적구간", area_options, default=area_options)
    age_options = [str(x) for x in df["연식구간"].dropna().cat.categories]
    selected_ages = st.multiselect("연식구간", age_options, default=age_options)

filtered = df[
    df["계약연도"].isin(selected_years)
    & df["구군"].isin(selected_districts)
    & df["계약월"].isin(selected_months)
    & df["면적구간"].astype(str).isin(selected_areas)
    & df["연식구간"].astype(str).isin(selected_ages)
].copy()
filtered = ensure_price_columns(filtered)

if filtered.empty:
    st.warning("선택한 필터에 해당하는 거래가 없습니다.")
    st.stop()

if price_basis == "총 매매가격":
    target_col = "가격_만원"
    target_label = "총 매매가격(만원)"
    format_price = format_manwon
elif price_basis == "㎡당 가격":
    target_col = "㎡당가격_만원"
    target_label = "㎡당 가격(만원/㎡)"
    format_price = lambda v: f"{v:,.1f}만원/㎡"
else:
    target_col = "평당가격_만원"
    target_label = "평당 가격(만원/평)"
    format_price = format_pyeong

numeric_cols = {
    "전용면적": "전용면적",
    "층": "층",
    "건축년도": "건축년도",
    "연식": "연식",
    "계약월": "계약월",
}
corr_rows = []
for label, col in numeric_cols.items():
    corr_rows.append(
        {
            "변수": label,
            "Pearson": filtered[[col, target_col]].corr(method="pearson").iloc[0, 1],
            "Spearman": filtered[[col, target_col]].corr(method="spearman").iloc[0, 1],
        }
    )
for label, col in {"시군구": "구군", "면적구간": "면적구간", "층구간": "층구간", "연식구간": "연식구간"}.items():
    corr_rows.append({"변수": label, "Pearson": np.nan, "Spearman": corr_ratio(filtered[col], filtered[target_col])})
corr_df = pd.DataFrame(corr_rows)

overall_median = filtered[target_col].median()
district_league = (
    filtered.groupby("구군", observed=True)
    .agg(
        volume=("가격_만원", "size"),
        median_price=(target_col, "median"),
        pyeong_median=("평당가격_만원", "median"),
        ppm_median=("㎡당가격_만원", "median"),
        area_median=("전용면적", "median"),
        new_share=("연식", lambda x: (x <= 5).mean() * 100),
    )
    .reset_index()
    .sort_values("median_price", ascending=False)
)
district_league["시세지수"] = district_league["median_price"] / overall_median * 100
district_league["순위"] = np.arange(1, len(district_league) + 1)
best_region = district_league.iloc[0]

trend_by_region = (
    filtered.groupby(["구군", "계약월일자"], observed=True)[target_col]
    .median()
    .reset_index()
    .sort_values(["구군", "계약월일자"])
)
trend_change = (
    trend_by_region.groupby("구군", observed=True)
    .agg(first_month=("계약월일자", "first"), last_month=("계약월일자", "last"), first_price=(target_col, "first"), last_price=(target_col, "last"))
    .reset_index()
)
trend_change["기간변동"] = trend_change["last_price"] - trend_change["first_price"]
district_league = district_league.merge(trend_change[["구군", "기간변동"]], on="구군", how="left")

focus_options = ["대구 전체"] + sorted(filtered["구군"].dropna().unique())
if "focus_region" not in st.session_state or st.session_state.focus_region not in focus_options:
    st.session_state.focus_region = "대구 전체"

top_left, top_right = st.columns([1.1, 2.7])
with top_left:
    st.selectbox("관심 지역", focus_options, key="focus_region")
with top_right:
    st.caption(f"자료: {source_label}")

map_col, guide_col = st.columns([1.65, 1])
with map_col:
    st.markdown('<div class="section-label">지도에서 지역 선택</div>', unsafe_allow_html=True)
    district_map = make_district_map(district_league, st.session_state.focus_region)
    map_event = st.plotly_chart(
        district_map,
        use_container_width=True,
        key="district_map",
        on_select="rerun",
        selection_mode="points",
    )
    try:
        selected_points = map_event["selection"]["points"]
    except (KeyError, TypeError):
        selected_points = []
    if selected_points:
        clicked_region = selected_points[0].get("customdata", [None])[0]
        if clicked_region in focus_options and clicked_region != st.session_state.focus_region:
            st.session_state.focus_region = clicked_region
            st.rerun()

with guide_col:
    st.markdown('<div class="section-label">읽는 법</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-box">
            원 크기는 거래량, 색은 대구 전체 대비 시세지수입니다.<br>
            <b>100</b>보다 높으면 대구 중앙값보다 비싼 지역이고,
            점을 클릭하면 해당 지역의 월별 시세와 거래량이 아래에 표시됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

focus_region = st.session_state.focus_region
focus_df = filtered if focus_region == "대구 전체" else filtered[filtered["구군"] == focus_region].copy()
focus_median = focus_df[target_col].median()
price_index = focus_median / overall_median * 100 if overall_median else np.nan

monthly = (
    focus_df.groupby("계약월일자", observed=True)
    .agg(volume=("가격_만원", "size"), median_price=(target_col, "median"), avg_price=(target_col, "mean"))
    .reset_index()
    .sort_values("계약월일자")
)
latest = monthly.iloc[-1]
previous = monthly.iloc[-2] if len(monthly) > 1 else latest
median_delta = latest["median_price"] - previous["median_price"]

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("선택 거래", f"{len(focus_df):,}건")
with m2:
    metric_card("중앙 시세", format_price(focus_median), format_eok(focus_median) if price_basis == "총 매매가격" else None)
with m3:
    metric_card("대구 대비 지수", f"{price_index:,.1f}", "대구 전체 중앙값을 100으로 둔 상대 가격")
with m4:
    metric_card(f"{latest['계약월일자']:%Y-%m} 중앙값", format_price(latest["median_price"]), delta=format_signed_price(median_delta, format_price))
with m5:
    metric_card("최상위 지역", str(best_region["구군"]), f"중앙값 {format_price(best_region['median_price'])}")

st.markdown(
    f"""
    <div class="insight-box">
        <b>{focus_region}</b> 기준 중앙값은 <b>{format_price(focus_median)}</b>입니다.
        대구 전체를 100으로 보면 시세지수는 <b>{price_index:,.1f}</b>이고,
        현재 필터에서 가장 높은 지역은 <b>{best_region['구군']}</b>입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

board_left, board_right = st.columns([1.75, 1])
with board_left:
    st.markdown('<div class="section-label">월별 매매 시세와 거래량</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["계약월일자"],
            y=monthly["volume"],
            name="거래량",
            marker_color="rgba(37, 99, 235, 0.22)",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["계약월일자"],
            y=monthly["median_price"],
            name="중앙 시세",
            mode="lines+markers+text",
            text=monthly["median_price"].map(format_price),
            textposition="top center",
            line=dict(color="#1d4ed8", width=3),
            marker=dict(size=8, color="#1d4ed8"),
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=26, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title="계약연월", tickformat="%Y-%m", gridcolor="#eef2f7"),
        yaxis=dict(title=target_label, gridcolor="#eef2f7"),
        yaxis2=dict(title="거래량", overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

with board_right:
    st.markdown('<div class="section-label">지역 시세 순위</div>', unsafe_allow_html=True)
    league_view = district_league[["순위", "구군", "시세지수", "median_price", "pyeong_median", "volume", "기간변동"]].rename(
        columns={
            "median_price": "중앙값",
            "pyeong_median": "평당중앙값",
            "volume": "거래건수",
        }
    )
    st.dataframe(
        league_view.style.format(
            {
                "시세지수": "{:,.1f}",
                "중앙값": "{:,.0f}",
                "평당중앙값": "{:,.0f}",
                "기간변동": "{:+,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

st.divider()

left, right = st.columns([1.05, 1.4])
with left:
    st.subheader("가격 영향 요인")
    st.dataframe(
        corr_df.style.format({"Pearson": "{:.3f}", "Spearman": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("범주형 변수는 Spearman 칸에 correlation ratio η를 표시했습니다.")

with right:
    plot_df = corr_df.copy()
    plot_df["상관강도"] = plot_df["Pearson"].fillna(plot_df["Spearman"]).abs()
    fig = px.bar(
        plot_df.sort_values("상관강도"),
        x="상관강도",
        y="변수",
        orientation="h",
        text=plot_df.sort_values("상관강도")["상관강도"].map(lambda x: f"{x:.3f}"),
        color="변수",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, xaxis_title="절대 상관 강도", yaxis_title=None, height=330, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

tab_region, tab_area, tab_month, tab_floor_age, tab_scatter, tab_table = st.tabs(
    ["지역", "면적", "계약월", "층수/연식", "산점도", "상세표"]
)

with tab_region:
    region = (
        filtered.groupby("구군", observed=True)
        .agg(
            거래건수=("가격_만원", "size"),
            중앙값=(target_col, "median"),
            평균=(target_col, "mean"),
            pyeong_median=("평당가격_만원", "median"),
        )
        .reset_index()
        .rename(columns={"pyeong_median": "평당가격중앙값"})
        .sort_values("중앙값", ascending=False)
    )
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.bar(
            region,
            x="구군",
            y="중앙값",
            color="구군",
            text=region["중앙값"].map(lambda x: f"{x:,.0f}"),
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(showlegend=False, yaxis_title=target_label, xaxis_title=None, height=440)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(
            region.style.format({"중앙값": "{:,.0f}", "평균": "{:,.0f}", "평당가격중앙값": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    heat = (
        filtered.groupby(["구군", "면적구간"], observed=True)[target_col]
        .median()
        .reset_index()
        .pivot(index="구군", columns="면적구간", values=target_col)
    )
    fig = px.imshow(
        heat,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"color": target_label},
    )
    fig.update_layout(height=460, xaxis_title="면적구간", yaxis_title="시군구")
    st.plotly_chart(fig, use_container_width=True)

with tab_area:
    area = (
        filtered.groupby("면적구간", observed=True)
        .agg(거래건수=("가격_만원", "size"), 중앙값=(target_col, "median"), 평균=(target_col, "mean"))
        .reset_index()
    )
    fig = px.bar(
        area,
        x="면적구간",
        y="중앙값",
        color="면적구간",
        text=area["중앙값"].map(lambda x: f"{x:,.0f}"),
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False, yaxis_title=target_label, xaxis_title=None, height=430)
    st.plotly_chart(fig, use_container_width=True)

    sample = filtered.sample(min(len(filtered), 6000), random_state=7)
    fig = px.scatter(
        sample,
        x="전용면적",
        y=target_col,
        color="구군",
        hover_data=["단지명", "구군", "읍면동", "층", "건축년도", "계약월"],
        opacity=0.45,
        trendline="ols",
    )
    fig.update_layout(yaxis_title=target_label, xaxis_title="전용면적(㎡)", height=520)
    st.plotly_chart(fig, use_container_width=True)

with tab_month:
    month = (
        filtered.groupby("계약월일자", observed=True)
        .agg(거래건수=("가격_만원", "size"), 중앙값=(target_col, "median"), 평균=(target_col, "mean"))
        .reset_index()
        .sort_values("계약월일자")
    )
    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = px.line(month, x="계약월일자", y="중앙값", markers=True, text=month["중앙값"].map(lambda x: f"{x:,.0f}"))
        fig.update_traces(textposition="top center")
        fig.update_layout(yaxis_title=target_label, xaxis_title="계약연월", xaxis_tickformat="%Y-%m", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(month, x="계약월일자", y="거래건수", color="거래건수", color_continuous_scale="Teal")
        fig.update_layout(showlegend=False, xaxis_title="계약연월", xaxis_tickformat="%Y-%m", yaxis_title="거래건수", height=430)
        st.plotly_chart(fig, use_container_width=True)

with tab_floor_age:
    c1, c2 = st.columns(2)
    with c1:
        floor = (
            filtered.groupby("층구간", observed=True)
            .agg(거래건수=("가격_만원", "size"), 중앙값=(target_col, "median"))
            .reset_index()
        )
        fig = px.bar(
            floor,
            x="층구간",
            y="중앙값",
            color="층구간",
            text=floor["중앙값"].map(lambda x: f"{x:,.0f}"),
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(showlegend=False, yaxis_title=target_label, xaxis_title=None, height=430)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        age = (
            filtered.groupby("연식구간", observed=True)
            .agg(거래건수=("가격_만원", "size"), 중앙값=(target_col, "median"))
            .reset_index()
        )
        fig = px.bar(
            age,
            x="연식구간",
            y="중앙값",
            color="연식구간",
            text=age["중앙값"].map(lambda x: f"{x:,.0f}"),
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(showlegend=False, yaxis_title=target_label, xaxis_title=None, height=430)
        st.plotly_chart(fig, use_container_width=True)

    box_sample = filtered.sample(min(len(filtered), 8000), random_state=11)
    fig = px.box(box_sample, x="연식구간", y=target_col, color="연식구간", points=False)
    fig.update_layout(showlegend=False, yaxis_title=target_label, xaxis_title="연식구간", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab_scatter:
    sample = filtered.sample(min(len(filtered), 8000), random_state=3)
    x_axis = st.selectbox("X축", ["전용면적", "층", "건축년도", "연식", "계약월"], index=0)
    fig = px.scatter(
        sample,
        x=x_axis,
        y=target_col,
        color="구군",
        size="전용면적",
        hover_data=["단지명", "시군구", "가격_만원", "평당가격_만원", "㎡당가격_만원", "층", "건축년도", "계약월"],
        opacity=0.5,
        trendline="ols",
    )
    fig.update_layout(yaxis_title=target_label, height=620)
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    cols = [
        "시군구",
        "단지명",
        "전용면적",
        "면적구간",
        "계약년월",
        "계약일",
        "가격_만원",
        "평당가격_만원",
        "㎡당가격_만원",
        "층",
        "건축년도",
        "연식",
        "거래유형",
    ]
    existing_cols = [col for col in cols if col in filtered.columns]
    show = filtered[existing_cols].sort_values("가격_만원", ascending=False)
    st.dataframe(
        show,
        column_config={
            "가격_만원": st.column_config.NumberColumn("가격_만원", format="%d"),
            "평당가격_만원": st.column_config.NumberColumn("평당가격_만원", format="%d"),
            "㎡당가격_만원": st.column_config.NumberColumn("㎡당가격_만원", format="%.1f"),
            "전용면적": st.column_config.NumberColumn("전용면적", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=560,
    )
