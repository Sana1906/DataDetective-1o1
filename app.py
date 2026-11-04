"""
DataDetective — High-Tech Corporate Autonomous Data Investigation Platform
===========================================================================
Retains all original 10 tabs and core logic, upgraded with an elite corporate 
dark-mode design system, high-visibility typography, glassmorphism cards, and sleek charts.
"""

import io
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re as _re

# Streamlit HTML normalization
_orig_markdown = st.markdown

def _markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = _re.sub(r"(?m)^[ 	]+", "", body)
        body = _re.sub(r"\s* ", " ", body)
    return _orig_markdown(body, *args, **kwargs)

st.markdown = _markdown
import extensions as ext
import privacy
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ----------------------------------------------------------------------
# Page config & High-Tech Corporate Dark Theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DataDetective Enterprise",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Dark Background & High-Contrast Typography */
    html, body, [class*="css"], .stApp {
        background-color: #090d16 !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Force high visibility for all text elements */
    p, span, label, h1, h2, h3, h4, h5, h6, div, caption, .stMarkdown {
        color: #f8fafc !important;
    }

    .sub-text, .section-caption {
        color: #94a3b8 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #05080f !important;
        border-right: 1px solid #1e293b !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    .brand {
        padding: 6px 8px 20px 8px;
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #f8fafc !important;
    }

    .brand-name span {
        color: #38bdf8 !important;
    }

    .brand-tagline {
        color: #94a3b8 !important;
        font-size: 11px;
        margin-top: 4px;
        line-height: 1.4;
    }

    .sidebar-label {
        color: #38bdf8 !important;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 18px 8px 8px;
        text-transform: uppercase;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .header-title {
        font-size: 25px;
        font-weight: 800;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #94a3b8 !important;
        font-size: 12px;
        margin-top: 4px;
    }

    .secure-pill {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 7px 14px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* KPI cards / Glass Cards */
    .kpi-card, .health-panel {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
    }

    .kpi-label {
        color: #94a3b8 !important;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .4px;
    }

    .kpi-value, .big-score {
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8 !important;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }
    
    .big-score {
        font-size: 44px !important;
    }

    .kpi-note {
        color: #34d399 !important;
        font-size: 11px;
        margin-top: 5px;
        font-weight: 600;
    }

    /* Section headers */
    .section-title {
        color: #f8fafc !important;
        font-size: 19px;
        font-weight: 750;
        margin: 22px 0 10px;
    }

    .section-caption {
        color: #94a3b8 !important;
        font-size: 12px;
        margin-bottom: 14px;
    }

    /* Welcome / empty state */
    .welcome {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 30px;
        color: #f8fafc;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
    }

    .welcome h1 {
        margin: 0 0 8px 0;
        font-size: 30px;
        letter-spacing: -0.7px;
        color: #f8fafc !important;
    }

    .welcome p {
        color: #94a3b8 !important;
        font-size: 13px;
        max-width: 720px;
        line-height: 1.6;
    }

    .workflow {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 22px;
    }

    .workflow-item {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 11px;
        padding: 14px;
    }

    .workflow-number {
        color: #38bdf8 !important;
        font-weight: 800;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
    }

    .workflow-name {
        font-size: 13px;
        font-weight: 700;
        margin-top: 5px;
        color: #f8fafc !important;
    }

    /* Discovery & Anomaly Boxes */
    .discovery-box {
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38bdf8;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #0f172a;
        border-radius: 10px;
        color: #f8fafc !important;
    }

    .anomaly-box {
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(30, 27, 75, 0.5);
        border-radius: 10px;
        color: #f8fafc !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0f172a;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1e293b;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0f172a !important;
        border: 1px dashed #334155 !important;
    }

    /* Tabs Styling - High Visibility Contrast */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #94a3b8 !important;
        background-color: #0f172a !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        border: 1px solid #1e293b !important;
        margin-right: 4px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #090d16 !important;
        background-color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        border-color: #7dd3fc !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
        background-color: #0f172a;
    }

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    /* Reset workspace button custom color */
    section[data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #a78bfa !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background: linear-gradient(135deg, #6d28d9 100%, #5b21b6 100%) !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        padding: 30px 0 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "report_log" not in st.session_state:
    st.session_state.report_log = []


def log(section: str, text: str):
    """Append a line to the report log so the final report can be built."""
    st.session_state.report_log.append((section, text))


# ----------------------------------------------------------------------
# ---- STAGE 1: Dataset Understanding ----
# ----------------------------------------------------------------------
def _is_identifier_like(s: pd.Series) -> bool:
    n = int(s.notna().sum())
    if n < 20 or pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    return s.nunique(dropna=True) / n >= 0.95


def understand_dataset(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    id_cols = [c for c in df.columns if _is_identifier_like(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    for c in categorical_cols[:]:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) and sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.7:
                datetime_cols.append(c)
                categorical_cols.remove(c)

    target_candidates = [
        c for c in categorical_cols if 1 < df[c].nunique() <= 20
    ] + numeric_cols

    return {
        "rows": df.shape[0],
        "id_cols": id_cols,
        "cols": df.shape[1],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "target_candidates": target_candidates,
    }


# ----------------------------------------------------------------------
# ---- STAGE 2: Data Quality / Health Score ----
# ----------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> dict:
    n = len(df)
    missing = df.isna().sum()
    missing_pct = (missing.sum() / (df.shape[0] * df.shape[1]) * 100) if n else 0
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / n * 100) if n else 0

    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_counts = {}
    total_outliers = 0
    for c in numeric_cols:
        col = df[c].dropna()
        if len(col) < 5:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cnt = ((col < lower) | (col > upper)).sum()
        if cnt > 0:
            outlier_counts[c] = int(cnt)
            total_outliers += cnt

    high_corr_pairs = []
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and val > 0.9:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

    imbalanced_cols = []
    for c in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[c].value_counts(normalize=True)
        if 1 < len(vc) <= 15 and vc.iloc[0] > 0.85:
            imbalanced_cols.append((c, round(vc.iloc[0] * 100, 1)))

    score = 100
    score -= min(missing_pct * 1.5, 30)
    score -= min(duplicate_pct * 1.5, 15)
    outlier_pct = (total_outliers / (n * max(len(numeric_cols), 1)) * 100) if n else 0
    score -= min(outlier_pct * 2, 20)
    score -= min(len(high_corr_pairs) * 3, 15)
    score -= min(len(imbalanced_cols) * 5, 20)
    score = max(0, round(score))

    if score >= 85:
        verdict = "Optimal — Ready for modeling"
        color = "#10b981"
    elif score >= 65:
        verdict = "Good — Minor cleanup recommended"
        color = "#f59e0b"
    elif score >= 45:
        verdict = "Fair — Cleaning strongly recommended"
        color = "#f97316"
    else:
        verdict = "Poor — Significant cleaning required"
        color = "#ef4444"

    return {
        "score": score,
        "verdict": verdict,
        "color": color,
        "missing": missing[missing > 0].sort_values(ascending=False),
        "missing_pct": round(missing_pct, 2),
        "duplicates": int(duplicates),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_counts": outlier_counts,
        "total_outliers": int(total_outliers),
        "high_corr_pairs": high_corr_pairs,
        "imbalanced_cols": imbalanced_cols,
    }


# ----------------------------------------------------------------------
# ---- Privacy / Security Scan ----
# ----------------------------------------------------------------------
def privacy_scan(df: pd.DataFrame) -> pd.DataFrame:
    return privacy.scan_compat(df)


# ----------------------------------------------------------------------
# ---- STAGE 3: Automated, context-aware EDA ----
# ----------------------------------------------------------------------
def _correlation_frame(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    usable_num = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    data = df[usable_num].copy() if usable_num else pd.DataFrame(index=df.index)
    for c in categorical_cols:
        vals = sorted(df[c].dropna().astype(str).unique())
        if len(vals) == 2:
            coded = df[c].astype(str).where(df[c].notna()).map({vals[0]: 0, vals[1]: 1})
            data[f"{c} ({vals[1]}=1)"] = coded
    return data


def _strength(r: float) -> str:
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "weak"
    if r < 0.5:
        return "moderate"
    if r < 0.7:
        return "strong"
    return "very strong"


def _safe_sample(df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=42)


def render_eda(df: pd.DataFrame, info: dict):
    id_cols = set(info.get("id_cols", []))
    numeric_cols = [c for c in info["numeric_cols"] if c not in id_cols]
    all_cat_cols = [c for c in info["categorical_cols"] if c not in id_cols]
    low_card_cats = [c for c in all_cat_cols if 1 < df[c].nunique(dropna=True) <= 25]
    high_card_cats = [c for c in all_cat_cols if df[c].nunique(dropna=True) > 25]
    datetime_cols = info.get("datetime_cols", [])

    st.markdown("#### Column profile")
    st.caption("Quick inventory of every column: type, missingness, uniqueness and basic statistics.")

    profile_rows = []
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        missing_pct = 100.0 * n_missing / len(df) if len(df) else 0
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        is_id = c in id_cols
        sample_vals = s.dropna().astype(str).head(3).tolist() if nunique > 0 else []
        sample_str = ", ".join(sample_vals) if sample_vals else "—"

        row = {
            "Column": c,
            "Type": dtype,
            "Missing %": round(missing_pct, 1),
            "Unique": nunique,
            "Identifier?": "Yes" if is_id else "",
            "Sample values": sample_str[:80] + ("…" if len(sample_str) > 80 else ""),
        }
        if pd.api.types.is_numeric_dtype(s) and not is_id:
            row["Mean"] = round(float(s.mean()), 3) if s.notna().any() else None
            row["Std"] = round(float(s.std()), 3) if s.notna().sum() > 1 else None
            row["Min"] = round(float(s.min()), 3) if s.notna().any() else None
            row["Max"] = round(float(s.max()), 3) if s.notna().any() else None
        else:
            row["Mean"] = row["Std"] = row["Min"] = row["Max"] = None
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # Missing-value overview
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if len(missing_counts) > 0:
        st.markdown("#### Missing values")
        miss_df = missing_counts.reset_index()
        miss_df.columns = ["Column", "Missing count"]
        miss_df["Missing %"] = (miss_df["Missing count"] / len(df) * 100).round(1)
        fig_miss = px.bar(
            miss_df.head(20),
            x="Missing %",
            y="Column",
            orientation="h",
            title="Columns with missing values (top 20)",
            template="plotly_dark",
            color="Missing %",
            color_continuous_scale=["#93C5FD", "#1D4ED8"],
        )
        fig_miss.update_layout(
            height=max(280, 28 * min(len(miss_df), 20) + 80),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("No missing values detected in any column.")

    # Numeric distributions
    st.markdown("#### Numeric distributions")
    if not numeric_cols:
        st.info("No usable numeric columns found for distribution plots.")
    else:
        default_num = numeric_cols[: min(4, len(numeric_cols))]
        chosen_num = st.multiselect("Select numeric columns", options=numeric_cols, default=default_num, key="eda_num_cols")
        if chosen_num:
            plot_df = _safe_sample(df)
            n_cols = 2 if len(chosen_num) > 1 else 1
            grid = st.columns(n_cols)
            for i, c in enumerate(chosen_num):
                with grid[i % n_cols]:
                    fig = px.histogram(
                        plot_df, x=c,
                        nbins=min(40, max(10, int(df[c].nunique() ** 0.5) + 5)),
                        title=f"Distribution of {c}",
                        marginal="box",
                        template="plotly_dark",
                        color_discrete_sequence=["#38bdf8"],
                    )
                    fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f8fafc")
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Categorical breakdowns
    st.markdown("#### Categorical breakdowns")
    usable_cats = low_card_cats + high_card_cats
    if not usable_cats:
        st.info("No categorical columns available for frequency charts.")
    else:
        cat_choice = st.selectbox("Categorical column", options=usable_cats, key="eda_cat_col")
        top_n = st.slider("Show top N categories", min_value=5, max_value=30, value=15, key="eda_cat_topn")

        series = df[cat_choice].fillna("(Missing)").astype(str)
        vc = series.value_counts()
        plot_data = vc.head(top_n) if len(vc) > top_n else vc
        counts = plot_data.reset_index()
        counts.columns = [cat_choice, "count"]
        counts["share"] = (counts["count"] / counts["count"].sum() * 100).round(1)

        fig = px.bar(
            counts, x=cat_choice, y="count",
            text=counts["share"].map(lambda x: f"{x:.0f}%"),
            title=f"Frequency of {cat_choice}",
            template="plotly_dark",
            color="count",
            color_continuous_scale=["#93C5FD", "#1D4ED8"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=380, showlegend=False, coloraxis_showscale=False,
            xaxis_tickangle=-30, margin=dict(l=10, r=10, t=50, b=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.markdown("#### Correlation analysis")
    corr_data = _correlation_frame(df, numeric_cols, all_cat_cols)
    if corr_data.shape[1] >= 2:
        corr = corr_data.corr(numeric_only=True)
        mask = np.triu(np.ones(corr.shape, dtype=bool))
        shown = corr.mask(mask)

        fig = px.imshow(
            shown, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto", title="Pearson correlation (lower triangle)",
            template="plotly_dark",
        )
        fig.update_layout(
            height=max(360, 55 * len(shown) + 120),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# ---- STAGE 4: Data Detective Mode ----
# ----------------------------------------------------------------------
def discover_patterns(df: pd.DataFrame, info: dict) -> list:
    discoveries = []
    numeric_cols = info["numeric_cols"]
    categorical_cols = info["categorical_cols"]

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and 0.7 <= val < 1.0:
                    pairs.append((corr.columns[i], corr.columns[j], val))
        pairs.sort(key=lambda x: -x[2])
        for a, b, v in pairs[:3]:
            discoveries.append(f"An unusually strong relationship was found between **{a}** and **{b}** (correlation = {v:.2f}).")

    if not discoveries:
        discoveries.append("No strongly unusual relationships were detected — dataset looks consistent.")
    return discoveries


def detect_anomalies(df: pd.DataFrame, info: dict):
    numeric_cols = info["numeric_cols"]
    if len(numeric_cols) < 2 or len(df) < 20:
        return None, "Not enough numeric data to run anomaly detection."

    X = df[numeric_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)

    result = df.copy()
    result["_anomaly"] = preds == -1
    result["_anomaly_score"] = scores
    n_anomalies = int(result["_anomaly"].sum())
    pct = n_anomalies / len(df) * 100
    return result, f"{n_anomalies} records ({pct:.1f}%) flagged as unusual out of {len(df)}."


# ----------------------------------------------------------------------
# ---- STAGE 5: ML Engine ----
# ----------------------------------------------------------------------
def prep_features(df: pd.DataFrame, target: str, feature_cols: list):
    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    for c in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[c] = X[c].astype(str).fillna("missing")
        X[c] = LabelEncoder().fit_transform(X[c])

    for c in X.select_dtypes(include=np.number).columns:
        X[c] = X[c].fillna(X[c].median())

    is_classification = (y.dtype == object) or (y.nunique() <= 15 and y.dtype != float)
    if is_classification:
        y_le = LabelEncoder()
        y_enc = y_le.fit_transform(y.astype(str))
        return X, y_enc, True, y_le
    return X, y.values, False, None


def check_leakage(X: pd.DataFrame, y, is_classification: bool) -> list:
    warnings = []
    try:
        if is_classification:
            for c in X.columns:
                if X[c].nunique() == len(X):
                    warnings.append(f"**{c}** has unique value per row — potential identifier leakage.")
        else:
            corr = X.assign(_target=y).corr(numeric_only=True)["_target"].drop("_target", errors="ignore")
            for c, v in corr.items():
                if pd.notna(v) and abs(v) > 0.98:
                    warnings.append(f"**{c}** correlates with target at {v:.3f} — check for target leakage.")
    except Exception:
        pass
    return warnings


def run_ml_lab(df: pd.DataFrame, info: dict):
    st.subheader("🎯 Choose a target column")
    target = st.selectbox("What should the model predict?", info["target_candidates"])
    feature_cols = [c for c in df.columns if c != target]
    feature_cols = st.multiselect("Feature columns to use", feature_cols, default=feature_cols[: min(10, len(feature_cols))])

    if not feature_cols:
        st.warning("Select at least one feature column.")
        return

    if st.button("🚀 Run automated model comparison", type="primary"):
        with st.spinner("Training and comparing models..."):
            X, y, is_classification, y_encoder = prep_features(df, target, feature_cols)
            if len(X) < 20:
                st.error("Not enough rows with valid target value.")
                return

            leakage_warnings = check_leakage(X, y, is_classification)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            results = []
            models = {}

            if is_classification:
                candidates = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                }
                if HAS_XGB and len(np.unique(y)) > 1:
                    candidates["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
                for name, model in candidates.items():
                    try:
                        use_scaled = name == "Logistic Regression"
                        model.fit(X_train_s if use_scaled else X_train, y_train)
                        preds = model.predict(X_test_s if use_scaled else X_test)
                        avg = "binary" if len(np.unique(y)) == 2 else "weighted"
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, average=avg, zero_division=0),
                            "Recall": recall_score(y_test, preds, average=avg, zero_division=0),
                            "F1": f1_score(y_test, preds, average=avg, zero_division=0),
                        })
                        models[name] = model
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")
                metric_for_best = "F1"
            else:
                candidates = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
                    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
                }
                if HAS_XGB:
                    candidates["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
                for name, model in candidates.items():
                    try:
                        use_scaled = name == "Linear Regression"
                        model.fit(X_train_s if use_scaled else X_train, y_train)
                        preds = model.predict(X_test_s if use_scaled else X_test)
                        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                        results.append({
                            "Model": name,
                            "R2": r2_score(y_test, preds),
                            "MAE": mean_absolute_error(y_test, preds),
                            "RMSE": rmse,
                        })
                        models[name] = model
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")
                metric_for_best = "R2"

            if not results:
                st.error("All models failed to train.")
                return

            results_df = pd.DataFrame(results).sort_values(metric_for_best, ascending=False)
            best_row = results_df.iloc[0]
            best_name = best_row["Model"]
            best_model = models[best_name]

            st.session_state["ml_results"] = {
                "target": target,
                "is_classification": is_classification,
                "results_df": results_df,
                "best_name": best_name,
                "leakage_warnings": leakage_warnings,
                "feature_cols": feature_cols,
            }

        st.success(f"Best model: **{best_name}**")
        st.dataframe(results_df.style.format(precision=3), use_container_width=True)

        fig = px.bar(results_df, x="Model", y=metric_for_best, title=f"Model comparison ({metric_for_best})", color="Model", template="plotly_dark")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(fig, use_container_width=True)

        if leakage_warnings:
            st.subheader("🚨 Possible data leakage detected")
            for w in leakage_warnings:
                st.markdown(f"<div class='anomaly-box'>⚠️ {w}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# ---- STAGE 6: Report Generation ----
# ----------------------------------------------------------------------
def build_report(df: pd.DataFrame, info: dict, quality: dict, discoveries: list) -> str:
    lines = []
    lines.append("DATADETECTIVE — AUTOMATED DATA INVESTIGATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    lines.append("1. DATASET OVERVIEW")
    lines.append(f"Rows: {info['rows']} | Columns: {info['cols']}")

    lines.append("2. DATA QUALITY")
    lines.append(f"Data Health Score: {quality['score']}/100 — {quality['verdict']}")
    lines.append(f"Missing values: {quality['missing_pct']}% of all cells")
    lines.append(f"Duplicate rows: {quality['duplicates']} ({quality['duplicate_pct']}%)")

    lines.append("3. UNEXPECTED DISCOVERIES")
    for d in discoveries:
        lines.append(f"  - {d.replace('**', '')}")

    return "
".join(lines)


# ----------------------------------------------------------------------
# ---- Sidebar Workspace ----
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">🔎 Data<span>Detective</span></div>
        <div class="brand-tagline">Autonomous Data Investigation & ML Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Data Workspace</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a real CSV or Excel dataset for investigation."
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.session_state.df = df
            st.session_state.dataset_name = uploaded.name
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">Current Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px;">
                <div style="font-size:12px; font-weight:700; color:#f8fafc;">
                    {st.session_state.get("dataset_name", "Dataset")}
                </div>
                <div style="font-size:11px; color:#38bdf8; font-family:'JetBrains Mono'; margin-top:5px;">
                    {st.session_state.df.shape[0]:,} records · {st.session_state.df.shape[1]} attributes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("↻ Reset Workspace", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Platform Modules</div>', unsafe_allow_html=True)
    st.caption("Data Visualization")
    st.caption("Machine Learning")
    st.caption("Privacy & Security")
    st.caption("Automated Investigation")


# ----------------------------------------------------------------------
# ---- Main Application Interface (Exact 10 Original Tabs Preserved) ----
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="header-title">Data Investigation Workspace</div>
            <div class="header-subtitle">Turn raw datasets into trusted, explainable insights.</div>
        </div>
        <div class="secure-pill">● SECURE SESSION ACTIVE</div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.df is None:
    st.markdown(
        """
        <div class="welcome">
            <h1>Find what your data is hiding.</h1>
            <p>
                DataDetective automatically profiles your dataset, checks data quality and privacy risks, 
                discovers unexpected patterns, detects anomalies, compares machine-learning models and builds reports.
            </p>
            <div class="workflow">
                <div class="workflow-item">
                    <div class="workflow-number">01</div>
                    <div class="workflow-name">Understand</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">02</div>
                    <div class="workflow-name">Investigate</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">03</div>
                    <div class="workflow-name">Model</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">04</div>
                    <div class="workflow-name">Report</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">Dataset Profiling</div><div class="kpi-value">01</div><div class="kpi-note">Structure & target discovery</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">Detective Mode</div><div class="kpi-value">02</div><div class="kpi-note">Patterns & anomalies</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">ML Intelligence</div><div class="kpi-value">03</div><div class="kpi-note">Models & explanations</div></div>""", unsafe_allow_html=True)

else:
    df = st.session_state.df
    info = understand_dataset(df)

    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:11px; padding:11px 15px; margin-bottom:16px;">
            <span style="font-size:11px; color:#94a3b8; font-family:'JetBrains Mono';">ACTIVE DATASET</span>
            <span style="font-size:13px; font-weight:700; color:#38bdf8; margin-left:10px;">
                {st.session_state.get("dataset_name", "Uploaded dataset")}
            </span>
            <span style="font-size:11px; color:#94a3b8; margin-left:12px; font-family:'JetBrains Mono';">
                {info['rows']:,} rows · {info['cols']} columns
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # EXACT 10 ORIGINAL TABS RESTORED
    tabs = st.tabs([
        "Overview",
        "Data Health",
        "Privacy",
        "EDA",
        "Detective Mode",
        "Model Lab",
        "Spark",
        "Cloud",
        "Deep Learning",
        "Reports"
    ])

    # ---- Tab 0: Overview ----
    with tabs[0]:
        quality_preview = data_quality_report(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Records</div><div class="kpi-value">{info['rows']:,}</div><div class="kpi-note">Rows detected</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Attributes</div><div class="kpi-value">{info['cols']}</div><div class="kpi-note">Columns detected</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Data Health</div><div class="kpi-value">{quality_preview['score']}/100</div><div class="kpi-note">Quality score</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Targets</div><div class="kpi-value">{len(info['target_candidates'])}</div><div class="kpi-note">Prediction candidates</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown('<div class="section-title">Column intelligence</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing": df.isna().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)

    # ---- Tab 1: Data Health ----
    with tabs[1]:
        quality = data_quality_report(df)
        st.session_state["quality"] = quality

        st.markdown('<div class="section-title">Data Health Assessment</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"""
                <div class="health-panel">
                    <div style="font-size:11px; color:#94a3b8; font-weight:700;">DATA HEALTH SCORE</div>
                    <div class="big-score" style="color:{quality['color']} !important;">{quality['score']}/100</div>
                    <div style="font-size:12px; color:#f8fafc; font-weight:600;">{quality['verdict']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=quality["score"],
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": quality["color"]},
                        "steps": [
                            {"range": [0, 45], "color": "rgba(239, 68, 68, 0.2)"},
                            {"range": [45, 65], "color": "rgba(249, 115, 22, 0.2)"},
                            {"range": [65, 85], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [85, 100], "color": "rgba(16, 185, 129, 0.2)"},
                        ],
                    },
                    number={"font": {"size": 42, "color": "#f8fafc"}},
                    domain={"x": [0.1, 0.9], "y": [0, 1]},
                )
            )
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Privacy ----
    with tabs[2]:
        privacy.render_privacy(df)

    # ---- Tab 3: EDA ----
    with tabs[3]:
        st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        render_eda(df, info)

    # ---- Tab 4: Detective Mode ----
    with tabs[4]:
        st.markdown('<div class="section-title">Data Detective Mode</div>', unsafe_allow_html=True)
        discoveries = discover_patterns(df, info)
        st.session_state["discoveries"] = discoveries

        st.markdown("#### Unexpected discoveries")
        for d in discoveries:
            st.markdown(f"<div class='discovery-box'>🔎 {d}</div>", unsafe_allow_html=True)

        st.markdown("#### Anomaly detection (Isolation Forest)")
        result, msg = detect_anomalies(df, info)
        st.info(msg)
        if result is not None:
            anomalies = result[result["_anomaly"]].drop(columns=["_anomaly"]).sort_values("_anomaly_score").head(20)
            st.dataframe(anomalies, use_container_width=True)

    # ---- Tab 5: Model Lab ----
    with tabs[5]:
        st.markdown('<div class="section-title">Model Lab</div>', unsafe_allow_html=True)
        run_ml_lab(df, info)

    # ---- Tab 6: Spark ----
    with tabs[6]:
        ext.render_spark(df)

    # ---- Tab 7: Cloud ----
    with tabs[7]:
        ext.render_cloud(df)

    # ---- Tab 8: Deep Learning ----
    with tabs[8]:
        ext.render_dl(df, info, prep_features)

    # ---- Tab 9: Reports ----
    with tabs[9]:
        st.markdown('<div class="section-title">Investigation Report</div>', unsafe_allow_html=True)
        quality = st.session_state.get("quality") or data_quality_report(df)
        discoveries = st.session_state.get("discoveries") or discover_patterns(df, info)
        report_text = build_report(df, info, quality, discoveries)

        st.code(report_text, language="text")
        st.download_button(
            label="⬇️ Download Text Report",
            data=report_text,
            file_name="datadetective_investigation_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown(
    """
    <div class="app-footer">
        DataDetective · Autonomous Data Investigation Platform · Secure Enterprise Session
    </div>
    """,
    unsafe_allow_html=True,
)
