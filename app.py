"""
DataDetective — An Autonomous Data Investigation & ML Platform
================================================================
A Streamlit web app that lets a user upload a real dataset (CSV / Excel)
and automatically:
  1. Understands the dataset (types, shape, target candidates)
  2. Scores its Data Health (missing values, duplicates, outliers, imbalance)
  3. Runs a Privacy Scan (flags likely sensitive columns)
  4. Performs context-aware, automated EDA
  5. Runs "Data Detective Mode" — hidden pattern & anomaly discovery
  6. Trains and compares multiple ML models (classification or regression,
     auto-detected from the chosen target column)
  7. Explains *why* the best model was chosen and flags possible data leakage
  8. Generates a downloadable investigation report

Run with:
    streamlit run app.py
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

# Streamlit treats HTML indented by 4+ spaces (or containing blank lines) as a
# code block. Normalise every HTML snippet so it renders as real HTML.
_orig_markdown = st.markdown


def _markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = _re.sub(r"(?m)^[ \t]+", "", body)
        body = _re.sub(r"\n\s*\n", "\n", body)
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

# Optional: gradient boosting if available, else silently skip
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ----------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DataDetective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1f3a;
        border-right: 1px solid #173455;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] * {
        color: #eef5ff !important;
    }

    .brand {
        padding: 6px 8px 20px 8px;
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .brand-name span {
        color: #4d9cff;
    }

    .brand-tagline {
        color: #91a7c2 !important;
        font-size: 11px;
        margin-top: 4px;
        line-height: 1.4;
    }

    .sidebar-label {
        color: #7890ad !important;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 18px 8px 8px;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 22px;
        box-shadow: 0 5px 18px rgba(22, 43, 72, 0.04);
    }

    .header-title {
        font-size: 25px;
        font-weight: 800;
        color: #10233f;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #718096;
        font-size: 12px;
        margin-top: 4px;
    }

    .secure-pill {
        background: #edf7f2;
        color: #16845a;
        border: 1px solid #cdebdc;
        border-radius: 20px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 14px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 5px 18px rgba(22, 43, 72, 0.035);
    }

    .kpi-label {
        color: #74839a;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .4px;
    }

    .kpi-value {
        color: #10233f;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-note {
        color: #16845a;
        font-size: 11px;
        margin-top: 5px;
        font-weight: 600;
    }

    /* Section headers */
    .section-title {
        color: #10233f;
        font-size: 19px;
        font-weight: 750;
        margin: 22px 0 10px;
    }

    .section-caption {
        color: #7a8799;
        font-size: 12px;
        margin-bottom: 14px;
    }

    /* Welcome / empty state */
    .welcome {
        background: linear-gradient(135deg, #0b1f3a 0%, #12355e 100%);
        border-radius: 18px;
        padding: 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(11, 31, 58, .16);
    }

    .welcome h1 {
        margin: 0 0 8px 0;
        font-size: 30px;
        letter-spacing: -0.7px;
    }

    .welcome p {
        color: #c5d5e8;
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
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 11px;
        padding: 14px;
    }

    .workflow-number {
        color: #66a9ff;
        font-weight: 800;
        font-size: 11px;
    }

    .workflow-name {
        font-size: 13px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Discovery */
    .discovery-box {
        border: 1px solid #dbe6f4;
        border-left: 4px solid #347fdb;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(22,43,72,.035);
    }

    .anomaly-box {
        border: 1px solid #f0d8c9;
        border-left: 4px solid #df784f;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #fffaf7;
        border-radius: 10px;
    }

    /* Health score */
    .health-panel {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(22,43,72,.035);
    }

    .big-score {
        font-size: 48px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #102a4b;
        border-radius: 12px;
        padding: 4px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #102a4b !important;
        border: 1px dashed #4a6f98 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #687890;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #246fc1;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 650;
        border: 1px solid #d7e0eb;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #e3e9f1;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #91a0b2;
        font-size: 10px;
        padding: 25px 0 8px;
    }

    @media (max-width: 900px) {
        .workflow {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    
    /* Reset workspace button color */
section[data-testid="stSidebar"] button {
    background-color: #7C3AED !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #6D28D9 !important;
    color: #FFFFFF !important;
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
    """True when almost every value is unique (IDs, names, emails). Charts of these are meaningless."""
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
        verdict = "Excellent — ready for modeling"
        color = "#00b894"
    elif score >= 65:
        verdict = "Good — minor cleanup recommended"
        color = "#fdcb6e"
    elif score >= 45:
        verdict = "Fair — cleaning strongly recommended before modeling"
        color = "#e17055"
    else:
        verdict = "Poor — this dataset needs significant cleaning"
        color = "#d63031"

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


def privacy_scan(df: pd.DataFrame) -> pd.DataFrame:
    return privacy.scan_compat(df)


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

    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Numeric distributions")
    if numeric_cols:
        chosen_num = st.multiselect("Select numeric columns", options=numeric_cols, default=numeric_cols[: min(4, len(numeric_cols))], key="eda_num_cols")
        if chosen_num:
            plot_df = _safe_sample(df)
            n_cols = 2 if len(chosen_num) > 1 else 1
            grid = st.columns(n_cols)
            for i, c in enumerate(chosen_num):
                with grid[i % n_cols]:
                    fig = px.histogram(plot_df, x=c, title=f"Distribution of {c}", marginal="box", template="plotly_white")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation analysis")
    corr_data = _correlation_frame(df, numeric_cols, all_cat_cols)
    if corr_data.shape[1] >= 2:
        corr = corr_data.corr(numeric_only=True)
        mask = np.triu(np.ones(corr.shape, dtype=bool))
        fig = px.imshow(corr.mask(mask), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Pearson correlation (lower triangle)", template="plotly_white")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


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
        discoveries.append("No strongly unusual relationships were detected — the dataset looks fairly consistent.")
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


def prep_features(df: pd.DataFrame, target: str, feature_cols: list):
    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    for c in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[c] = X[c].astype(str).fillna("missing")
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])

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
                    warnings.append(f"**{c}** has a unique value for every row — potential identifier leakage.")
        else:
            corr = X.assign(_target=y).corr(numeric_only=True)["_target"].drop("_target", errors="ignore")
            for c, v in corr.items():
                if pd.notna(v) and abs(v) > 0.98:
                    warnings.append(f"**{c}** is correlated with target at {v:.3f} — check for target leakage.")
    except Exception:
        pass
    return warnings


def run_ml_lab(df: pd.DataFrame, info: dict):
    st.subheader("🎯 Choose a target column")
    target = st.selectbox("What should the model predict?", info["target_candidates"], key="ml_target_sel")
    feature_cols = [c for c in df.columns if c != target]
    feature_cols = st.multiselect("Feature columns to use", feature_cols, default=feature_cols[: min(10, len(feature_cols))], key="ml_feat_sel")

    if not feature_cols:
        st.warning("Select at least one feature column.")
        return

    if st.button("🚀 Run automated model comparison", type="primary", key="run_ml_btn"):
        with st.spinner("Training and comparing models..."):
            X, y, is_classification, y_encoder = prep_features(df, target, feature_cols)
            if len(X) < 20:
                st.error("Not enough rows with a valid target value to train a model (need at least 20).")
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


def build_report(df: pd.DataFrame, info: dict, quality: dict, discoveries: list) -> str:
    lines = [
        "DATADETECTIVE — AUTOMATED DATA INVESTIGATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"\n1. DATASET OVERVIEW\nRows: {info['rows']} | Columns: {info['cols']}",
        f"\n2. DATA QUALITY\nData Health Score: {quality['score']}/100 — {quality['verdict']}",
        f"\n3. UNEXPECTED DISCOVERIES"
    ]
    for d in discoveries:
        lines.append(f"  - {d.replace('**', '')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# ---- Professional application shell ----
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">🔎 Data<span>Detective</span></div>
        <div class="brand-tagline">Autonomous Data Investigation & ML Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">INDUSTRY DOMAIN</div>', unsafe_allow_html=True)
    domain = st.selectbox(
        "Select Domain Focus",
        ["General / Multi-Sector", "Healthcare & Clinical", "Finance & Banking", "Retail & E-Commerce", "Human Resources", "IoT & Manufacturing"],
        key="sidebar_domain_selector"
    )

    st.markdown('<div class="sidebar-label">DATA WORKSPACE</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sidebar-label">CURRENT DATASET</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#102a4b;border:1px solid #21466f;border-radius:10px;padding:12px;">
                <div style="font-size:12px;font-weight:700;">{st.session_state.get("dataset_name", "Dataset")}</div>
                <div style="font-size:10px;color:#9db2cb;margin-top:5px;">
                    {st.session_state.df.shape[0]:,} records · {st.session_state.df.shape[1]} attributes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("↻  Reset workspace", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
            st.rerun()


# ----------------------------------------------------------------------
# ---- Main Shell ----
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="header-title">Data Investigation Workspace</div>
            <div class="header-subtitle">Turn raw datasets into trusted, explainable insights.</div>
        </div>
        <div class="secure-pill">● SECURE SESSION</div>
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
                discovers unexpected patterns, detects anomalies, compares machine-learning models, and builds reports.
            </p>
            <div class="workflow">
                <div class="workflow-item">
                    <div class="workflow-number">01</div>
                    <div class="workflow-name">Audit & Health</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">02</div>
                    <div class="workflow-name">Explore EDA</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">03</div>
                    <div class="workflow-name">Detective Mode</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-number">04</div>
                    <div class="workflow-name">ML Lab & Reports</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    df = st.session_state.df
    info = understand_dataset(df)

    # CONSOLIDATED 5-TAB NAVIGATION OVERHAUL
    tabs = st.tabs([
        "1. Data Health & Privacy",
        "2. Exploratory Engine (EDA)",
        "3. Detective Mode (Insights)",
        "4. ML & Deep Learning Lab",
        "5. Executive Report & Cloud"
    ])

    # ---- TAB 1: Data Health & Privacy ----
    with tabs[0]:
        quality = data_quality_report(df)
        st.session_state["quality"] = quality

        st.markdown('<div class="section-title">Data Health &amp; Privacy Audit</div>', unsafe_allow_html=True)
        
        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"""
                <div class="health-panel">
                    <div style="font-size:11px;color:#7a8799;font-weight:700;">DATA HEALTH SCORE</div>
                    <div class="big-score" style="color:{quality['color']};">{quality['score']}/100</div>
                    <div style="font-size:12px;color:#526174;">{quality['verdict']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=quality["score"],
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": quality["color"]}},
                    title={"text": "Overall Data Health"}
                )
            )
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Privacy &amp; PII Scan</div>', unsafe_allow_html=True)
        privacy.render_privacy(df)

    # ---- TAB 2: Exploratory Engine (EDA) ----
    with tabs[1]:
        st.markdown('<div class="section-title">Exploratory Data Analysis Engine</div>', unsafe_allow_html=True)
        render_eda(df, info)

    # ---- TAB 3: Detective Mode (Insights) ----
    with tabs[2]:
        st.markdown('<div class="section-title">Data Detective Mode &amp; Anomaly Hunting</div>', unsafe_allow_html=True)
        discoveries = discover_patterns(df, info)
        st.session_state["discoveries"] = discoveries

        for d in discoveries:
            st.markdown(f"<div class='discovery-box'>🔎 {d}</div>", unsafe_allow_html=True)

        result, msg = detect_anomalies(df, info)
        st.info(msg)
        if result is not None:
            anomalies = result[result["_anomaly"]].drop(columns=["_anomaly"]).sort_values("_anomaly_score").head(20)
            st.dataframe(anomalies, use_container_width=True)

    # ---- TAB 4: ML & Deep Learning Lab ----
    with tabs[3]:
        st.markdown('<div class="section-title">Machine Learning &amp; Deep Learning Lab</div>', unsafe_allow_html=True)
        run_ml_lab(df, info)
        
        st.markdown("---")
        ext.render_dl(df, info, prep_features)

    # ---- TAB 5: Executive Report & Cloud ----
    with tabs[4]:
        st.markdown('<div class="section-title">Executive Investigation Report &amp; Cloud Export</div>', unsafe_allow_html=True)
        
        quality = st.session_state.get("quality") or data_quality_report(df)
        discoveries = st.session_state.get("discoveries") or discover_patterns(df, info)
        report_text = build_report(df, info, quality, discoveries)

        st.code(report_text, language="text")
        
        st.markdown("### 📥 Download Report")
        st.download_button(
            label="⬇️ Download Investigation Report (.txt)",
            data=report_text,
            file_name="datadetective_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        
        st.markdown("---")
        ext.render_cloud(df)
        ext.render_spark(df)

st.markdown(
    """
    <div class="app-footer">
        DataDetective · Autonomous Data Investigation Platform · Built for Multi-Sector Analytics.
    </div>
    """,
    unsafe_allow_html=True,
)
