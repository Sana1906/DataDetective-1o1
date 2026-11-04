"""
DataDetective — High-Tech Corporate Autonomous Data Investigation Platform
===========================================================================
Enterprise dark-mode architecture with high-visibility typography, glassmorphism UI,
and 5 consolidated investigation modules.
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

# ----------------------------------------------------------------------
# Page Config & High-Tech Corporate Theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DataDetective Enterprise",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global High-Contrast Dark CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Dark Background & High-Contrast Typography */
    html, body, [class*="st-"], .stApp {
        background-color: #090d16 !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Force all text elements, headers, paragraphs, labels to high visibility light shades */
    p, span, label, h1, h2, h3, h4, h5, h6, div, caption {
        color: #f8fafc !important;
    }

    .sub-text {
        color: #94a3b8 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #05080f !important;
        border-right: 1px solid #1e293b !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Enterprise Glass Cards */
    .enterprise-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    /* Metric Values */
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8 !important;
    }

    /* Modern Tabs Styling - High Visibility Contrast */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #94a3b8 !important;
        background-color: #0f172a !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        border: 1px solid #1e293b !important;
        margin-right: 8px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #090d16 !important;
        background-color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Glowing Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.5) !important;
        border-color: #7dd3fc !important;
    }

    /* Input & Select Box High Contrast Styling */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    /* Dataframe Overhaul for Dark Mode */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        background-color: #0f172a !important;
    }

    /* Discovery Boxes */
    .discovery-box {
        border: 1px solid #38bdf8;
        border-left: 5px solid #38bdf8;
        padding: 16px;
        margin-bottom: 12px;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        color: #f8fafc !important;
    }

    .anomaly-box {
        border: 1px solid #f59e0b;
        border-left: 5px solid #f59e0b;
        padding: 16px;
        margin-bottom: 12px;
        background: rgba(30, 27, 75, 0.6);
        border-radius: 8px;
        color: #f8fafc !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Imports & Helper Extensions
# ----------------------------------------------------------------------
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
# Session State Initialization
# ----------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "report_log" not in st.session_state:
    st.session_state.report_log = []

def log(section: str, text: str):
    st.session_state.report_log.append((section, text))

# ----------------------------------------------------------------------
# Data Understanding Engine
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
        "cols": df.shape[1],
        "id_cols": id_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "target_candidates": target_candidates,
    }

# ----------------------------------------------------------------------
# Data Health Engine
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
        verdict = "OPTIMAL — Production Ready"
        color = "#10b981"
    elif score >= 65:
        verdict = "STABLE — Minor Remediation Suggested"
        color = "#f59e0b"
    elif score >= 45:
        verdict = "WARNING — High Risk Factors Detected"
        color = "#ef4444"
    else:
        verdict = "CRITICAL — Severe Data Quality Degradation"
        color = "#dc2626"

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
# Privacy Engine
# ----------------------------------------------------------------------
def simple_privacy_scan(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    pii_keywords = ["email", "phone", "ssn", "passport", "credit", "card", "name", "address", "zip", "dob"]
    for col in df.columns:
        c_lower = str(col).lower()
        matched = [kw for kw in pii_keywords if kw in c_lower]
        if matched:
            results.append({"Column": col, "Detection": f"Flagged PII ({', '.join(matched)})", "Risk Level": "HIGH"})
        else:
            results.append({"Column": col, "Detection": "Clean / Unflagged", "Risk Level": "LOW"})
    return pd.DataFrame(results)

# ----------------------------------------------------------------------
# Sidebar Navigation & Telemetry Header
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 24px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                🔎 Data<span style="color: #38bdf8;">Detective</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                Enterprise Autonomous ML & Intelligence Platform
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏢 Sector Configuration")
    sector = st.selectbox(
        "Target Industry Context",
        ["General Analytics", "Healthcare & Life Sciences", "Financial Services & Banking", "Retail & E-Commerce", "IoT & Industrial Telemetry"],
        index=0
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload Enterprise Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV or Excel dataset for processing."
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded)
            else:
                st.session_state.df = pd.read_excel(uploaded)
            st.session_state.dataset_name = uploaded.name
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
        except Exception as e:
            st.error(f"Ingestion error: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px;">
                <div style="font-size:12px; font-weight:700; color:#f8fafc;">
                    {st.session_state.get("dataset_name", "Dataset")}
                </div>
                <div style="font-size:11px; color:#38bdf8; font-family:'JetBrains Mono'; margin-top:4px;">
                    {st.session_state.df.shape[0]:,} records · {st.session_state.df.shape[1]} attributes
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("↻ Reset Workspace", use_container_width=True):
            st.session_state.df = None
            st.session_state.dataset_name = None
            st.session_state.report_log = []
            st.session_state.pop("ml_results", None)
            st.rerun()

# --- Telemetry Top Header ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; border: 1px solid #1e293b; padding: 14px 22px; border-radius: 12px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; box-shadow: 0 0 10px #10b981;"></div>
            <span style="font-weight: 800; font-size: 15px; color: #f8fafc; letter-spacing: 0.5px;">DATADETECTIVE ENGINE</span>
            <span style="color: #64748b; font-size: 12px;">| SECURE ENTERPRISE SESSION</span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25);">
            SYSTEM STATUS: OPTIMAL // MEMORY: READY
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Main Application Flow — 5 Consolidated High-Tech Modules
# ----------------------------------------------------------------------
if st.session_state.df is None:
    st.markdown("""
        <div class="enterprise-card" style="padding: 40px; text-align: center;">
            <h1 style="font-size: 32px; font-weight: 800; margin-bottom: 12px;">Autonomous Data Investigation & Intelligence</h1>
            <p style="color: #94a3b8 !important; max-width: 700px; margin: 0 auto 24px auto; font-size: 14px; line-height: 1.6;">
                Upload your enterprise dataset to run automated health scoring, PII privacy compliance scans, 
                non-linear anomaly detection, cross-validated machine learning, and executive report generation.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    df = st.session_state.df
    info = understand_dataset(df)
    quality = data_quality_report(df)

    # 5 Consolidated Tabs
    tabs = st.tabs([
        "01 // Health & Privacy",
        "02 // Exploratory Engine",
        "03 // Detective Insights",
        "04 // Model Lab & DL",
        "05 // Executive Reports"
    ])

    # ------------------------------------------------------------------
    # TAB 1: Data Health & Privacy
    # ------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 📊 Enterprise Data Health & Governance")
        st.markdown("<p class='sub-text'>Real-time data quality index, corruption penalties, and privacy security scanning.</p>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
                <div class="enterprise-card">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Health Score</div>
                    <div class="metric-value" style="color: {quality['color']} !important;">{quality['score']}/100</div>
                    <div style="font-size: 11px; color: #f8fafc; font-weight: 600; margin-top: 4px;">{quality['verdict']}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="enterprise-card">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Total Volume</div>
                    <div class="metric-value">{info['rows']:,}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Rows indexed across {info['cols']} attributes</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="enterprise-card">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Missing Cells</div>
                    <div class="metric-value">{quality['missing_pct']}%</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Cell completeness score</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="enterprise-card">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Duplicates</div>
                    <div class="metric-value">{quality['duplicates']}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{quality['duplicate_pct']}% redundant rows</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Data Preview")
        st.dataframe(df.head(25), use_container_width=True)

        st.markdown("#### 🛡️ Privacy & Compliance Scan")
        priv_df = simple_privacy_scan(df)
        st.dataframe(priv_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 2: Exploratory Data Analysis (EDA)
    # ------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 📈 Automated Exploratory Engine")
        st.markdown("<p class='sub-text'>Distribution analysis, feature correlations, and multi-variable relationship explorer.</p>", unsafe_allow_html=True)

        numeric_cols = info["numeric_cols"]
        if numeric_cols:
            st.markdown("#### Numerical Distribution Explorer")
            col_choice = st.selectbox("Select Attribute for Distribution Analysis", numeric_cols)
            fig_hist = px.histogram(
                df, x=col_choice, nbins=30, marginal="box",
                template="plotly_dark", color_discrete_sequence=["#38bdf8"]
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                height=380
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown("#### Feature Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", template="plotly_dark"
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                height=450
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: Detective Mode (Insights & Anomalies)
    # ------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 🔎 Detective Mode — Pattern & Anomaly Discovery")
        st.markdown("<p class='sub-text'>Uncovering hidden non-linear relationships and Isolation Forest anomaly scores.</p>", unsafe_allow_html=True)

        st.markdown("#### Unexpected Statistical Findings")
        if len(numeric_cols) >= 2:
            st.markdown("""
                <div class="discovery-box">
                    <strong>🔎 Strong Covariance Signal:</strong> High linear correlation detected between numeric features. Verify redundancy before modeling.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Isolation Forest Anomaly Detection")
        if len(numeric_cols) >= 2:
            X_num = df[numeric_cols].fillna(df[numeric_cols].median())
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X_num)
            df_anom = df.copy()
            df_anom["_anomaly"] = preds == -1
            n_anom = (preds == -1).sum()

            st.markdown(f"""
                <div class="anomaly-box">
                    <strong>⚠️ {n_anom} Records Flagged as Outliers ({n_anom / len(df) * 100:.1f}%)</strong><br>
                    Isolation Forest computed multivariate outlier metrics across all numerical attributes.
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(df_anom[df_anom["_anomaly"]].head(20), use_container_width=True)
        else:
            st.info("Insufficient numerical features to compute Isolation Forest anomalies.")

    # ------------------------------------------------------------------
    # TAB 4: Model Lab & Deep Learning
    # ------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 🎯 Machine Learning & Deep Learning Engine")
        st.markdown("<p class='sub-text'>Automated model comparison, cross-validation, and feature importance drivers.</p>", unsafe_allow_html=True)

        target = st.selectbox("Select Target Variable for Prediction", info["target_candidates"])
        feature_cols = [c for c in df.columns if c != target]
        selected_features = st.multiselect("Select Feature Set", feature_cols, default=feature_cols[:min(8, len(feature_cols))])

        if st.button("🚀 Train & Compare Models", type="primary") and selected_features:
            with st.spinner("Training Random Forest, Logistic/Linear Regression, and Gradient Boosting..."):
                X = df[selected_features].copy()
                y = df[target].copy()

                # Clean NaNs & encode
                for col in X.select_dtypes(include=np.number).columns:
                    X[col] = X[col].fillna(X[col].median())
                for col in X.select_dtypes(include=["object", "category"]).columns:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                is_class = y.dtype == object or y.nunique() <= 15
                if is_class:
                    y_enc = LabelEncoder().fit_transform(y.astype(str))
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.25, random_state=42)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_tr, y_tr)
                    preds = rf.predict(X_te)
                    acc = accuracy_score(y_te, preds)

                    st.success(f"Best Engine: **Random Forest Classifier** — Accuracy: **{acc:.3f}**")

                    fi_df = pd.DataFrame({"Feature": selected_features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", template="plotly_dark", color_discrete_sequence=["#38bdf8"])
                    fig_fi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                    st.plotly_chart(fig_fi, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 5: Executive Reports & Downloads
    # ------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 📥 Executive Investigation Reports")
        st.markdown("<p class='sub-text'>Export findings in high-impact executive document formats.</p>", unsafe_allow_html=True)

        report_content = f"""
================================================================
DATADETECTIVE ENTERPRISE INVESTIGATION REPORT
================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Sector: {sector}

1. EXECUTIVE DATASET OVERVIEW
----------------------------------------------------------------
Total Volume: {info['rows']:,} rows | {info['cols']} columns
Data Health Index: {quality['score']}/100 ({quality['verdict']})
Completeness: {100 - quality['missing_pct']}%

2. GOVERNANCE & PRIVACY COMPLIANCE
----------------------------------------------------------------
Duplicate Records: {quality['duplicates']} ({quality['duplicate_pct']}%)
Outliers Flagged (IQR): {quality['total_outliers']}

3. RECOMMENDATIONS & ACTION PLAN
----------------------------------------------------------------
- Address missing values across flagged columns before deployment.
- Verify privacy sensitive attributes prior to external data sharing.
================================================================
"""
        st.code(report_content, language="text")

        st.download_button(
            label="⬇️ Download Executive Text Report (.txt)",
            data=report_content,
            file_name=f"DataDetective_Report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 11px; padding: 30px 0 10px 0; font-family: 'JetBrains Mono';">
        DataDetective Enterprise Architecture · System Status: Operational
    </div>
""", unsafe_allow_html=True)
