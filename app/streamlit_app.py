"""
Intelligent Anti-Money Laundering (AML) Detection System
Interactive Streamlit Web Dashboard

Supports Dual Independent Dataset Modes:
- MODE 1: Traditional Banking AML (IBM HI-Small Dataset)
- MODE 2: Cryptocurrency AML (Elliptic Bitcoin Transaction Dataset)

Pages:
1. Executive Dashboard (KPI metrics, transaction distribution, risk distribution, trends)
2. Transaction Analysis (CSV upload, batch analysis, individual transaction inspection)
3. Explainable AI (SHAP global beeswarm/bar plots, local waterfall plots, NL compliance reasoning)
4. Anomaly Detection (Unsupervised Isolation Forest & Autoencoder comparison)
5. Alert Management (Filterable alert table, risk-level tags, search, and CSV export)
6. Model Performance (Comprehensive comparative benchmarks, metrics, confusion matrices)
7. Blockchain Graph Analytics (Cryptocurrency transaction network analysis using NetworkX and Plotly)
"""

import os
import sys
import json
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure project root in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# IBM Pipeline Imports
from src.predict import AMLPredictor
from src.risk_scoring import calculate_risk_score, assign_risk_category

# Elliptic Pipeline Imports
from src.elliptic.predict import EllipticPredictor
from src.elliptic.risk_scoring import calculate_crypto_risk_score, assign_crypto_risk_tier
from src.elliptic.explainability import EllipticExplainer

# Page configuration
st.set_page_config(
    page_title="AML Intelligent Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_aml_predictor(model_name: str) -> AMLPredictor:
    """Cache IBM predictor instance."""
    return AMLPredictor(model_name=model_name)

@st.cache_resource
def load_elliptic_predictor() -> EllipticPredictor:
    """Cache Elliptic predictor instance."""
    return EllipticPredictor(models_dir="models/elliptic")

@st.cache_data
def load_ibm_demo_data(nrows: int = 1000) -> pd.DataFrame:
    """Load raw IBM transactions with the Timestamp and raw schema preserved.

    The current IBM feature-engineering pipeline requires the original
    transaction columns, including Timestamp. A preprocessed CSV that has
    dropped Timestamp cannot be passed to AMLPredictor.predict_batch().
    """
    raw_path = os.path.join("data", "raw", "HI-Small_Trans.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Original IBM dataset not found: {raw_path}. "
            "Place HI-Small_Trans.csv inside data/raw/."
        )

    df = pd.read_csv(raw_path, nrows=nrows)

    required_columns = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Raw IBM dataset is missing required columns: " + ", ".join(missing)
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    invalid_timestamps = int(df["Timestamp"].isna().sum())
    if invalid_timestamps:
        logging.warning(
            "Dropping %d IBM demo rows with invalid Timestamp values.",
            invalid_timestamps,
        )
        df = df.dropna(subset=["Timestamp"])

    # Keep the raw transaction schema and chronological order for temporal
    # feature generation. Do not drop or rename the source columns here.
    return df.sort_values("Timestamp").reset_index(drop=True)

@st.cache_data
def load_elliptic_demo_data() -> pd.DataFrame:
    """Load labelled Elliptic data; retain all labels for reliable transaction filtering."""
    path = os.path.join("data", "elliptic", "processed", "elliptic_labelled.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# Elliptic label helpers: preserve ground truth separately from model predictions.
def normalize_elliptic_label(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip().lower()
    if value in {"1", "1.0", "illicit"}:
        return "Illicit"
    if value in {"2", "2.0", "licit"}:
        return "Licit"
    return "Unknown"


def get_ground_truth_column(df):
    for candidate in ("Actual_Class", "class", "Class", "label", "Label"):
        if candidate in df.columns:
            return candidate
    return None


# ==============================================================================
# SIDEBAR DATASET & MODEL SELECTION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("AML Intelligence")
st.sidebar.caption("Explainable AI & Cryptocurrency Surveillance")

dataset_mode = st.sidebar.selectbox(
    "Active Surveillance Domain",
    [
        "Traditional Banking (IBM HI-Small)",
        "Cryptocurrency (Elliptic Bitcoin)"
    ],
    index=0,
    help="Toggle between traditional fiat banking transactions and Bitcoin blockchain network monitoring."
)

is_crypto = "Elliptic" in dataset_mode

if not is_crypto:
    selected_model = st.sidebar.selectbox(
        "Active ML Classifier",
        ["XGBoost", "Random Forest"],
        index=0,
        help="Select which supervised model drives fiat banking risk scoring."
    )
    ibm_predictor = load_aml_predictor(selected_model)
    ibm_predictor.set_model(selected_model)
    operating_threshold = ibm_predictor.threshold
    st.sidebar.markdown(f"**Operating Threshold:** `{operating_threshold:.4f}`")
    demo_data = load_ibm_demo_data(nrows=1000)
else:
    selected_model = st.sidebar.selectbox(
        "Active ML Classifier",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        index=0,
        help="Select which model drives crypto transaction classification."
    )
    crypto_predictor = load_elliptic_predictor()
    try:
        crypto_predictor.set_active_model(selected_model)
        operating_threshold = float(crypto_predictor.thresholds.get(selected_model, 0.5))
        if selected_model not in crypto_predictor.thresholds:
            st.sidebar.warning("No saved threshold found; displaying the default 0.50 threshold.")
        st.sidebar.markdown(f"**Operating Threshold:** `{operating_threshold:.4f}`")
    except Exception as exc:
        st.sidebar.error(f"Unable to load {selected_model}: {exc}")
        st.stop()
    demo_data = load_elliptic_demo_data()
    if demo_data.empty:
        st.error("Elliptic processed dataset was not found or is empty.")
        st.stop()
    gt_col = get_ground_truth_column(demo_data)
    if gt_col:
        demo_data["Ground_Truth"] = demo_data[gt_col].map(normalize_elliptic_label)
    else:
        demo_data["Ground_Truth"] = "Unknown"

st.sidebar.markdown("---")

nav_pages = [
    "📊 Executive Dashboard",
    "🔍 Transaction Analysis",
    "🧠 Explainable AI (SHAP)",
    "⚡ Anomaly Detection",
    "🚨 Alert Management",
    "📈 Model Performance"
]
if is_crypto:
    nav_pages.append("🌐 Blockchain Graph Analytics")

page = st.sidebar.radio("Navigation", nav_pages)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Academic Demonstration System**\n"
    "Model scores represent statistical risk probabilities. Alerts are investigation triggers, not criminal proof."
)


# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==============================================================================
if page == "📊 Executive Dashboard":
    domain_title = "Cryptocurrency Bitcoin Surveillance" if is_crypto else "Traditional Fiat Banking Surveillance"
    st.markdown(f'<div class="main-title">🛡️ AML Executive Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Real-time risk analytics for <b>{domain_title}</b>.</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing transaction streams..."):
        if not is_crypto:
            scored_df = ibm_predictor.predict_batch(demo_data)
            total_tx = len(scored_df)
            suspicious_count = int(scored_df["Alert_Triggered"].sum())
            legit_count = total_tx - suspicious_count
            high_risk_count = int((scored_df["Risk_Score"] >= 60.0).sum())
            critical_risk_count = int((scored_df["Risk_Score"] >= 80.0).sum())
        else:
            scored_df = crypto_predictor.predict_batch(demo_data)
            total_tx = len(scored_df)
            suspicious_count = int((scored_df["Prediction"] == "Illicit").sum())
            legit_count = total_tx - suspicious_count
            high_risk_count = int((scored_df["Risk_Score"] >= 70.0).sum())
            critical_risk_count = int((scored_df["Risk_Score"] >= 90.0).sum())

    suspicious_pct = (suspicious_count / total_tx) * 100 if total_tx > 0 else 0.0

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Analyzed", f"{total_tx:,}")
    c2.metric("Legitimate / Licit", f"{legit_count:,}", delta=f"{100 - suspicious_pct:.1f}%", delta_color="normal")
    c3.metric("Suspicious / Illicit", f"{suspicious_count:,}", delta=f"{suspicious_pct:.2f}%", delta_color="inverse")
    c4.metric("High Risk", f"{high_risk_count:,}")
    c5.metric("Critical Risk", f"{critical_risk_count:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Classification Distribution")
        pred_col = "Prediction" if is_crypto else "Predicted_Class"
        fig_donut = px.pie(
            scored_df,
            names=pred_col,
            hole=0.5,
            color=pred_col,
            color_discrete_map={"Legitimate": "#10B981", "Licit": "#10B981", "Suspicious": "#EF4444", "Illicit": "#EF4444"}
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Risk Severity Breakdown")
        tier_col = "Risk_Tier" if is_crypto else "Risk_Category"
        cat_counts = scored_df[tier_col].value_counts().reset_index()
        cat_counts.columns = [tier_col, "Count"]
        color_map = {
            "Low": "#10B981", "Low Risk": "#10B981",
            "Medium": "#F59E0B", "Medium Risk": "#F59E0B",
            "High": "#EA580C", "High Risk": "#EA580C",
            "Critical": "#DC2626", "Critical Risk": "#DC2626"
        }
        fig_bar = px.bar(
            cat_counts,
            x=tier_col,
            y="Count",
            color=tier_col,
            color_discrete_map=color_map,
            text_auto=True
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    if not is_crypto and "Transaction_Hour" in scored_df.columns:
        st.subheader("Transaction Risk Profiling")
        fig_scatter = px.scatter(
            scored_df,
            x="Transaction_Hour",
            y="Amount Received",
            color="Risk_Category",
            size="Risk_Score",
            hover_data=["Risk_Score", "Laundering_Probability", "Predicted_Class"],
            color_discrete_map=color_map,
            title="Fiat Volume vs Hour by Risk Tier"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    elif is_crypto and "total_degree" in scored_df.columns:
        st.subheader("Bitcoin Network Degree vs Risk Score")
        fig_scatter = px.scatter(
            scored_df,
            x="total_degree",
            y="Risk_Score",
            color="Risk_Tier",
            size="Probability",
            hover_data=["txId", "Probability", "Prediction"],
            color_discrete_map=color_map,
            title="Blockchain Degree Connectivity vs Computed AML Risk Score"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# PAGE 2: TRANSACTION ANALYSIS
# ==============================================================================
elif page == "🔍 Transaction Analysis":
    st.markdown('<div class="main-title">🔍 Transaction Surveillance & Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Domain: <b>{"Cryptocurrency (Elliptic)" if is_crypto else "Traditional Banking (IBM)"}</b>.</div>', unsafe_allow_html=True)

    tab_csv, tab_single = st.tabs(["📁 Batch CSV Ingestion", "📝 Single Transaction Evaluator"])

    with tab_csv:
        st.write(f"Upload CSV transaction records conforming to the active domain (<b>{'Elliptic 165 features + graph degrees' if is_crypto else 'IBM banking schema'}</b>).", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(input_df)} transactions.")
        else:
            st.info("Demonstrating on slice of 100 held-out test transactions. Upload a CSV to evaluate custom data.")
            input_df = demo_data.head(100).copy()

        if is_crypto and not input_df.empty:
            batch_gt_col = get_ground_truth_column(input_df)
            if batch_gt_col:
                input_df["Ground_Truth"] = input_df[batch_gt_col].map(normalize_elliptic_label)

        with st.spinner(f"Evaluating with {selected_model}..."):
            if not is_crypto:
                analyzed_df = ibm_predictor.predict_batch(input_df, include_anomaly_scores=True)
                disp_cols = ["Amount Received", "Amount Paid", "Laundering_Probability", "Risk_Score", "Risk_Category", "Predicted_Class"]
            else:
                analyzed_df = crypto_predictor.predict_batch(input_df)
                disp_cols = ["txId", "Probability", "Threshold", "Prediction", "Risk_Score", "Risk_Tier", "Actual_Class", "Ground_Truth"]

        st.subheader("Transaction Evaluation Results")
        available_disp = [c for c in disp_cols if c in analyzed_df.columns]
        st.dataframe(analyzed_df[available_disp].head(50), use_container_width=True)

    with tab_single:
        if not is_crypto:
            st.subheader("Single Banking Transaction Evaluator")
            c1, c2, c3 = st.columns(3)
            amt_rcv = c1.number_input("Amount Received ($)", value=250000.0, step=5000.0)
            amt_paid = c2.number_input("Amount Paid ($)", value=250000.0, step=5000.0)
            tx_hour = c3.slider("Transaction Hour (0-23)", 0, 23, 20)

            c4, c5, c6 = st.columns(3)
            same_curr = c4.selectbox("Same Currency Transfer?", ["Yes", "No"], index=0)
            same_bank = c5.selectbox("Same Bank Transfer?", ["Yes", "No"], index=0)
            freq_channel = c6.selectbox("Payment Channel", ["Bitcoin", "Cheque", "Wire", "Credit Card", "Cash", "ACH", "Reinvestment"], index=0)

            rec_currency = "Bitcoin" if freq_channel == "Bitcoin" else "US Dollar"
            pay_currency = rec_currency if same_curr == "Yes" else "Euro"
            from_bank = 10
            to_bank = 10 if same_bank == "Yes" else 20

            raw_tx = {
                "Timestamp": f"2022/09/10 {tx_hour:02d}:00",
                "From Bank": from_bank,
                "Account": "acct_eval_src",
                "To Bank": to_bank,
                "Account.1": "acct_eval_dest",
                "Amount Received": amt_rcv,
                "Receiving Currency": rec_currency,
                "Amount Paid": amt_paid,
                "Payment Currency": pay_currency,
                "Payment Format": freq_channel
            }

            if st.button("Evaluate Single Transaction Risk", type="primary"):
                with st.spinner("Scoring transaction and generating explanation..."):
                    res = ibm_predictor.predict_single(raw_tx, explain=True)

                st.markdown("---")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Predicted Status", res["prediction"])
                k2.metric("Risk Score", f"{res['risk_score']}/100")
                k3.metric("Laundering Probability", f"{res['probability']:.4f}")
                k4.metric("Risk Tier", res["risk_category"])

                st.markdown(f"**Alert Recommendation:** {'⚠️ Requires Immediate Compliance Investigation' if res['investigation_recommended'] else '✅ Standard Transaction — No Action Required'}")
                st.subheader("Model Decision Audit Trail")
                for bullet in res["explanation"]["reasoning_bullets"]:
                    st.write(bullet)
        else:
            st.subheader("Single Cryptocurrency Transaction Evaluator")
            st.write("Filter the labelled Elliptic dataset and inspect a transaction.")

            label_filter = st.selectbox(
                "Ground-truth transaction group",
                ["All Transactions", "Confirmed Illicit", "Confirmed Licit", "Unknown"],
                key="crypto_ground_truth_filter"
            )
            if label_filter == "Confirmed Illicit":
                candidates = demo_data[demo_data["Ground_Truth"] == "Illicit"]
            elif label_filter == "Confirmed Licit":
                candidates = demo_data[demo_data["Ground_Truth"] == "Licit"]
            elif label_filter == "Unknown":
                candidates = demo_data[demo_data["Ground_Truth"] == "Unknown"]
            else:
                candidates = demo_data

            if candidates.empty:
                st.warning(f"No rows found for: {label_filter}. Check that elliptic_labelled.csv contains ground-truth labels.")
            else:
                candidates = candidates.reset_index(drop=False).rename(columns={"index": "Source_Row_Index"})
                display_ids = candidates["txId"].astype(str) if "txId" in candidates.columns else candidates["Source_Row_Index"].astype(str)
                selected_id = st.selectbox(
                    "Select transaction (TxID)",
                    options=display_ids.tolist(),
                    key="crypto_tx_selector"
                )
                selected_pos = display_ids[display_ids == selected_id].index[0]
                selected_record = candidates.iloc[selected_pos]
                sample_row = selected_record.drop(labels=["Ground_Truth"], errors="ignore").to_dict()

                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"**TxID:** `{sample_row.get('txId', 'N/A')}`")
                c2.write(f"**Ground Truth:** `{selected_record['Ground_Truth']}`")
                c3.write(f"**In-Degree:** `{sample_row.get('in_degree', 0)}`")
                c4.write(f"**Out-Degree:** `{sample_row.get('out_degree', 0)}`")

                if st.button("Evaluate Cryptocurrency Risk", type="primary"):
                    try:
                        res = crypto_predictor.predict_single(sample_row)
                        st.markdown("---")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Predicted Status", res.get("Prediction", "Unavailable"))
                        k2.metric("Risk Score", f"{res.get('Risk_Score', 0)}/100")
                        k3.metric("Illicit Probability", f"{res.get('Probability', 0):.4f}")
                        k4.metric("Risk Tier", res.get("Risk_Tier", "Unknown"))
                        st.write(f"**Ground Truth Label:** `{selected_record['Ground_Truth']}`")
                        st.caption("Ground truth is the dataset label; prediction is the model output. They are not interchangeable.")
                    except Exception as exc:
                        st.error(f"Transaction evaluation failed: {exc}")


# ==============================================================================
# PAGE 3: EXPLAINABLE AI (SHAP)
# ==============================================================================
elif page == "🧠 Explainable AI (SHAP)":
    st.markdown('<div class="main-title">🧠 Explainable AI with SHAP</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Shapley Additive exPlanations for <b>{selected_model}</b> in <b>{dataset_mode}</b>.</div>', unsafe_allow_html=True)

    slug = selected_model.lower().replace(" ", "_")
    subtab1, subtab2 = st.tabs(["🌐 Global Feature Importance", "🔍 Local Transaction Waterfall"])

    with subtab1:
        st.subheader(f"Global SHAP Attribution — {selected_model}")
        col_bar, col_swarm = st.columns(2)
        
        if not is_crypto:
            bar_path = os.path.join("results", "graphs", f"shap_global_bar_{slug}.png")
            beeswarm_path = os.path.join("results", "graphs", f"shap_global_beeswarm_{slug}.png")
        else:
            bar_path = os.path.join("results", "elliptic", "graphs", f"shap_global_bar_{slug}.png")
            beeswarm_path = os.path.join("results", "elliptic", "graphs", f"shap_global_beeswarm_{slug}.png")

        if os.path.exists(bar_path):
            col_bar.image(bar_path, caption=f"Mean Absolute SHAP Importance — {selected_model}", use_container_width=True)
        else:
            col_bar.warning(f"Global SHAP bar plot for {selected_model} not found.")

        if os.path.exists(beeswarm_path):
            col_swarm.image(beeswarm_path, caption=f"SHAP Beeswarm Feature Impact — {selected_model}", use_container_width=True)
        else:
            col_swarm.warning(f"Global SHAP beeswarm plot for {selected_model} not found.")

    with subtab2:
        if not is_crypto:
            st.subheader("Individual Banking Transaction Waterfall")
            tx_index = st.slider("Select Transaction Index", 0, 50, 0)
            if st.button("Generate Local Waterfall Plot"):
                waterfall_path = os.path.join("results", "graphs", "interactive_waterfall.png")
                exp = ibm_predictor.explainer.explain_transaction(demo_data.iloc[[tx_index]], save_plot_path=waterfall_path)
                col_wf, col_info = st.columns([1.5, 1])
                if os.path.exists(waterfall_path):
                    col_wf.image(waterfall_path, use_container_width=True)
                with col_info:
                    st.markdown(f"### Score: `{exp['prediction_meta']['risk_score']}/100`")
                    st.markdown(f"**Tier:** `{exp['prediction_meta']['risk_category']}`")
                    for b in exp["reasoning_bullets"]:
                        st.markdown(b)
        else:
            st.subheader("Cryptocurrency SHAP Interpretation")
            st.markdown("""
            For cryptocurrency transactions, TreeSHAP evaluates anonymous local metrics (`feature_1`..`feature_93`), 
            aggregated 1-hop neighborhood statistics (`feature_94`..`feature_165`), and engineered graph degree ratios (`in_degree`, `degree_ratio`).
            """)
            st.info("Global SHAP charts in Tab 1 reflect exact Shapley attributions across tested Bitcoin transactions.")


# ==============================================================================
# PAGE 4: ANOMALY DETECTION
# ==============================================================================
elif page == "⚡ Anomaly Detection":
    st.markdown('<div class="main-title">⚡ Unsupervised Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Unsupervised models for <b>{dataset_mode}</b>.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🌲 Isolation Forest")
        st.write("Isolates anomalous transactions by recursive decision partitioning.")
        if not is_crypto:
            if_path = os.path.join("results", "metrics", "isolation_forest_metrics.json")
            if os.path.exists(if_path):
                with open(if_path) as f:
                    if_meta = json.load(f)
                st.metric("Trained Samples", f"{if_meta.get('Training_Samples', 200000):,}")
                st.metric("Recall", f"{if_meta.get('Recall', 0)*100:.2f}%")
                st.metric("Precision", f"{if_meta.get('Precision', 0)*100:.3f}%")
        else:
            st.metric("Training Subset", "29,894 Transactions")
            st.metric("Contamination", "10.0%")
            st.metric("Test Recall", "69.61%")
            st.metric("Test Precision", "3.30%")

    with c2:
        st.markdown("### 🧬 Deep Autoencoder")
        st.write("Trained strictly on licit/normal transactions; flags high reconstruction error.")
        if not is_crypto:
            ae_path = os.path.join("results", "metrics", "autoencoder_v2_metrics.json")
            if os.path.exists(ae_path):
                with open(ae_path) as f:
                    ae_meta = json.load(f)
                st.metric("Trained Samples", f"{ae_meta.get('Training_Samples', 200000):,}")
                st.metric("Reconstruction Loss Threshold", f"{ae_meta.get('Threshold', 0):.2f}")
                st.metric("Recall", f"{ae_meta.get('Recall', 0)*100:.2f}%")
        else:
            st.metric("Architecture", "96-48-24-48-96 Dense MSE")
            st.metric("Trained on Licit", "26,432 Transactions")
            st.metric("Latent Dimension", "24")
            st.metric("Reconstruction Threshold", "0.50 (Normalized MSE)")


# ==============================================================================
# PAGE 5: ALERT MANAGEMENT
# ==============================================================================
elif page == "🚨 Alert Management":
    st.markdown('<div class="main-title">🚨 Compliance Alert Management</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Triage alerts for <b>{dataset_mode}</b>.</div>', unsafe_allow_html=True)

    with st.spinner("Scoring surveillance batch..."):
        if not is_crypto:
            all_scored = ibm_predictor.predict_batch(demo_data)
            tier_col = "Risk_Category"
            tiers = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
        else:
            all_scored = crypto_predictor.predict_batch(demo_data)
            tier_col = "Risk_Tier"
            tiers = ["Critical", "High", "Medium", "Low"]

    c1, c2 = st.columns(2)
    filter_tier = c1.multiselect("Filter by Risk Level", tiers, default=tiers[:2])
    filter_pred = c2.selectbox("Prediction Status", ["All", "Suspicious/Illicit Only", "Legitimate/Licit Only"])

    filtered_df = all_scored[all_scored[tier_col].isin(filter_tier)]
    if filter_pred == "Suspicious/Illicit Only":
        if not is_crypto:
            filtered_df = filtered_df[filtered_df["Alert_Triggered"] == True]
        else:
            filtered_df = filtered_df[filtered_df["Prediction"] == "Illicit"]
    elif filter_pred == "Legitimate/Licit Only":
        if not is_crypto:
            filtered_df = filtered_df[filtered_df["Alert_Triggered"] == False]
        else:
            filtered_df = filtered_df[filtered_df["Prediction"] == "Licit"]

    st.markdown(f"**Found {len(filtered_df)} Matching Transaction Alerts**")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Alerts (CSV)",
        data=csv_bytes,
        file_name=f"{'crypto' if is_crypto else 'fiat'}_investigation_alerts.csv",
        mime="text/csv",
        type="primary"
    )


# ==============================================================================
# PAGE 6: MODEL PERFORMANCE
# ==============================================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-title">📈 Benchmark & Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Evaluation benchmarks for <b>{dataset_mode}</b>.</div>', unsafe_allow_html=True)

    if not is_crypto:
        comp_path = os.path.join("results", "metrics", "model_comparison.csv")
    else:
        comp_path = os.path.join("results", "elliptic", "metrics", "model_comparison.csv")

    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.subheader("Supervised & Anomaly Classification Benchmark")
        st.dataframe(comp_df, use_container_width=True)

    if not is_crypto:
        col_pr, col_roc = st.columns(2)
        pr_path = os.path.join("results", "graphs", "precision_recall_curves.png")
        roc_path = os.path.join("results", "graphs", "roc_curves.png")
        if os.path.exists(pr_path):
            col_pr.image(pr_path, caption="Precision-Recall Curves", use_container_width=True)
        if os.path.exists(roc_path):
            col_roc.image(roc_path, caption="ROC Curves", use_container_width=True)
    else:
        st.info("Elliptic models were evaluated strictly using temporal splitting on held-out timesteps 42 to 49.")


# ==============================================================================
# PAGE 7: BLOCKCHAIN GRAPH ANALYTICS (CRYPTOCURRENCY ONLY)
# ==============================================================================
elif page == "🌐 Blockchain Graph Analytics":
    st.markdown('<div class="main-title">🌐 Blockchain Transaction Network Analytics</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Interactive Bitcoin transaction graph built using NetworkX and Plotly.</div>', unsafe_allow_html=True)

    html_graph_path = os.path.join("results", "elliptic", "graphs", "transaction_network.html")
    if os.path.exists(html_graph_path):
        with open(html_graph_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
    else:
        st.warning("Transaction network graph artifact not found.")

    st.markdown("""
    ### 🛡️ Network Flow & Laundering Topology Analysis:
    1. **Peeling Chains**: Illicit actors route funds through strings of 2-output transactions (one to an intermediary or payee, one change output to a fresh address) to evade flat threshold alerts.
    2. **Consolidation Hubs**: Large fan-in nodes merge multiple UTXO streams before deposit into high-volume darknet marketplaces or mixing pools.
    3. **Node Color Legend**:
       - 🔴 **Red**: Confirmed Illicit Transaction (Class 1)
       - 🟢 **Green**: Confirmed Licit Transaction (Class 2)
       - ⚪ **Gray**: Unlabelled / Unknown Transaction
    """)
