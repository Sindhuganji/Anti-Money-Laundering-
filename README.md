# Intelligent Anti-Money Laundering (AML) Detection System Using Machine Learning and Explainable AI

A production-grade, explainable machine learning platform engineered to detect illicit financial patterns across both **Traditional Banking (IBM HI-Small Dataset)** and **Cryptocurrency Blockchain Networks (Elliptic Bitcoin Dataset)** under severe class imbalance.

---

## 🏛️ System Architecture

```
AML_Project/
├── data/
│   ├── raw/HI-Small_Trans.csv             # IBM AML Transaction dataset
│   ├── processed/                         # Fiat preprocessed and engineered data splits
│   └── elliptic/
│       ├── raw/                           # Raw Elliptic Bitcoin classes, features & edgelist
│       └── processed/                     # Engineered cryptocurrency features & splits
├── models/
│   ├── aml_pipeline_*                     # Fiat preprocessing scalers & frequency encoders
│   ├── xgboost_model.pkl                  # Fiat XGBoost model (Threshold: 0.8989)
│   ├── random_forest_model.pkl            # Fiat Random Forest model (Threshold: 0.6688)
│   ├── isolation_forest_model.pkl         # Fiat Isolation Forest anomaly detector
│   ├── aml_autoencoder_v2.keras           # Fiat Deep Autoencoder
│   └── elliptic/
│       ├── elliptic_scaler.pkl            # Crypto RobustScaler
│       ├── elliptic_xgboost.pkl           # Crypto XGBoost model (Threshold: 0.6700)
│       ├── elliptic_random_forest.pkl     # Crypto Random Forest model (Threshold: 0.5000)
│       ├── elliptic_logistic_regression.pkl (Threshold: 0.5300)
│       ├── elliptic_isolation_forest.pkl  (Threshold: 0.0500)
│       └── elliptic_autoencoder.keras     # Crypto Deep Autoencoder (Threshold: 0.5000)
├── src/                                   # Production codebase
│   ├── preprocessing.py, feature_engineering.py, predict.py, explainability.py (Fiat)
│   └── elliptic/
│       ├── data_acquisition.py            # Phase 1 validation & integrity checks
│       ├── feature_engineering.py         # Graph degrees & temporal transformations
│       ├── train.py                       # Model training with temporal split
│       ├── explainability.py              # Crypto TreeSHAP global & local attributions
│       ├── risk_scoring.py                # Composite 0-100 risk scoring
│       ├── graph_analysis.py              # NetworkX topological DAG analysis
│       └── predict.py                     # Unified cryptocurrency inference engine
├── app/
│   └── streamlit_app.py                   # Multi-Domain Interactive Streamlit Dashboard
├── results/
│   ├── graphs/ & metrics/                 # Fiat evaluation charts & metrics
│   └── elliptic/                          # Crypto evaluation metrics, SHAP plots & network HTML
├── tests/                                 # 31 Comprehensive unit tests (100% passing)
│   ├── test_pipeline.py, test_models.py, test_inference.py (Fiat)
│   └── test_elliptic_data.py, test_elliptic_pipeline.py, test_elliptic_models.py, test_elliptic_dashboard.py
├── ELLIPTIC_DATASET_REPORT.md             # Dataset acquisition & schema verification report
├── ELLIPTIC_MODEL_EVALUATION.md           # Comparative model performance report
├── ELLIPTIC_SHAP_REPORT.md                # Explainable AI & feature attribution report
├── ELLIPTIC_GRAPH_ANALYSIS.md             # Graph topology & payment flow analysis report
├── FINAL_PROJECT_REPORT.md                # Consolidated multi-domain system report
└── requirements.txt
```

---

## ⚡ Key Highlights & Methodology

1. **Dual Independent Surveillance Domains:**
   - **Mode 1: Traditional Banking (IBM HI-Small)**: Evaluates accounts, inter-bank transfers, currencies, and velocity metrics across 5M+ records.
   - **Mode 2: Cryptocurrency (Elliptic Bitcoin)**: Evaluates 203,769 Bitcoin transactions across 49 discrete timesteps, integrating directed flow edges, graph degrees, and 1-hop neighborhood aggregations.
2. **Zero Target Leakage & Strict Temporal Splitting:**
   - For the Elliptic dataset, training is restricted to Timesteps 1–34, validation to Timesteps 35–41, and evaluation to Timesteps 42–49. Future transactions never leak into the past.
   - Unlabelled/unknown transactions (157,205 records) are strictly excluded from supervised ground truth.
3. **Threshold Optimization ($F_2$-Score):**
   - Thresholds are tuned on validation splits favoring minority recall to catch high-risk laundering operations before applying to test sets.
4. **Transparent Explainability:**
   - TreeSHAP attributions for both XGBoost and Random Forest, identifying peeling chains, consolidation funnels, and abnormal velocity spikes.
5. **Interactive Graph Analytics:**
   - Embedded NetworkX & Plotly interactive DAG network visualizing illicit, licit, and unknown transaction clusters.

---

## 📊 Model Performance Comparison (Elliptic Test Set: Timesteps 42–49)

| Model | Type | Optimal Threshold | Precision | Recall | F1-Score | F2-Score | PR-AUC | ROC-AUC | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Supervised | 0.6700 | **0.7433** | 0.4755 | **0.5800** | **0.5124** | **0.5630** | **0.8730** | **0.0079** |
| **Random Forest** | Supervised | 0.5000 | 0.7452 | 0.4730 | 0.5787 | 0.5103 | 0.5488 | 0.8590 | 0.0078 |
| **Logistic Regression**| Supervised | 0.5300 | 0.1793 | 0.6373 | 0.2799 | 0.4218 | 0.2537 | 0.8616 | 0.1411 |
| **Isolation Forest** | Unsupervised | 0.0500 | 0.0330 | **0.6961** | 0.0630 | 0.1386 | 0.0271 | 0.2263 | 0.9877 |
| **Deep Autoencoder** | Unsupervised | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0297 | 0.2920 | 0.0074 |

---

## 🚀 Execution & Setup Instructions

### 1. Installation
```bash
# Clone the repository
cd AML_Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run All Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Retrain / Evaluate Pipelines
```bash
# Traditional Banking (IBM)
python src/train.py

# Cryptocurrency (Elliptic Bitcoin)
python src/elliptic/data_acquisition.py
python src/elliptic/feature_engineering.py
python src/elliptic/train.py
python src/elliptic/explainability.py
python src/elliptic/graph_analysis.py
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app/streamlit_app.py
```
Toggle the **Active Surveillance Domain** in the sidebar between:
- `Traditional Banking (IBM HI-Small)`
- `Cryptocurrency (Elliptic Bitcoin)`
