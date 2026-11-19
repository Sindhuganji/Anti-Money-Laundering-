# Final Project Report: Intelligent Anti-Money Laundering Detection Using Explainable AI

## 1. Executive Overview
This document delivers the final, consolidated technical report for the **Intelligent Anti-Money Laundering (AML) Detection Platform with Explainable AI**. The system has been architected to provide enterprise-grade, multi-domain financial crime surveillance across both:
1. **Traditional Banking (IBM HI-Small Transaction Dataset)**: Tabular fiat payment records encompassing accounts, banks, timestamps, currencies, and payment formats.
2. **Cryptocurrency Monitoring (Elliptic Bitcoin Transaction Dataset)**: Blockchain Directed Acyclic Graph (DAG) payment flows spanning 49 discrete timesteps, featuring local transaction attributes, aggregated 1-hop graph neighborhood properties, and topological connectivity measures.

---

## 2. Architectural Blueprint & Pipeline Separation

The AML platform guarantees strict decoupling between the fiat banking pipeline and the cryptocurrency graph pipeline. No features or scalers from one domain ever leak into or cross-contaminate the other:

```
AML_Project/
├── app/
│   └── streamlit_app.py                # Dual-Domain Streamlit Web Application
├── data/
│   ├── raw/HI-Small_Trans.csv          # Fiat banking transactions
│   ├── processed/                      # Preprocessed & engineered fiat datasets
│   └── elliptic/
│       ├── raw/                        # elliptic_txs_classes.csv, features, edgelist
│       └── processed/                  # elliptic_features_engineered.csv, labelled, unknown
├── models/
│   ├── aml_pipeline_*                  # Fiat preprocessing scalers & frequency encoders
│   ├── xgboost_model.pkl               # Fiat XGBoost model (Threshold: 0.8989)
│   ├── random_forest_model.pkl         # Fiat Random Forest model (Threshold: 0.6688)
│   └── elliptic/
│       ├── elliptic_scaler.pkl         # Crypto RobustScaler
│       ├── elliptic_xgboost.pkl        # Crypto XGBoost model (Threshold: 0.6700)
│       ├── elliptic_random_forest.pkl  # Crypto Random Forest model (Threshold: 0.5000)
│       ├── elliptic_logistic_regression.pkl (Threshold: 0.5300)
│       ├── elliptic_isolation_forest.pkl (Threshold: 0.0500)
│       └── elliptic_autoencoder.keras  # Crypto Deep Autoencoder (Threshold: 0.5000)
├── results/
│   ├── graphs/                         # Fiat SHAP and ROC/PR visualizations
│   ├── metrics/                        # Fiat model benchmarks
│   └── elliptic/
│       ├── graphs/                     # Crypto SHAP bar/beeswarm & transaction_network.html
│       ├── metrics/                    # Crypto model comparative metrics
│       └── reports/                    # Temporal distributions & class summaries
└── src/
    ├── preprocessing.py, feature_engineering.py, predict.py, explainability.py (Fiat)
    └── elliptic/
        ├── data_acquisition.py         # Phase 1 validation & integrity checks
        ├── feature_engineering.py      # Graph metrics & temporal transformations
        ├── train.py                    # Model training & temporal validation
        ├── explainability.py           # Crypto TreeSHAP attributions
        ├── risk_scoring.py             # Composite 0-100 risk scoring
        ├── graph_analysis.py           # NetworkX topological DAG visualization
        └── predict.py                  # Crypto unified inference engine
```

---

## 3. Dataset Characteristics & Integrity

### Traditional Banking (IBM HI-Small)
- Total transactions analyzed: 5,078,345 records.
- Feature set: 24 engineered features (temporal hours/days, foreign currency flags, inter-bank transfers, sender/receiver historical frequency encodings, and transaction volume ratios).
- Class imbalance: Severe minority laundering rate (~0.1%).

### Cryptocurrency (Elliptic Bitcoin Transaction Graph)
- **Total transactions**: 203,769 transactions across 49 timesteps.
- **Labelled ground truth**: 46,564 transactions:
  - Illicit (Class 1): 4,545 (9.76% of labelled)
  - Licit (Class 2): 42,019 (90.24% of labelled)
- **Unknown (Unlabelled)**: 157,205 transactions (77.15% of all nodes).
  - *Compliance Rule Enforced*: Unknown transactions were **strictly excluded** from supervised ground-truth training and preserved solely for graph topology and anomaly scoring.
- **Graph connectivity**: 234,356 directed payment flow edges.

---

## 4. Rigorous Machine Learning Evaluation

All models were evaluated using strict **chronological/temporal splits** to ensure zero lookahead bias:
- **Crypto Training**: Timesteps 1–34 (29,894 transactions)
- **Crypto Validation**: Timesteps 35–41 (7,829 transactions, utilized for $F_2$ optimal threshold selection)
- **Crypto Held-Out Test Set**: Timesteps 42–49 (8,841 unseen future transactions; 408 Illicit, 8,433 Licit)

### Cryptocurrency Model Performance Benchmark (Test Partition: Timesteps 42–49)
| Model | Type | Optimal Threshold | Precision | Recall | F1-Score | F2-Score | PR-AUC | ROC-AUC | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Supervised | 0.6700 | **0.7433** | 0.4755 | **0.5800** | **0.5124** | **0.5630** | **0.8730** | **0.0079** (67 / 8,433) |
| **Random Forest** | Supervised | 0.5000 | 0.7452 | 0.4730 | 0.5787 | 0.5103 | 0.5488 | 0.8590 | 0.0078 (66 / 8,433) |
| **Logistic Regression**| Supervised | 0.5300 | 0.1793 | 0.6373 | 0.2799 | 0.4218 | 0.2537 | 0.8616 | 0.1411 (1,190 / 8,433) |
| **Isolation Forest** | Unsupervised | 0.0500 | 0.0330 | **0.6961** | 0.0630 | 0.1386 | 0.0271 | 0.2263 | 0.9877 (8,329 / 8,433) |
| **Deep Autoencoder** | Unsupervised | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0297 | 0.2920 | 0.0074 (62 / 8,433) |

*Key Analytical Insight*: Supervised gradient boosting (**XGBoost**) and ensemble trees (**Random Forest**) vastly outperform unsupervised baselines by exploiting the rich 1-hop neighborhood aggregated embeddings and engineered degree ratios. They maintain an exceptionally low false positive rate of **<0.8%**, critical for compliance analysts facing alert fatigue.

---

## 5. Explainable AI & SHAP Attributions

Global feature attributions were calculated via TreeSHAP across 183 input dimensions (165 raw + 18 engineered):
1. **Network Centrality & Degree Metrics**: `in_degree`, `out_degree`, `degree_ratio`, and `total_degree` emerge among the top 10 most influential features, validating that rapid splitting (peeling chains) and fan-in pooling are principal crypto laundering signatures.
2. **Neighborhood vs Local Attributes**: `neighbor_feature_mean` and specific neighborhood aggregations in the 94–165 index range demonstrated higher mean absolute SHAP values than isolated single-transaction metrics.
3. **Artifacts Preserved**:
   - `results/elliptic/graphs/shap_global_bar_xgboost.png`
   - `results/elliptic/graphs/shap_global_beeswarm_xgboost.png`
   - `results/elliptic/graphs/shap_global_bar_random_forest.png`
   - `results/elliptic/graphs/shap_global_beeswarm_random_forest.png`

---

## 6. Risk Scoring & Compliance Triage

Risk scores are calibrated into an intuitive 0–100 integer range with 4 compliance tiers:
- **Low Risk (0–29)**: Standard commercial or retail transfers; automatic clearing.
- **Medium Risk (30–69)**: Moderate anomaly score; standard audit log queue.
- **High Risk (70–89)**: Elevated probability; queued for Level-1 compliance review.
- **Critical Risk (90–100)**: Immediate escalation to Suspicious Activity Report (SAR) filing.

Composite calculation:
$$\text{Base Score} = \text{Clamp}(P_{\text{illicit}}, 0, 1) \times 100$$
$$\text{Network Penalty} = \min(5.0, \text{total\_degree} \times 0.2) + (3.0 \text{ if } \text{degree\_ratio} \notin [0.15, 0.85] \text{ else } 0)$$
$$\text{Final Risk Score} = \text{Clamp}(\text{Base Score} + \text{Network Penalty}, 0, 100)$$

---

## 7. Interactive Streamlit Surveillance Application

The unified web dashboard seamlessly toggles between:
- **Mode 1**: Traditional Banking (IBM HI-Small)
- **Mode 2**: Cryptocurrency (Elliptic Bitcoin)

### Features Implemented:
1. **Executive Dashboard**: Real-time KPI counters, classification donuts, risk tier bar breakdowns, and scatter profiling.
2. **Transaction Analysis**: Full batch CSV upload pipeline and single transaction evaluator with natural language decision reasoning.
3. **Explainable AI (SHAP)**: Dynamic switching between XGBoost and Random Forest with model-specific bar and beeswarm plots.
4. **Anomaly Detection**: Comparative evaluation of Isolation Forest and Deep Autoencoders.
5. **Alert Management**: Risk-tier filtering, prediction triage, and one-click CSV alert bundle export.
6. **Model Performance**: Full benchmark scorecards with ROC and PR curves.
7. **Blockchain Graph Analytics**: Embedded NetworkX/Plotly 2D interactive transaction network graph with color-coded ground truth classes and PageRank sizing.

---

## 8. Verification & Test Suite Summary
The entire test suite passes cleanly with 31 comprehensive unit tests:
- `test_elliptic_data.py`: 5 tests passed (file validation, schemas, row counts).
- `test_elliptic_pipeline.py`: 4 tests passed (metadata, feature alignment, target leakage).
- `test_elliptic_models.py`: 5 tests passed (model inference, threshold logic, risk scoring).
- `test_elliptic_dashboard.py`: 5 tests passed (Streamlit UI AppTests, multi-domain toggling).
- `test_inference.py`: 4 tests passed (IBM inference pipeline).
- `test_models.py`: 4 tests passed (IBM models & thresholds).
- `test_pipeline.py`: 4 tests passed (IBM feature engineering & scaling).

**Total Test Count**: 31 / 31 Passed (100% Success).
