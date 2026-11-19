# Implementation Status Report

**Project:** Intelligent Anti-Money Laundering Detection Using Explainable AI  
**Audit Date:** November 17, 2026  
**Repository:** `/Users/pushpasrisindhuganji/AML_Project`

---

## 1. Classification Key

| Symbol | Status |
|--------|--------|
| ✅ | Implemented and verified (tests passed / output confirmed) |
| ⚠️ | Implemented but not fully tested |
| 🔶 | Partially implemented |
| ❌ | Planned but not implemented |

---

## 2. MODE 1 — Traditional Banking AML (IBM HI-Small)

### 2.1 Data Pipeline
| Feature | Status | Evidence |
|---------|--------|----------|
| Raw data loading (`HI-Small_Trans.csv`) | ✅ | `data/raw/HI-Small_Trans.csv` exists; 5,078,345 records |
| Schema validation | ✅ | `src/data_loader.py` |
| Missing value imputation (Median `SimpleImputer`) | ✅ | `models/aml_pipeline_imputer.pkl` exists; metadata confirms `impute_strategy: median` |
| RobustScaler normalization | ✅ | `models/aml_pipeline_scaler.pkl` exists; metadata confirms `scaler_type: robust` |
| Stratified train/val/test split (64/16/20%) | ✅ | Train: 3,250,140; Val: 812,536; Test: 1,015,669 records confirmed |
| Frequency encoding of categorical columns | ✅ | `models/aml_pipeline_freq_encoders.json` exists |
| Duplicate transaction detection | ✅ | `duplicate_check.db` created during preprocessing |
| SMOTE oversampling | ❌ | Not implemented; class imbalance handled via `class_weight='balanced'` |

### 2.2 Feature Engineering (24 Features)
| Feature | Status |
|---------|--------|
| `Amount Received`, `Amount Paid` | ✅ |
| `Amount_Difference`, `Absolute_Amount_Difference` | ✅ |
| `Amount_Ratio` | ✅ |
| `Log_Amount_Received`, `Log_Amount_Paid` | ✅ |
| `Transaction_Hour`, `Transaction_Day`, `Transaction_Month`, `Transaction_Weekday` | ✅ |
| `Is_Weekend`, `Is_Night_Transaction` | ✅ |
| `Same_Currency`, `Same_Bank` | ✅ |
| `Source_Transaction_Frequency`, `Destination_Transaction_Frequency` | ✅ |
| `Previous_Source_Transactions`, `Previous_Source_Amount` | ✅ |
| `Previous_Destination_Transactions`, `Previous_Destination_Amount` | ✅ |
| `Receiving Currency_Frequency`, `Payment Currency_Frequency`, `Payment Format_Frequency` | ✅ |

### 2.3 Supervised ML Models
| Model | Status | Threshold | F1-Score | PR-AUC | ROC-AUC |
|-------|--------|-----------|---------|---------|---------|
| Random Forest (`class_weight='balanced'`) | ✅ | 0.6688 (F2-optimized) | 0.4432 | 0.4903 | 0.9751 |
| XGBoost (`scale_pos_weight` tuned) | ✅ | 0.8989 (F2-optimized) | 0.5344 | 0.6164 | 0.9812 |

### 2.4 Unsupervised Anomaly Detection
| Model | Status | Recall | Precision | ROC-AUC |
|-------|--------|--------|-----------|---------|
| Isolation Forest | ✅ | 27.92% | 0.156% | 0.5987 |
| Deep Autoencoder V2 | ✅ | 4.44% | 0.123% | 0.5672 |

### 2.5 Explainable AI
| Feature | Status | Evidence |
|---------|--------|----------|
| SHAP TreeExplainer for XGBoost | ✅ | `results/graphs/shap_global_bar_xgboost.png` |
| SHAP TreeExplainer for Random Forest | ✅ | `results/graphs/shap_global_bar_random_forest.png` |
| Global bar plots | ✅ | Both model-specific PNGs exist |
| Global beeswarm plots | ✅ | Both model-specific PNGs exist |
| Local waterfall explanation | ✅ | `results/graphs/interactive_waterfall.png` |
| Natural language reasoning bullets | ✅ | `src/explainability.py` |

### 2.6 Risk Scoring
| Feature | Status |
|---------|--------|
| 0–100 risk score calculation | ✅ |
| Low / Medium / High / Critical tiers | ✅ |
| Alert triggered flag | ✅ |

---

## 3. MODE 2 — Cryptocurrency AML (Elliptic Bitcoin Dataset)

### 3.1 Dataset Acquisition & Validation
| Feature | Status | Evidence |
|---------|--------|----------|
| `elliptic_txs_features.csv` downloaded | ✅ | 658 MB; 203,769 rows |
| `elliptic_txs_classes.csv` downloaded | ✅ | 3.2 MB; 203,769 rows |
| `elliptic_txs_edgelist.csv` downloaded | ✅ | 4.3 MB; 234,355 directed edges |
| File integrity validation script | ✅ | `src/elliptic/data_acquisition.py` |
| Cross-file ID consistency verification | ✅ | 0 mismatches confirmed |
| `ELLIPTIC_DATASET_REPORT.md` generated | ✅ | Root directory |

### 3.2 Data Preprocessing
| Feature | Status | Evidence |
|---------|--------|----------|
| Memory-optimized `float32` loading | ✅ | Chunked reading in `feature_engineering.py` |
| RobustScaler (fit on train only) | ✅ | `models/elliptic/elliptic_scaler.pkl` |
| Strict temporal split (Train: ts 1–34, Val: ts 35–41, Test: ts 42–49) | ✅ | `src/elliptic/train.py` |
| Exclusion of unknown labels from supervised training | ✅ | `df_labelled` file uses only targets {0, 1} |
| `elliptic_labelled.csv` (46,564 records) | ✅ | `data/elliptic/processed/` |
| `elliptic_unknown.csv` (157,205 records) | ✅ | `data/elliptic/processed/` |

### 3.3 Graph Feature Engineering (183 Total Features)
| Feature Group | Status |
|---------------|--------|
| Original 165 anonymized features (`feature_1`..`feature_165`) | ✅ |
| Graph: `in_degree`, `out_degree`, `total_degree` | ✅ |
| Graph: `degree_ratio`, `degree_diff` | ✅ |
| Temporal: `time_step_norm`, `time_step_sin`, `time_step_cos` | ✅ |
| Local stats: `local_feature_mean`, `std`, `skew`, `max`, `min` | ✅ |
| Neighborhood stats: `neighbor_feature_mean`, `std`, `max`, `min` | ✅ |
| Cross-domain: `local_vs_neighbor_mean_ratio` | ✅ |
| NetworkX graph construction for degree computation | ✅ |
| Full NetworkX pagerank, betweenness centrality | ⚠️ | Used in `graph_analysis.py` subgraph only, not as ML features |

### 3.4 Supervised ML Models (Crypto)
| Model | Status | Threshold | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|-------|--------|-----------|-----------|--------|---------|---------|---------|
| Logistic Regression (`class_weight='balanced'`) | ✅ | 0.53 | 0.1793 | 0.6373 | 0.2799 | 0.2537 | 0.8616 |
| Random Forest (`class_weight='balanced'`) | ✅ | 0.50 | 0.7452 | 0.4730 | 0.5787 | 0.5488 | 0.8590 |
| XGBoost (`scale_pos_weight=7.63`) | ✅ | 0.67 | 0.7433 | 0.4755 | 0.5800 | 0.5630 | 0.8730 |

### 3.5 Unsupervised Anomaly Detection (Crypto)
| Model | Status | Threshold | Recall | F1-Score |
|-------|--------|-----------|--------|---------|
| Isolation Forest (`contamination=0.10`) | ✅ | 0.05 | 69.61% | 0.0630 |
| Deep Autoencoder (96-48-24-48-96 Dense) | ✅ | 0.50 | 0.00% | 0.0000 |

> **Note on Autoencoder:** The autoencoder trained successfully on licit transactions but produced zero recall on the test partition (timesteps 42–49). This is likely due to distribution shift across temporal segments. The autoencoder is preserved as a completed artifact but its discriminative performance on this dataset is poor.

### 3.6 Explainable AI (Crypto)
| Feature | Status | Evidence |
|---------|--------|----------|
| SHAP TreeExplainer for XGBoost | ✅ | `results/elliptic/graphs/shap_global_bar_xgboost.png` |
| SHAP TreeExplainer for Random Forest | ✅ | `results/elliptic/graphs/shap_global_bar_random_forest.png` |
| Global SHAP beeswarm plots (both models) | ✅ | Confirmed |
| Local waterfall explanation (Crypto) | ❌ | Not implemented — only IBM has individual waterfall |
| `ELLIPTIC_SHAP_REPORT.md` | ✅ | Root directory |

### 3.7 Risk Scoring (Crypto)
| Feature | Status |
|---------|--------|
| `calculate_crypto_risk_score()` with network penalty | ✅ |
| `assign_crypto_risk_tier()` — 4 tiers | ✅ |
| `format_crypto_risk_assessment()` | ✅ |
| Unknown transaction flag | ✅ |
| Compliance disclaimer | ✅ |

### 3.8 Graph Analytics
| Feature | Status | Evidence |
|---------|--------|----------|
| NetworkX DiGraph construction | ✅ | `src/elliptic/graph_analysis.py` |
| Degree centrality | ✅ | In-degree / out-degree computed |
| PageRank computation | ✅ | Used for node sizing in visualization |
| Spring layout 2D positions | ✅ | `nx.spring_layout()` |
| Interactive Plotly network HTML | ✅ | `results/elliptic/graphs/transaction_network.html` (4.9 MB) |
| Node color-coding by ground truth class | ✅ | Red=Illicit, Green=Licit, Gray=Unknown |
| Community detection | ❌ | Not implemented |
| Betweenness centrality as ML feature | ❌ | Not implemented |
| `ELLIPTIC_GRAPH_ANALYSIS.md` | ✅ | Root directory |

---

## 4. Unified Streamlit Dashboard

| Page | Status | Notes |
|------|--------|-------|
| Dataset domain selector (IBM / Elliptic) | ✅ | Sidebar selectbox, full isolation |
| Executive Dashboard | ✅ | KPIs, distribution charts, risk tier breakdowns |
| Transaction Analysis — CSV Batch Upload | ✅ | Fully connected to active pipeline |
| Transaction Analysis — Single Transaction (IBM) | ✅ | Full raw transaction evaluator |
| Transaction Analysis — Single Transaction (Crypto) | ✅ | Row selector from processed dataset |
| Explainable AI (SHAP) — Global Plots | ✅ | Model-specific, domain-specific paths |
| Explainable AI (SHAP) — Local Waterfall (IBM) | ✅ | Generates on-demand |
| Explainable AI (SHAP) — Local Waterfall (Crypto) | ❌ | Not implemented — info message shown |
| Anomaly Detection | ✅ | Metrics shown for both domains |
| Alert Management | ✅ | Filtering, download, domain-specific columns |
| Model Performance | ✅ | Benchmark metrics tables + IBM charts |
| Blockchain Graph Analytics | ✅ | Embedded HTML iframe of transaction network |

---

## 5. Testing

| Test Module | Status | Tests Passed |
|-------------|--------|-------------|
| `test_pipeline.py` | ✅ | 4 / 4 |
| `test_models.py` | ✅ | 4 / 4 |
| `test_inference.py` | ✅ | 4 / 4 |
| `test_elliptic_data.py` | ✅ | 5 / 5 |
| `test_elliptic_pipeline.py` | ✅ | 4 / 4 |
| `test_elliptic_models.py` | ✅ | 5 / 5 |
| `test_elliptic_dashboard.py` | ✅ | 5 / 5 |
| **Total** | **✅** | **31 / 31** |
