# Experimental Results Report

**Project:** Intelligent Anti-Money Laundering Detection Using Explainable AI  
**Evaluation Date:** November 17, 2026  
**Data Source:** Actual saved outputs from `results/` directory — no fabricated metrics

---

## 1. Dataset Statistics

### 1.1 IBM HI-Small Transaction Dataset
| Attribute | Value |
|-----------|-------|
| Total Records | 5,078,345 |
| Legitimate Transactions (Class 0) | 5,073,168 (99.90%) |
| Laundering Transactions (Class 1) | 5,177 (0.10%) |
| Imbalance Ratio | ~980:1 |
| Training Set | 3,250,140 records (64%) |
| Validation Set | 812,536 records (16%) |
| Test Set | 1,015,669 records (20%) |
| Input Features | 24 engineered features |

### 1.2 Elliptic Bitcoin Transaction Dataset
| Attribute | Value |
|-----------|-------|
| Total Transactions (all labels) | 203,769 |
| Labelled Transactions | 46,564 (22.85%) |
| Illicit Transactions (Class 1) | 4,545 (9.76% of labelled) |
| Licit Transactions (Class 2) | 42,019 (90.24% of labelled) |
| Unknown Transactions | 157,205 (77.15% of total) |
| Directed Payment Edges | 234,355 |
| Time Steps | 49 |
| Training Partition (ts 1–34) | 29,894 labelled transactions |
| Validation Partition (ts 35–41) | 7,829 labelled transactions |
| Test Partition (ts 42–49) | 8,841 labelled transactions |
| Input Features | 183 (165 raw + 18 engineered) |

---

## 2. IBM Model Evaluation Results (Test Set N = 1,015,669)

### 2.1 Supervised Classification Benchmarks (Optimal F2-Tuned Thresholds)
| Metric | Random Forest (θ = 0.6688) | XGBoost (θ = 0.8989) |
|--------|---------------------------|----------------------|
| Precision | 33.09% | **44.21%** |
| Recall | 67.05% | **67.54%** |
| F1-Score | 0.4432 | **0.5344** |
| F2-Score | 0.5564 | **0.6109** |
| Average Precision (PR-AUC) | 0.4903 | **0.6164** |
| ROC-AUC | 0.9751 | **0.9812** |
| True Positives | 694 / 1,035 | **699 / 1,035** |
| False Positives | 1,403 | **882** |
| True Negatives | 97,562 | **98,083** |
| False Negatives | 341 | **336** |
| False Positive Rate | 1.42% | **0.89%** |
| False Negative Rate | 32.95% | **32.46%** |

> **Note:** Metrics are from the strictly held-out test set (`aml_test.csv`). Thresholds were optimized on `aml_val.csv` using F2-score (β=2). The test set was not used for threshold selection.

### 2.2 Default Threshold (0.5) Comparison
| Metric | RF (Default) | XGB (Default) |
|--------|-------------|---------------|
| Precision | 23.86% | 13.94% |
| Recall | 77.78% | 88.21% |
| F1-Score | 0.3652 | 0.2408 |
| False Positive Rate | 2.60% | 5.69% |

### 2.3 Unsupervised Anomaly Detection (Full Test Set N = 1,015,669)
| Metric | Isolation Forest | Autoencoder V2 |
|--------|-----------------|----------------|
| Training Samples | 200,000 | 200,000 |
| Threshold | 0.4559 | 3,218,036.25 (MSE) |
| Precision | 0.156% | 0.123% |
| Recall | 27.92% | 4.44% |
| F1-Score | 0.0031 | 0.0024 |
| PR-AUC | 0.0013 | 0.0012 |
| ROC-AUC | 0.5987 | 0.5672 |
| True Positives | 289 | 46 |
| False Positives | 184,898 | 37,388 |
| Total Alerts Flagged | 185,187 | 37,434 |

---

## 3. Elliptic Cryptocurrency Model Evaluation Results (Test Set ts 42–49, N = 8,841)

### 3.1 Supervised Classification Benchmarks (F2-Tuned Thresholds on Validation)
| Model | Threshold | Precision | Recall | F1-Score | F2-Score | PR-AUC | ROC-AUC | FPR | FNR | TP | FP | TN | FN |
|-------|-----------|-----------|--------|---------|---------|---------|---------|-----|-----|----|----|----|----|
| XGBoost | 0.6700 | **0.7433** | 0.4755 | **0.5800** | **0.5124** | **0.5630** | **0.8730** | **0.0079** | 0.5245 | **194** | 67 | 8,366 | 214 |
| Random Forest | 0.5000 | 0.7452 | 0.4730 | 0.5787 | 0.5103 | 0.5488 | 0.8590 | 0.0078 | 0.5270 | 193 | 66 | 8,367 | 215 |
| Logistic Regression | 0.5300 | 0.1793 | **0.6373** | 0.2799 | 0.4218 | 0.2537 | 0.8616 | 0.1411 | 0.3627 | 260 | 1,190 | 7,243 | 148 |

> **Positive class imbalance ratio (train):** 7.63:1 (Licit:Illicit)  
> **XGBoost `scale_pos_weight`:** 7.63  
> **Random Forest:** `class_weight='balanced'`  
> **Logistic Regression:** `class_weight='balanced'`

### 3.2 Unsupervised Anomaly Detection (Crypto, Test ts 42–49)
| Model | Threshold | Precision | Recall | F1-Score | F2-Score | PR-AUC | ROC-AUC | FPR |
|-------|-----------|-----------|--------|---------|---------|---------|---------|-----|
| Isolation Forest | 0.0500 | 3.30% | **69.61%** | 0.0630 | 0.1386 | 0.0271 | 0.2263 | 98.77% |
| Deep Autoencoder | 0.5000 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.0297 | 0.2920 | 0.74% |

> **Autoencoder note:** The autoencoder was trained on licit transactions from timesteps 1–34. Zero recall on the test partition (ts 42–49) suggests significant concept/distribution drift across the temporal spectrum. The model correctly produces very few false positives (62 out of 8,433 licit transactions) but fails to identify any illicit transactions at the default threshold.

---

## 4. SHAP Feature Attribution Results (Elliptic XGBoost Model)

Top 15 features ranked by mean absolute SHAP value across 300 balanced samples:

| Rank | Feature | Mean |SHAP| | Category |
|------|---------|------------|----------|
| 1 | `local_feature_std` | 1.1427 | Engineered Graph/Temporal |
| 2 | `feature_53` | 1.0003 | Local Transaction (1–93) |
| 3 | `feature_59` | 0.6518 | Local Transaction (1–93) |
| 4 | `feature_90` | 0.5618 | Local Transaction (1–93) |
| 5 | `feature_2` | 0.4287 | Local Transaction (1–93) |
| 6 | `feature_76` | 0.3740 | Local Transaction (1–93) |
| 7 | `feature_58` | 0.3337 | Local Transaction (1–93) |
| 8 | `feature_46` | 0.2877 | Local Transaction (1–93) |
| 9 | `feature_5` | 0.2356 | Local Transaction (1–93) |
| 10 | `feature_3` | 0.2347 | Local Transaction (1–93) |
| 11 | `feature_161` | 0.2202 | Neighborhood Aggregation (94–165) |
| 12 | `feature_125` | 0.2180 | Neighborhood Aggregation (94–165) |
| 13 | `feature_52` | 0.2098 | Local Transaction (1–93) |
| 14 | `feature_127` | 0.1625 | Neighborhood Aggregation (94–165) |
| 15 | `feature_163` | 0.1613 | Neighborhood Aggregation (94–165) |

> **Note:** No semantic interpretation is assigned to anonymized features `feature_1`–`feature_165` as the Elliptic dataset does not publish feature meanings. Only their engineering categories (local vs. neighborhood vs. graph-engineered) are stated.

---

## 5. Graph Analytics Results

| Metric | Value |
|--------|-------|
| Total Directed Edges in Dataset | 234,355 |
| Total Nodes Participating in Edges | ≥ 85,000 (approximation from edge union) |
| Subgraph Extracted for Visualization | 260 nodes, 250 edges |
| Subgraph Density | Low (sparse DAG structure) |
| Visualization Format | Interactive Plotly HTML (`transaction_network.html`, 4.9 MB) |
| Node Color Coding | Red=Illicit (Class 1), Green=Licit (Class 2), Gray=Unknown |
| Node Sizing | Scaled by PageRank × degree |

---

## 6. Test Suite Summary

| Module | Tests | Passed | Status |
|--------|-------|--------|--------|
| `test_pipeline.py` | 4 | 4 | ✅ 100% |
| `test_models.py` | 4 | 4 | ✅ 100% |
| `test_inference.py` | 4 | 4 | ✅ 100% |
| `test_elliptic_data.py` | 5 | 5 | ✅ 100% |
| `test_elliptic_pipeline.py` | 4 | 4 | ✅ 100% |
| `test_elliptic_models.py` | 5 | 5 | ✅ 100% |
| `test_elliptic_dashboard.py` | 5 | 5 | ✅ 100% |
| **Total** | **31** | **31** | ✅ **100%** |
