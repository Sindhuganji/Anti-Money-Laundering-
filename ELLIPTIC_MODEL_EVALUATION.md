# Elliptic Bitcoin AML Model Evaluation Report

## Executive Summary
This report presents the rigorous evaluation of five machine learning models trained for Cryptocurrency Anti-Money Laundering detection on the Elliptic Bitcoin Transaction Dataset. 

Strict **temporal splitting** was enforced:
- **Training partition**: Timesteps 1–34 (29,894 transactions)
- **Validation partition**: Timesteps 35–41 (7,829 transactions)
- **Hold-out Test partition**: Timesteps 42–49 (8,841 transactions)

Classification decision thresholds were strictly tuned on the validation set using the $F_2$-score (favoring minority illicit transaction recall) and evaluated on unseen future transactions.

---

## 1. Comparative Model Performance (Hold-Out Test Set)

| Model | Optimal Threshold | Precision | Recall | F1-Score | F2-Score | PR-AUC | ROC-AUC | FPR | FNR | TP | FP | TN | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.53 | 0.1793 | 0.6373 | 0.2799 | 0.4218 | 0.2537 | 0.8616 | 0.1411 | 0.3627 | 260 | 1190 | 7243 | 148 |
| **Random Forest** | 0.5 | 0.7452 | 0.473 | 0.5787 | 0.5103 | 0.5488 | 0.859 | 0.0078 | 0.527 | 193 | 66 | 8367 | 215 |
| **XGBoost** | 0.67 | 0.7433 | 0.4755 | 0.58 | 0.5124 | 0.563 | 0.873 | 0.0079 | 0.5245 | 194 | 67 | 8366 | 214 |
| **Isolation Forest** | 0.05 | 0.033 | 0.6961 | 0.063 | 0.1386 | 0.0271 | 0.2263 | 0.9877 | 0.3039 | 284 | 8329 | 104 | 124 |
| **Autoencoder** | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0297 | 0.292 | 0.0074 | 1.0 | 0 | 62 | 8371 | 408 |

---

## 2. Key Findings & Architecture Recommendation
1. **Best Supervised Model**: **XGBoost** achieved the highest $F_1$-score (0.58) and PR-AUC (0.563).
2. **Supervised vs. Unsupervised**: Supervised algorithms (Random Forest and XGBoost) leverage graph degree and neighbor aggregations effectively, significantly outperforming purely unsupervised anomaly detection baselines.
3. **No Target Leakage**: All scalers and feature engineering were computed strictly respecting chronological boundaries, ensuring models do not benefit from lookahead bias.
4. **Artifacts Saved**: All weights, scalers, and thresholds have been serialized under `models/elliptic/`.
