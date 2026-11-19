# Project Architecture & Data Flow

**Project:** Intelligent Anti-Money Laundering Detection Using Explainable AI  
**Document:** System Architecture, Data Flow, and Component Interactions

---

## 1. High-Level System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    USER / COMPLIANCE ANALYST                     ║
╚═════════════════════════════╤════════════════════════════════════╝
                              │ Web Browser
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║              STREAMLIT WEB APPLICATION (app/streamlit_app.py)   ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │            SIDEBAR CONTROLS                                │  ║
║  │  [Dataset Selector]  [Model Selector]  [Threshold Badge]  │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                              │                                   ║
║          ┌───────────────────┴───────────────────┐              ║
║          │                                       │              ║
║          ▼                                       ▼              ║
║  ┌───────────────────┐               ┌───────────────────────┐  ║
║  │   MODE 1          │               │   MODE 2              │  ║
║  │   Traditional     │               │   Cryptocurrency      │  ║
║  │   Banking AML     │               │   AML                 │  ║
║  │   (IBM HI-Small)  │               │   (Elliptic Bitcoin)  │  ║
║  └─────────┬─────────┘               └───────────┬───────────┘  ║
╚════════════╪═════════════════════════════════════╪══════════════╝
             │                                     │
             ▼                                     ▼
╔════════════════════════╗         ╔═════════════════════════════╗
║  IBM PIPELINE          ║         ║  ELLIPTIC PIPELINE          ║
║                        ║         ║                             ║
║  src/predict.py        ║         ║  src/elliptic/predict.py    ║
║  AMLPredictor          ║         ║  EllipticPredictor          ║
║                        ║         ║                             ║
║  → preprocessing.py    ║         ║  → feature_engineering.py  ║
║  → feature_eng.py      ║         ║  → train.py artifacts       ║
║  → risk_scoring.py     ║         ║  → risk_scoring.py          ║
║  → explainability.py   ║         ║  → explainability.py        ║
╚════════════╤═══════════╝         ╚═══════════════╤═════════════╝
             │                                     │
             ▼                                     ▼
╔════════════════════════╗         ╔═════════════════════════════╗
║  IBM DATA              ║         ║  ELLIPTIC DATA              ║
║                        ║         ║                             ║
║  HI-Small_Trans.csv    ║         ║  elliptic_txs_features.csv  ║
║  5,078,345 records     ║         ║  elliptic_txs_classes.csv   ║
║  11 raw columns        ║         ║  elliptic_txs_edgelist.csv  ║
║  → 24 features         ║         ║  203,769 transactions       ║
║                        ║         ║  → 183 features             ║
╚════════════╤═══════════╝         ╚═══════════════╤═════════════╝
             │                                     │
             ▼                                     ▼
╔════════════════════════╗         ╔═════════════════════════════╗
║  IBM MODELS            ║         ║  ELLIPTIC MODELS            ║
║                        ║         ║                             ║
║  random_forest.pkl     ║         ║  elliptic_xgboost.pkl       ║
║  xgboost_model.pkl     ║         ║  elliptic_rf.pkl            ║
║  isolation_forest.pkl  ║         ║  elliptic_logreg.pkl        ║
║  autoencoder_v2.keras  ║         ║  elliptic_iso_forest.pkl    ║
║                        ║         ║  elliptic_autoencoder.keras ║
╚════════════════════════╝         ╚═════════════════════════════╝
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                   ╔════════════════════════════╗
                   ║  PREDICTION ENGINE         ║
                   ║                            ║
                   ║  → Softmax probabilities   ║
                   ║  → Optimal threshold       ║
                   ║  → Binary classification   ║
                   ╚════════════╤═══════════════╝
                                │
                   ┌────────────┼─────────────┐
                   ▼            ▼             ▼
         ╔══════════════╗ ╔═══════════╗ ╔══════════════╗
         ║ RISK SCORING ║ ║ SHAP XAI  ║ ║ GRAPH        ║
         ║              ║ ║           ║ ║ ANALYTICS    ║
         ║ 0–100 Score  ║ ║ TreeSHAP  ║ ║ (Crypto      ║
         ║ Low/Med/High ║ ║ Waterfall ║ ║  only)       ║
         ║ Critical     ║ ║ Beeswarm  ║ ║ NetworkX     ║
         ║ Tiers        ║ ║ Bar Plots ║ ║ Plotly HTML  ║
         ╚══════════════╝ ╚═══════════╝ ╚══════════════╝
                   │            │             │
                   └────────────┼─────────────┘
                                ▼
                   ╔════════════════════════════╗
                   ║  ALERT MANAGEMENT          ║
                   ║                            ║
                   ║  Risk-tier filtering        ║
                   ║  Tabular alert display     ║
                   ║  CSV export                ║
                   ╚════════════════════════════╝
```

---

## 2. Data Flow — IBM HI-Small Pipeline

```
HI-Small_Trans.csv (5,078,345 rows)
         │
         ▼
src/data_loader.py
   → Schema validation
   → Timestamp parsing
   → Column selection
         │
         ▼
src/feature_engineering.py
   → Temporal features (Hour, Day, Month, Weekday, Is_Weekend, Is_Night)
   → Monetary features (Amount_Ratio, Log transforms, Amount_Difference)
   → Cross-entity flags (Same_Currency, Same_Bank)
   → Account velocity (Source/Destination transaction frequency & amounts)
   → Categorical frequency encoding (Currency, Payment Format)
   → Output: 24 engineered features + target (Is Laundering)
         │
         ▼
src/preprocessing.py → AMLPreprocessingPipeline
   → Stratified split: Train 64% | Val 16% | Test 20%
   → Fit SimpleImputer (median) on TRAIN only
   → Fit RobustScaler on TRAIN only
   → Save artifacts to models/aml_pipeline_*
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
      Train Set       Val Set       Test Set
  3,250,140 rows   812,536 rows  1,015,669 rows
         │              │
         ▼              ▼
   src/train.py
      → RandomForest (class_weight='balanced')
      → XGBoost (scale_pos_weight calculated)
      → F2-score threshold sweep on Val Set
      → Optimal thresholds: RF=0.6688, XGB=0.8989
      → Save models as models/*.pkl
      → Save thresholds as *_threshold.json
         │
         ▼
  src/predict.py → AMLPredictor
      → load_model() → load scaler/imputer
      → predict_batch(df) → transform + predict_proba
      → risk_score = calculate_risk_score(prob)
      → Alert_Triggered = (prob >= threshold)
         │
         ├──────────────────────────┐
         ▼                          ▼
  src/risk_scoring.py        src/explainability.py
      → 0–100 score mapping       → SHAP TreeExplainer
      → 4 risk tiers              → Global bar / beeswarm
      → Investigation flag        → Local waterfall
```

---

## 3. Data Flow — Elliptic Cryptocurrency Pipeline

```
Raw Files (data/elliptic/raw/)
   ├── elliptic_txs_features.csv (658 MB, 203,769 rows × 167 cols)
   ├── elliptic_txs_classes.csv (3.2 MB, 203,769 rows)
   └── elliptic_txs_edgelist.csv (4.3 MB, 234,355 edges)
         │
         ▼
src/elliptic/data_acquisition.py
   → File existence & size validation
   → Schema verification
   → Cross-file txId consistency check (zero mismatches)
   → Reports: ELLIPTIC_DATASET_REPORT.md
         │
         ▼
src/elliptic/feature_engineering.py
   → Load features with float32 optimization
   → Merge features + classes on txId
   → NetworkX edge list → compute in/out degrees via value_counts()
   → Engineered features (18 added):
       in_degree, out_degree, total_degree, degree_ratio, degree_diff
       time_step_norm, time_step_sin, time_step_cos
       local_feature_mean/std/skew/max/min (features 1–93)
       neighbor_feature_mean/std/max/min (features 94–165)
       local_vs_neighbor_mean_ratio
   → Total: 183 features
   → Split by label: elliptic_labelled.csv (46,564), elliptic_unknown.csv (157,205)
   → Save metadata: elliptic_metadata.json
         │
         ▼
src/elliptic/train.py
   → Temporal split (strict):
       Train: time_step ≤ 34  (29,894 rows; Illicit: 3,462, Licit: 26,432)
       Val:   35 ≤ ts ≤ 41   (7,829 rows; Illicit: 675, Licit: 7,154)
       Test:  ts > 41         (8,841 rows; Illicit: 408, Licit: 8,433)
   → Fit RobustScaler on TRAIN only → models/elliptic/elliptic_scaler.pkl
   → Train 5 models:
       Logistic Regression, Random Forest, XGBoost (supervised)
       Isolation Forest, Autoencoder (unsupervised)
   → F2-sweep threshold on Val set
   → Save to models/elliptic/
   → Evaluate on Test set → results/elliptic/metrics/
   → Generate ELLIPTIC_MODEL_EVALUATION.md
         │
         ▼
src/elliptic/predict.py → EllipticPredictor
   → Load scaler, feature_names, models, thresholds
   → predict_batch(df) → transform + predict_proba
   → Compute risk scores with network penalty
         │
         ├──────────────────────────┐
         ▼                          ▼
src/elliptic/risk_scoring.py  src/elliptic/explainability.py
   → calculate_crypto_risk_score()   → SHAP TreeExplainer
   → Network penalty for degree      → Global bar + beeswarm
   → 4 tiers (Low/Med/High/Critical) → ELLIPTIC_SHAP_REPORT.md
         │
         ▼
src/elliptic/graph_analysis.py
   → Load edgelist → NetworkX DiGraph
   → Extract illicit-centered subgraph (260 nodes, 250 edges)
   → Compute PageRank, degree
   → Spring layout
   → Plotly scatter/edge traces
   → Export transaction_network.html (4.9 MB)
   → ELLIPTIC_GRAPH_ANALYSIS.md
```

---

## 4. Streamlit Dashboard Architecture

```
app/streamlit_app.py (startup)
         │
         ▼
st.sidebar.selectbox("Active Surveillance Domain")
         │
     ┌───┴───────────────────────────────────┐
     │  "Traditional Banking (IBM HI-Small)" │
     │  → @st.cache_resource: load_aml_predictor()
     │  → AMLPredictor(model_name)
     │  → load_ibm_demo_data(1000 rows from aml_test.csv)
     │
     │  "Cryptocurrency (Elliptic Bitcoin)"  │
     │  → @st.cache_resource: load_elliptic_predictor()
     │  → EllipticPredictor(models_dir="models/elliptic")
     │  → load_elliptic_demo_data(1000 rows from elliptic_labelled.csv)
     └───────────────────────────────────────┘
         │
         ▼
st.sidebar.selectbox("Active ML Classifier")
  IBM:   XGBoost | Random Forest
  Crypto: XGBoost | Random Forest | Logistic Regression
         │
         ▼
st.sidebar.radio("Navigation")
  ┌─ 📊 Executive Dashboard
  ├─ 🔍 Transaction Analysis
  ├─ 🧠 Explainable AI (SHAP)
  ├─ ⚡ Anomaly Detection
  ├─ 🚨 Alert Management
  ├─ 📈 Model Performance
  └─ 🌐 Blockchain Graph Analytics [Crypto Only]
```

---

## 5. ML Workflow Diagram

```
                     Raw Dataset
                          │
              ┌───────────▼───────────┐
              │   Data Validation     │
              │  & Quality Checks     │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Feature Engineering │
              │  (24 IBM / 183 Crypto)│
              └───────────┬───────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                   ▼             ▼
              Train Set      Val Set + Test Set
                   │
         ┌─────────┼──────────────────┐
         ▼         ▼                  ▼
    Logistic   Random          XGBoost
    Regression Forest        (Gradient Boosting)
         │         │                  │
         └─────────┼──────────────────┘
                   │
         ┌─────────▼──────────────┐
         │  F2-Score Threshold    │
         │  Optimization on Val   │
         └─────────┬──────────────┘
                   │
         ┌─────────▼──────────────┐
         │  Test Set Evaluation   │
         │  Precision/Recall/     │
         │  F1/PR-AUC/ROC-AUC    │
         └─────────┬──────────────┘
                   │
              ┌────┴────┐
              ▼         ▼
        SHAP XAI   Risk Scoring
         (TreeExplainer)  (0–100)
```
