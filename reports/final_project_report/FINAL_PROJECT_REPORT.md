# INTELLIGENT ANTI-MONEY LAUNDERING DETECTION USING EXPLAINABLE AI

---

**Project Type:** B.Tech Computer Science and Engineering — Major Academic Project  
**Domain:** Machine Learning, Financial Crime Detection, Blockchain Analytics, Explainable AI  
**Repository:** AML_Project (Dual-Domain: IBM HI-Small + Elliptic Bitcoin)  
**Report Date:** November 17, 2026  

---

## Abstract

Money laundering is a globally pervasive financial crime that involves concealing the illicit origins of criminal proceeds by routing them through layers of complex transactions. Traditional rule-based Anti-Money Laundering (AML) systems produce high volumes of false positive alerts, lack adaptability to evolving crime typologies, and offer no explanation for their decisions. The proliferation of cryptocurrency networks has further complicated the landscape, introducing pseudonymous transaction identities, decentralized architectures, and graph-topological laundering patterns.

This project develops an **Intelligent Anti-Money Laundering Detection Platform** using machine learning, unsupervised anomaly detection, Explainable AI (XAI) based on SHAP, and blockchain graph analytics. The platform operates across two independent, decoupled surveillance domains: **(1) Traditional Banking AML** using the IBM HI-Small Transaction Dataset (5,078,345 records), and **(2) Cryptocurrency AML** using the Elliptic Bitcoin Transaction Dataset (203,769 Bitcoin transactions across 49 timesteps). Five machine learning models are implemented and evaluated per domain. For the cryptocurrency pipeline, graph neural degree features, temporal cyclical encodings, and 1-hop neighbourhood aggregations extend the original 165 anonymized features to a 183-dimensional representation. All models are interpreted through TreeSHAP, enabling compliance analysts to audit individual transaction risk decisions. A unified Streamlit web dashboard integrates both pipelines, providing an interactive surveillance interface with 7 functional pages including a live Blockchain Graph Analytics visualization built with NetworkX and Plotly. The complete test suite achieves 31/31 tests passing.

**Keywords:** Anti-Money Laundering, Machine Learning, XGBoost, Random Forest, SHAP Explainability, Isolation Forest, Autoencoder, Graph Analytics, NetworkX, Elliptic Bitcoin Dataset, IBM HI-Small, Streamlit, Class Imbalance, Temporal Split

---

## Table of Contents

1. Introduction  
2. Problem Statement and Objectives  
3. Dataset Description  
   3.1 IBM HI-Small Transaction Dataset  
   3.2 Elliptic Bitcoin Transaction Dataset  
4. Data Preprocessing  
   4.1 IBM Banking Preprocessing Pipeline  
   4.2 Elliptic Cryptocurrency Preprocessing Pipeline  
5. Feature Engineering  
   5.1 Banking AML Feature Engineering  
   5.2 Cryptocurrency Graph Feature Engineering  
6. Machine Learning Algorithms  
   6.1 Logistic Regression  
   6.2 Random Forest  
   6.3 XGBoost  
   6.4 Isolation Forest  
   6.5 Autoencoder  
7. Model Evaluation  
   7.1 Evaluation Methodology  
   7.2 IBM Model Performance Results  
   7.3 Elliptic Cryptocurrency Model Results  
8. Explainable AI Using SHAP  
9. Cryptocurrency Graph Analytics  
10. Risk Scoring and Alert Management  
11. System Architecture  
12. Streamlit Application  
13. Implementation Details  
14. Results and Discussion  
15. Limitations and Future Scope  
16. Conclusion  
17. References  

---

## Chapter 1: Introduction

### 1.1 Background of Money Laundering

Money laundering is the process by which individuals or organizations conceal the origins of illegally obtained money, transforming criminal proceeds into apparently legitimate assets. The United Nations Office on Drugs and Crime (UNODC) estimates that between 2% and 5% of global GDP—equivalent to approximately USD 800 billion to USD 2 trillion annually—is laundered through financial systems worldwide. Money laundering enables criminal enterprises to fund further illegal activities, including drug trafficking, human trafficking, terrorism financing, and cybercrime.

The classical money laundering process proceeds through three recognized stages:

1. **Placement:** Criminal cash or crypto proceeds are injected into the financial system, often through cash deposits below regulatory reporting thresholds (a practice known as "structuring" or "smurfing").
2. **Layering:** Funds are transferred through a complex web of transactions, accounts, shell companies, or cryptocurrency wallets to obscure their origin and sever the audit trail.
3. **Integration:** Cleaned funds re-enter the legitimate economy through investments, real estate purchases, or business revenues.

### 1.2 Traditional Financial Money Laundering

In traditional fiat currency-based banking, money laundering exploits the high transaction volumes and cross-jurisdictional flows of modern financial institutions. Common typologies include structuring deposits below reporting thresholds, wire transfers through multiple international correspondent banks, trade-based laundering through over- or under-invoicing of goods and services, and the use of shell companies and nominee accounts to obscure beneficial ownership.

Financial institutions are required by anti-money laundering regulations—including the Bank Secrecy Act (BSA) in the United States, the EU's Anti-Money Laundering Directives (AMLD), and FATF recommendations—to implement transaction monitoring systems, file Suspicious Activity Reports (SARs), and conduct Know Your Customer (KYC) procedures.

### 1.3 Cryptocurrency-Based Money Laundering

The advent of decentralized cryptocurrencies—particularly Bitcoin—has created new and challenging laundering vectors. Bitcoin transactions are pseudonymous rather than anonymous: each transaction is recorded on the public blockchain and linked to wallet addresses, but the real-world identity behind an address is not inherently disclosed. This pseudonymity, combined with the permissionless nature of blockchain networks, enables several laundering typologies:

- **Peeling chains:** Illicit funds are progressively split across a chain of transactions, with each step reducing the original amount and distributing it across fresh addresses.
- **Consolidation funnels:** Multiple smaller UTXOs (Unspent Transaction Outputs) from criminal wallets are merged through high fan-in transactions before depositing to exchanges.
- **Mixing services / Coin joiners:** Third-party services that pool together Bitcoin from multiple users to break transaction traceability.
- **Darknet marketplace payments:** Bitcoin is extensively used as the payment medium for illegal goods and services on darknet platforms.

The Elliptic dataset, contributed by Elliptic Co. and published on Kaggle, represents the largest publicly available labelled Bitcoin transaction graph, with verified ground truth labels from blockchain forensic analysts.

### 1.4 Challenges in AML Detection

Detecting money laundering in financial transaction data presents several formidable challenges:

- **Extreme class imbalance:** In the IBM HI-Small dataset, only 0.10% of transactions (5,177 out of 5,078,345) are labelled as laundering, creating an approximate 980:1 imbalance ratio. This renders standard accuracy metrics meaningless and demands specialized imbalance-handling techniques.
- **Evolving criminal typologies:** Money launderers continuously adapt their strategies in response to detection systems, requiring models that generalize beyond observed historical patterns.
- **Large transaction volumes:** Financial institutions process millions of transactions daily. Scalable, real-time detection infrastructure is essential.
- **High false positive rates:** Existing rule-based AML systems typically generate extremely high false positive rates (95%+ of flagged alerts are legitimate), resulting in analyst alert fatigue and regulatory compliance burden.
- **Limited model interpretability:** Deep learning and ensemble models are inherently opaque, making it difficult to explain flagged transactions to compliance officers, regulators, or auditors.
- **Complex graph relationships:** Bitcoin laundering patterns are fundamentally graph-topological in nature, requiring network-aware feature extraction that classical tabular methods do not capture.

### 1.5 Motivation

This project is motivated by the need to advance beyond rule-based AML monitoring systems through the development of an intelligent, explainable, and comprehensive detection platform that:

1. Applies proven machine learning models (Random Forest, XGBoost) with appropriate class-imbalance handling.
2. Extends detection to unsupervised anomaly detection (Isolation Forest, Autoencoder) for detecting novel laundering patterns without labelled examples.
3. Integrates SHAP Explainable AI to provide interpretable, feature-level explanations of every flagged transaction.
4. Builds a Bitcoin transaction graph analytics module using NetworkX and Plotly to visually represent the network topology of suspected laundering clusters.
5. Provides a unified, production-ready web interface supporting both traditional banking and cryptocurrency surveillance domains.

---

## Chapter 2: Problem Statement and Objectives

### 2.1 Problem Statement

Detecting money laundering in modern financial systems involves the classification of transactions as illicit (suspicious) or legitimate within datasets characterized by:

- Extreme class imbalance (up to 980:1 in traditional banking, ~10:1 in cryptocurrency).
- High-dimensional, partially anonymized feature spaces.
- Complex temporal transaction sequences with behavioral drift.
- Graph-topological interdependencies in cryptocurrency networks.
- A regulatory requirement for explainable, auditable decision rationale.

The primary research question is: *Can a unified, multi-domain machine learning platform combining supervised classification, unsupervised anomaly detection, SHAP Explainable AI, and graph analytics meaningfully improve AML detection outcomes across both traditional banking and cryptocurrency transaction streams?*

### 2.2 Project Challenges Addressed

| Challenge | Approach Implemented |
|-----------|---------------------|
| Extreme class imbalance (~980:1 IBM) | `class_weight='balanced'` for RF/LR; `scale_pos_weight` for XGBoost |
| Future-to-past leakage in time-series | Strict temporal split for Elliptic (timesteps 1–34 train, 35–41 val, 42–49 test) |
| Threshold misalignment with imbalanced data | F2-score threshold optimization on validation set |
| Black-box model opacity | TreeSHAP global + local explanations |
| Graph-topological laundering patterns | NetworkX degree features + interactive Plotly visualization |
| Unknown/unlabelled transactions | Explicitly excluded from supervised training; preserved for anomaly scoring |
| Alert fatigue from false positives | F2-optimized thresholds prioritize recall; XGBoost achieves 0.89% FPR on IBM |

### 2.3 Objectives

1. Build a modular, fully decoupled preprocessing and inference pipeline for two financially distinct datasets (IBM banking and Elliptic Bitcoin).
2. Train and evaluate five ML models per domain using rigorous, leakage-free methodology.
3. Optimize classification thresholds using validation data exclusively, ensuring test set integrity.
4. Implement SHAP-based Explainable AI for both supervised models in both domains.
5. Construct a Bitcoin transaction graph analytics module using NetworkX that distinguishes ground-truth illicit, licit, and unknown nodes.
6. Build a composite 0–100 risk scoring system with four compliance tiers.
7. Develop a unified Streamlit dashboard supporting both surveillance domains with seven functional pages.
8. Achieve 100% unit test coverage across all modules.

---

## Chapter 3: Dataset Description

### 3.1 IBM HI-Small Transaction Dataset

**Source:** IBM Research AI — Anti-Money Laundering synthetic transaction dataset  
**File:** `data/raw/HI-Small_Trans.csv`

#### 3.1.1 Dataset Structure
| Attribute | Value |
|-----------|-------|
| Total Records | 5,078,345 (excluding header) |
| Number of Columns | 11 raw columns |
| Date Range | September 2022 |
| File Size | ~657 MB |

#### 3.1.2 Raw Column Descriptions
| Column | Type | Description |
|--------|------|-------------|
| Timestamp | String | Transaction datetime in `YYYY/MM/DD HH:MM` format |
| From Bank | Integer | Originating bank identifier |
| Account | String | Source account identifier |
| To Bank | Integer | Destination bank identifier |
| Account (second) | String | Destination account identifier |
| Amount Received | Float | Amount received by the beneficiary |
| Receiving Currency | String | Currency of received funds |
| Amount Paid | Float | Amount sent by the sender |
| Payment Currency | String | Currency of outgoing funds |
| Payment Format | Categorical | Channel (Bitcoin, Cheque, Wire, Credit Card, Cash, ACH, Reinvestment) |
| **Is Laundering** | **Binary** | **Target: 0 = Legitimate, 1 = Money Laundering** |

#### 3.1.3 Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate (0) | 5,073,168 | 99.90% |
| Laundering (1) | 5,177 | 0.10% |
| **Imbalance Ratio** | **~980:1** | — |

#### 3.1.4 Payment Format Distribution (Laundering vs Legitimate)
| Payment Format | Legitimate | Laundering |
|---------------|-----------|-----------|
| ACH | 596,314 | 4,483 (86.6% of all laundering) |
| Bitcoin | 146,035 | 56 |
| Cash | 490,783 | 108 |
| Cheque | 1,864,007 | 324 |
| Credit Card | 1,323,118 | 206 |
| Reinvestment | 481,056 | 0 |
| Wire | 171,855 | 0 |

#### 3.1.5 Data Quality
- **Missing Values:** None detected in raw file (preprocessing imputer added for robustness).
- **Duplicates:** Database-based duplicate check performed (`duplicate_check.db`).
- **Currency Coverage:** 15+ currencies including US Dollar, Euro, Yuan, Swiss Franc, Bitcoin, Ruble, Yen.

#### 3.1.6 Train/Validation/Test Split
| Partition | Records | Laundering Records |
|-----------|---------|-------------------|
| Training (64%) | 3,250,140 | ~3,313 |
| Validation (16%) | 812,536 | ~828 |
| Test (20%) | 1,015,669 | 1,035 (confirmed) |

---

### 3.2 Elliptic Bitcoin Transaction Dataset

**Source:** Elliptic Co., published on Kaggle (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  
**Files:** Three interlocked CSV files stored in `data/elliptic/raw/`

#### 3.2.1 File Inventory
| File | Size | Records | Description |
|------|------|---------|-------------|
| `elliptic_txs_features.csv` | 658 MB | 203,769 rows × 167 columns | Transaction features (headerless) |
| `elliptic_txs_classes.csv` | 3.2 MB | 203,769 rows × 2 columns | Ground truth labels per transaction |
| `elliptic_txs_edgelist.csv` | 4.3 MB | 234,355 rows × 2 columns | Directed payment flow edges |

#### 3.2.2 Features File Schema
The features file has no header row. Columns are mapped as follows:
| Mapped Column | Original Index | Description |
|---------------|----------------|-------------|
| `txId` | Column 0 | Unique Bitcoin transaction identifier |
| `time_step` | Column 1 | Discrete temporal timestep (1 to 49) |
| `feature_1` to `feature_93` | Columns 2–94 | Local transaction features (transaction-level metrics) |
| `feature_94` to `feature_165` | Columns 95–166 | Aggregated 1-hop neighbourhood features |

> **Important:** The Elliptic dataset does not publish the semantic meaning of individual features. Features are retained with their anonymous identifiers `feature_1` through `feature_165`. No invented interpretations are assigned.

#### 3.2.3 Classes File Schema
| Column | Values | Description |
|--------|--------|-------------|
| `txId` | Integer | Matches feature file txId |
| `class` | `'1'`, `'2'`, `'unknown'` | Ground truth classification |

**Class Mapping (applied internally):**
- Class `'1'` (string) → target `1` → **Illicit** transaction
- Class `'2'` (string) → target `0` → **Licit** transaction
- Class `'unknown'` → target `-1` → **Unknown** (excluded from supervised training)

#### 3.2.4 Class Distribution
| Class | Count | Percentage of All | Percentage of Labelled |
|-------|-------|------------------|----------------------|
| Illicit (1) | 4,545 | 2.23% | 9.76% |
| Licit (2) | 42,019 | 20.62% | 90.24% |
| Unknown | 157,205 | 77.15% | — |
| **Total** | **203,769** | 100% | — |

#### 3.2.5 Edgelist Schema
| Column | Description |
|--------|-------------|
| `txId1` | Source transaction (output creator) |
| `txId2` | Destination transaction (input consumer) |

**Graph Statistics:**
- Total directed edges: 234,355
- Nature: Directed Acyclic Graph (Bitcoin UTXO model) where outputs of ancestor transactions become inputs of descendant transactions.

#### 3.2.6 Temporal Characteristics
| Attribute | Value |
|-----------|-------|
| Total Timesteps | 49 |
| Time Step Nature | Discrete integer index (not calendar dates) |
| Training Timesteps | 1–34 |
| Validation Timesteps | 35–41 |
| Test Timesteps | 42–49 |

#### 3.2.7 Treatment of Unknown Transactions

The 157,205 unknown-labelled transactions are **strictly excluded** from supervised model training, validation, and test ground-truth evaluation throughout this project. They are preserved separately in `data/elliptic/processed/elliptic_unknown.csv` for potential unsupervised analysis and are flagged with `is_ground_truth_unknown: true` in risk assessment outputs. Assigning labels to these transactions based on model predictions would constitute fabrication of ground truth and is explicitly prohibited.

---

## Chapter 4: Data Preprocessing

### 4.1 IBM Banking Preprocessing Pipeline

The IBM preprocessing pipeline is implemented in `src/preprocessing.py` as the `AMLPreprocessingPipeline` class. All preprocessing artifacts are saved to `models/aml_pipeline_*` and loaded during inference.

#### 4.1.1 Data Loading and Validation
- Raw CSV loaded via `src/data_loader.py` with schema validation.
- Timestamp column parsed to extract temporal features before discarding the string.
- Column name normalization for `Account` (source) and `Account.1` (destination) disambiguation.

#### 4.1.2 Missing Value Imputation
A `SimpleImputer` with `strategy='median'` is fitted exclusively on the training partition. This prevents validation or test statistics from influencing the imputation of training features (data leakage prevention). The fitted imputer is serialized to `models/aml_pipeline_imputer.pkl`.

#### 4.1.3 Feature Scaling
A `RobustScaler` is applied after imputation, also fitted exclusively on training data. The RobustScaler uses the interquartile range (IQR) instead of variance, making it resilient to the extreme outliers common in financial transaction amounts. The scaler is saved to `models/aml_pipeline_scaler.pkl`.

**Mathematical formulation:**

$$\hat{x}_i = \frac{x_i - Q_{50}(X)}{Q_{75}(X) - Q_{25}(X)}$$

where $Q_{25}$, $Q_{50}$, $Q_{75}$ are the 25th, 50th, and 75th percentiles of feature $X$ computed on the training set.

#### 4.1.4 Stratified Train/Validation/Test Splitting
The dataset is split chronologically-aware with stratification on the target `Is Laundering` to preserve class ratios across partitions:

| Partition | Proportion | Records |
|-----------|-----------|---------|
| Training | 64% | 3,250,140 |
| Validation | 16% | 812,536 |
| Test | 20% | 1,015,669 |

#### 4.1.5 Class Imbalance Strategy
- **SMOTE is NOT applied.** The extreme 980:1 imbalance makes naive SMOTE computationally prohibitive and prone to interpolation artifacts.
- Supervised models use `class_weight='balanced'` (Random Forest, Logistic Regression) or `scale_pos_weight` calculated from training partition ratio (XGBoost).
- Unsupervised models (Isolation Forest, Autoencoder) are trained on the full training set without label constraints.

#### 4.1.6 Categorical Frequency Encoding
Categorical columns (`Receiving Currency`, `Payment Currency`, `Payment Format`) are transformed to numeric frequency-encoded values: each category is replaced by the proportion of training records belonging to that category. This encoding is fitted on training data and saved to `models/aml_pipeline_freq_encoders.json`.

---

### 4.2 Elliptic Cryptocurrency Preprocessing Pipeline

#### 4.2.1 Memory-Optimized Feature Loading
The Elliptic features CSV (658 MB) is loaded with memory-optimized data types:
- `txId`: `int64`
- `time_step`: `int16`
- `feature_1` through `feature_165`: `float32`

This reduces memory footprint by approximately 50% compared to default `float64` loading, enabling processing on a system with 8 GB RAM.

#### 4.2.2 Class Label Mapping and Unknown Exclusion
```
'1' (string) → target=1  (Illicit)
'2' (string) → target=0  (Licit)
'unknown'    → target=-1 (Unknown — excluded from supervised training)
```

After merging features with classes, the dataset is split into:
- `elliptic_labelled.csv`: 46,564 records with targets {0, 1}
- `elliptic_unknown.csv`: 157,205 records with target=-1

#### 4.2.3 Strict Temporal Splitting (No Lookahead)
To simulate realistic deployment conditions where models predict on future, unseen data, the Elliptic dataset is split strictly by time step:

| Partition | Time Steps | Records | Illicit | Licit |
|-----------|-----------|---------|---------|-------|
| Training | 1–34 | 29,894 | 3,462 | 26,432 |
| Validation | 35–41 | 7,829 | 675 | 7,154 |
| Test | 42–49 | 8,841 | 408 | 8,433 |

No transaction from timesteps 35+ is ever exposed to the model during training. This is a critical design decision: random shuffling across time steps would introduce future price signals and temporal correlations that do not exist at prediction time, inflating performance metrics artifactually.

#### 4.2.4 RobustScaler (Crypto)
A `RobustScaler` is fitted on the training partition (timesteps 1–34) and applied to validation and test partitions. Saved to `models/elliptic/elliptic_scaler.pkl`.

---

## Chapter 5: Feature Engineering

### 5.1 Banking AML Feature Engineering

The IBM banking feature engineering is implemented in `src/feature_engineering.py`. Starting from 10 raw columns, 24 predictive features are generated:

#### 5.1.1 Temporal Features
Extracted from the `Timestamp` column:

| Feature | Description |
|---------|-------------|
| `Transaction_Hour` | Hour of day (0–23) — captures night-time transaction anomalies |
| `Transaction_Day` | Day of month (1–31) |
| `Transaction_Month` | Month (1–12) |
| `Transaction_Weekday` | Day of week (0=Mon, 6=Sun) |
| `Is_Weekend` | Binary flag: 1 if Saturday or Sunday |
| `Is_Night_Transaction` | Binary flag: 1 if hour ∈ {22, 23, 0, 1, 2, 3, 4, 5} |

#### 5.1.2 Monetary Features
| Feature | Formula | Description |
|---------|---------|-------------|
| `Amount_Difference` | Received − Paid | Signed flow imbalance |
| `Absolute_Amount_Difference` | \|Received − Paid\| | Absolute imbalance |
| `Amount_Ratio` | Received / (Paid + ε) | Conversion rate proxy |
| `Log_Amount_Received` | log(1 + Received) | Log-transform for right-skewed distribution |
| `Log_Amount_Paid` | log(1 + Paid) | Log-transform |

#### 5.1.3 Cross-Entity Binary Flags
| Feature | Description |
|---------|-------------|
| `Same_Currency` | 1 if Receiving Currency = Payment Currency |
| `Same_Bank` | 1 if From Bank = To Bank |

#### 5.1.4 Account Velocity Features
Four frequency-based features capture historical transaction behaviour per account, computed using backward-looking window aggregations:

| Feature | Description |
|---------|-------------|
| `Source_Transaction_Frequency` | Number of prior transactions from source account |
| `Destination_Transaction_Frequency` | Number of prior transactions to destination account |
| `Previous_Source_Transactions` | Rolling count of recent source account transactions |
| `Previous_Source_Amount` | Rolling sum of recent source account amounts |
| `Previous_Destination_Transactions` | Rolling count of recent destination transactions |
| `Previous_Destination_Amount` | Rolling sum of recent destination amounts |

#### 5.1.5 Categorical Frequency Encoding
| Feature | Description |
|---------|-------------|
| `Receiving Currency_Frequency` | Proportion of training records with this receiving currency |
| `Payment Currency_Frequency` | Proportion with this payment currency |
| `Payment Format_Frequency` | Proportion with this payment format |

---

### 5.2 Cryptocurrency Graph Feature Engineering

The Elliptic feature engineering pipeline is implemented in `src/elliptic/feature_engineering.py`. Starting from 165 anonymized raw features, 18 additional features are computed, yielding 183 total input dimensions.

#### 5.2.1 Original Anonymized Features
| Range | Count | Category |
|-------|-------|----------|
| `feature_1` to `feature_93` | 93 | Local transaction features (e.g., fees, volume, input/output counts) |
| `feature_94` to `feature_165` | 72 | Aggregated 1-hop neighbourhood features |

No semantic meanings are assigned to individual anonymized features. The dataset authors (Elliptic Co.) have not published feature descriptions to protect proprietary intelligence.

#### 5.2.2 Graph Degree Features
Using the directed edge list, in-degree and out-degree are computed for every transaction node using Pandas `value_counts()` for scalability:

$$\text{in\_degree}(v) = |\{u \mid (u, v) \in E\}|$$

$$\text{out\_degree}(v) = |\{w \mid (v, w) \in E\}|$$

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `in_degree` | Count of incoming edges to txId | Fan-in: how many prior transactions fund this one |
| `out_degree` | Count of outgoing edges from txId | Fan-out: how many subsequent transactions this one funds |
| `total_degree` | in_degree + out_degree | Total network connectivity |
| `degree_ratio` | in_degree / (total_degree + 1) | Fraction of connections that are incoming (mixing/consolidation indicator) |
| `degree_diff` | out_degree − in_degree | Signed directionality: positive = more outputs than inputs (peeling) |

#### 5.2.3 Temporal Cyclical Features
To represent the circular nature of time steps without imposing artificial ordinality:

$$\text{time\_step\_norm} = \frac{\text{time\_step}}{49}$$

$$\text{time\_step\_sin} = \sin\!\left(\frac{2\pi \cdot \text{time\_step}}{49}\right)$$

$$\text{time\_step\_cos} = \cos\!\left(\frac{2\pi \cdot \text{time\_step}}{49}\right)$$

The sine-cosine encoding ensures that timestep 1 and timestep 49 are represented as near-neighbours in feature space, capturing temporal periodicity.

#### 5.2.4 Local Feature Summary Statistics
Aggregate statistics across the 93 local transaction features (feature_1..feature_93) per transaction row:

| Feature | Description |
|---------|-------------|
| `local_feature_mean` | Mean of features 1–93 |
| `local_feature_std` | Standard deviation of features 1–93 |
| `local_feature_skew` | Skewness of features 1–93 |
| `local_feature_max` | Maximum of features 1–93 |
| `local_feature_min` | Minimum of features 1–93 |

#### 5.2.5 Neighbourhood Aggregation Statistics
Aggregate statistics across the 72 neighbourhood features (feature_94..feature_165):

| Feature | Description |
|---------|-------------|
| `neighbor_feature_mean` | Mean of features 94–165 |
| `neighbor_feature_std` | Standard deviation of features 94–165 |
| `neighbor_feature_max` | Maximum of features 94–165 |
| `neighbor_feature_min` | Minimum of features 94–165 |

#### 5.2.6 Cross-Domain Contrast Feature
$$\text{local\_vs\_neighbor\_mean\_ratio} = \frac{\text{local\_feature\_mean} - \text{neighbor\_feature\_mean}}{|\text{neighbor\_feature\_mean}| + \varepsilon}$$

This ratio captures whether a transaction's local behaviour significantly deviates from its neighbourhood context—a potential signal for peeling chain anchor transactions.

#### 5.2.7 Leakage Prevention
The class label (`target`) is never used to compute any engineered feature. All graph statistics are derived solely from the edge list structure. All row-level statistical aggregations are computed independently per transaction without reference to labels.

---

## Chapter 6: Machine Learning Algorithms

### 6.1 Logistic Regression

**Implementation:** `sklearn.linear_model.LogisticRegression` with `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`.  
**Applied To:** Elliptic Cryptocurrency Dataset only.

Logistic Regression models the probability of a binary outcome using the sigmoid (logistic) function applied to a linear combination of features:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

The model parameters are optimized by minimizing the binary cross-entropy loss with L2 regularization:

$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right] + \lambda \|\mathbf{w}\|_2^2$$

**Class Imbalance:** The `class_weight='balanced'` parameter scales the loss contribution of each sample inversely proportional to its class frequency.

**Limitations:** The logistic regression model on the Elliptic dataset showed convergence warnings after 1,000 iterations, suggesting that the 183-dimensional feature space may benefit from dimensionality reduction or a regularized solver variant.

**Elliptic Test Performance:** Precision=17.93%, Recall=63.73%, F1=0.2799, PR-AUC=0.2537, ROC-AUC=0.8616.

---

### 6.2 Random Forest

**Implementation:** `sklearn.ensemble.RandomForestClassifier` with `n_estimators=100`, `max_depth=16`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42`.  
**Applied To:** IBM HI-Small (Mode 1) and Elliptic Bitcoin (Mode 2).

A Random Forest is an ensemble of $T$ decision trees, each trained on a bootstrap sample of the training data with random feature subsets at each split (the "Random Subspace" method):

$$\hat{y} = \text{majority\_vote}\!\left(\{h_t(\mathbf{x})\}_{t=1}^{T}\right)$$

For probability estimation, the predicted probability is the average of individual tree probabilities:

$$P(y=1 \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} P_t(y=1 \mid \mathbf{x})$$

**Class Imbalance:** `class_weight='balanced'` adjusts tree split quality criteria by inversely weighting minority class samples.

**IBM Test Performance (θ=0.6688):** Precision=33.09%, Recall=67.05%, F1=0.4432, PR-AUC=0.4903, ROC-AUC=0.9751, FPR=1.42%.

**Elliptic Test Performance (θ=0.50):** Precision=74.52%, Recall=47.30%, F1=0.5787, PR-AUC=0.5488, ROC-AUC=0.8590, FPR=0.78%.

---

### 6.3 XGBoost (Extreme Gradient Boosting)

**Implementation:** `xgboost.XGBClassifier` with `n_estimators=150`, `max_depth=6`, `learning_rate=0.08` (Elliptic) / default (IBM), `scale_pos_weight` calculated from training partition, `eval_metric='logloss'`, `random_state=42`.  
**Applied To:** IBM HI-Small (Mode 1) and Elliptic Bitcoin (Mode 2).

XGBoost builds an additive ensemble of regression trees, where each tree corrects the errors of the previous. The prediction is:

$$\hat{y}^{(m)} = \hat{y}^{(m-1)} + \eta \cdot h_m(\mathbf{x})$$

where $h_m$ is the $m$-th weak learner tree and $\eta$ is the learning rate. Trees are grown by minimizing a regularized objective:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N} l\!\left(y_i, \hat{y}_i^{(m-1)} + h_m(\mathbf{x}_i)\right) + \Omega(h_m)$$

where $l$ is the binary cross-entropy loss and $\Omega$ is a complexity regularization term.

**Class Imbalance:** `scale_pos_weight = count(negative) / count(positive)`. For IBM: ~980. For Elliptic: 7.63.

**IBM Test Performance (θ=0.8989):** Precision=44.21%, Recall=67.54%, F1=0.5344, PR-AUC=0.6164, ROC-AUC=0.9812, FPR=0.89%.

**Elliptic Test Performance (θ=0.67):** Precision=74.33%, Recall=47.55%, F1=0.5800, PR-AUC=0.5630, ROC-AUC=0.8730, FPR=0.79%.

**XGBoost is the best-performing model on both datasets** by F1-score, PR-AUC, and ROC-AUC.

---

### 6.4 Isolation Forest

**Implementation:** `sklearn.ensemble.IsolationForest` with `n_estimators=100`, `contamination=0.10` (Elliptic) or computed (IBM), `random_state=42`, `n_jobs=-1`.  
**Applied To:** IBM HI-Small (Mode 1) and Elliptic Bitcoin (Mode 2).

Isolation Forest is an unsupervised anomaly detection algorithm that isolates anomalies by recursively partitioning the feature space with random axis-aligned cuts. Anomalous points (outliers) require fewer cuts to isolate:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where $h(x)$ is the number of edges traversed from root to leaf for point $x$, and $c(n)$ is the average path length for $n$ samples. Points with $s \rightarrow 1$ are anomalous.

**IBM Performance:** Precision=0.156%, Recall=27.92%, ROC-AUC=0.5987, Total Alerts=185,187/1,015,669.  
**Elliptic Performance:** Precision=3.30%, Recall=69.61%, F1=0.063, ROC-AUC=0.2263.

> **Note:** Isolation Forest achieves reasonable recall but very poor precision, confirming that unsupervised methods alone cannot replace supervised classification for AML when labelled data is available.

---

### 6.5 Autoencoder (Deep Neural Network)

**Implementation:** TensorFlow/Keras Sequential model with architecture: `Input → Dense(96, ReLU) → Dropout(0.1) → Dense(48, ReLU) → Dense(24, ReLU) → Dense(48, ReLU) → Dropout(0.1) → Dense(96, ReLU) → Dense(input_dim, Linear)`.  
**Training:** Trained exclusively on **licit transactions** from the training partition. Anomaly score = per-sample MSE reconstruction error (higher = more anomalous).  
**Applied To:** IBM HI-Small (Mode 1) and Elliptic Bitcoin (Mode 2).

Autoencoders learn a compressed latent representation of normal (licit) transaction patterns. When an illicit transaction—which the autoencoder has never been trained on—is presented, it cannot reconstruct it accurately, resulting in high reconstruction error:

$$\mathcal{L}_{\text{recon}}(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 = \|\mathbf{x} - f_{\theta}(\mathbf{x})\|_2^2$$

**IBM Performance (Autoencoder V2):** Precision=0.123%, Recall=4.44%, F1=0.0024, ROC-AUC=0.5672.  
**Elliptic Performance:** Precision=0.00%, Recall=0.00%, F1=0.0000, ROC-AUC=0.2920.

> **Note:** The Elliptic Autoencoder produced zero recall on the test partition (timesteps 42–49). This is attributed to significant concept drift across temporal segments — the reconstruction threshold calibrated on validation timesteps 35–41 did not transfer to test timesteps 42–49. The model correctly produces low false positive rates (62/8,433 licit transactions flagged) but fails to detect any illicit transactions at the default threshold. This is a known limitation of threshold-based reconstruction anomaly detection under distributional shift.

---

## Chapter 7: Model Evaluation

### 7.1 Evaluation Methodology

#### 7.1.1 Why Standard Accuracy is Insufficient
With a 980:1 imbalance, a trivial model that classifies every transaction as legitimate achieves 99.90% accuracy while detecting zero laundering. Standard accuracy is therefore uninformative.

#### 7.1.2 Chosen Metrics
| Metric | Formula | Relevance to AML |
|--------|---------|-----------------|
| Precision | TP / (TP + FP) | Fraction of flagged transactions that are genuinely suspicious (alert quality) |
| Recall | TP / (TP + FN) | Fraction of actual laundering transactions detected (coverage) |
| F1-Score | 2·P·R / (P+R) | Harmonic mean balancing precision and recall |
| F2-Score | 5·P·R / (4·P+R) | Recall-weighted F-score (β=2 prioritizes catching more laundering) |
| PR-AUC | Area under P-R curve | Preferred over ROC-AUC for imbalanced datasets |
| ROC-AUC | Area under ROC curve | Discrimination ability across all thresholds |
| FPR | FP / (FP + TN) | False alert rate (alert fatigue indicator) |
| FNR | FN / (FN + TP) | Missed laundering rate |

#### 7.1.3 Threshold Optimization
For every supervised model, classification thresholds were optimized on the **validation set** by sweeping 91 candidate thresholds in [0.05, 0.95] and selecting the threshold maximizing the F2-score. The test set was **never used** for threshold selection.

---

### 7.2 IBM Model Performance Results

#### 7.2.1 Supervised Classification (Test Set N=1,015,669, Actual Positives=1,035)

| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC | FPR |
|-------|-----------|-----------|--------|-----|-----|--------|---------|-----|
| **XGBoost** | 0.8989 | **44.21%** | **67.54%** | **0.5344** | **0.6109** | **0.6164** | **0.9812** | **0.89%** |
| Random Forest | 0.6688 | 33.09% | 67.05% | 0.4432 | 0.5564 | 0.4903 | 0.9751 | 1.42% |

#### 7.2.2 Confusion Matrices (IBM Test Set)
**XGBoost (θ=0.8989):**
```
                  Predicted: 0    Predicted: 1
  Actual: 0        98,083            882
  Actual: 1           336            699
```

**Random Forest (θ=0.6688):**
```
                  Predicted: 0    Predicted: 1
  Actual: 0        97,562          1,403
  Actual: 1           341            694
```

#### 7.2.3 Unsupervised Detection (Full IBM Test Set N=1,015,669)

| Model | Precision | Recall | F1 | ROC-AUC | Alerts Generated |
|-------|-----------|--------|-----|---------|-----------------|
| Isolation Forest | 0.156% | 27.92% | 0.0031 | 0.5987 | 185,187 |
| Autoencoder V2 | 0.123% | 4.44% | 0.0024 | 0.5672 | 37,434 |

---

### 7.3 Elliptic Cryptocurrency Model Results

#### 7.3.1 Test Partition Information (Timesteps 42–49)
| Partition | Records | Illicit | Licit |
|-----------|---------|---------|-------|
| Test | 8,841 | 408 | 8,433 |

#### 7.3.2 Supervised Classification Benchmarks

| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC | TP | FP | FN |
|-------|-----------|-----------|--------|-----|-----|--------|---------|----|----|-----|
| **XGBoost** | 0.6700 | **74.33%** | 47.55% | **0.5800** | **0.5124** | **0.5630** | **0.8730** | 194 | 67 | 214 |
| Random Forest | 0.5000 | 74.52% | 47.30% | 0.5787 | 0.5103 | 0.5488 | 0.8590 | 193 | 66 | 215 |
| Logistic Regression | 0.5300 | 17.93% | **63.73%** | 0.2799 | 0.4218 | 0.2537 | 0.8616 | 260 | 1,190 | 148 |

#### 7.3.3 Unsupervised Anomaly Detection (Crypto)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|-----------|--------|-----|---------|
| Isolation Forest | 0.05 | 3.30% | 69.61% | 0.0630 | 0.2263 |
| Autoencoder | 0.50 | 0.00% | 0.00% | 0.0000 | 0.2920 |

---

## Chapter 8: Explainable AI Using SHAP

### 8.1 SHAP Theoretical Foundation

SHAP (SHapley Additive exPlanations) is a unified framework for model explanation grounded in cooperative game theory. For a machine learning model $f$ and an input $\mathbf{x}$, the SHAP value $\phi_j(\mathbf{x})$ for feature $j$ quantifies that feature's contribution to the difference between the model's prediction $f(\mathbf{x})$ and the expected model prediction $E[f]$:

$$f(\mathbf{x}) = E[f] + \sum_{j=1}^{p} \phi_j(\mathbf{x})$$

Shapley values are computed by considering all $2^p$ possible feature coalitions:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[ f_{S \cup \{j\}}(\mathbf{x}_{S \cup \{j\}}) - f_S(\mathbf{x}_S) \right]$$

where $F$ is the full feature set and $S$ iterates over all subsets not including feature $j$.

**TreeSHAP** is an efficient algorithm that computes exact Shapley values for tree ensembles (Random Forest, XGBoost) in $O(T \cdot L \cdot D^2)$ time, where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is the tree depth.

### 8.2 Global Feature Importance

Global SHAP explanations are computed by taking the mean absolute SHAP value across a balanced sample of 300 transactions (150 illicit, 150 licit):

$$\text{Global Importance}_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_j(\mathbf{x}_i)|$$

#### 8.2.1 Top 15 Elliptic XGBoost Features by SHAP Importance

| Rank | Feature | Mean |SHAP| | Category |
|------|---------|------------|----------|
| 1 | `local_feature_std` | 1.1427 | Engineered |
| 2 | `feature_53` | 1.0003 | Local Transaction |
| 3 | `feature_59` | 0.6518 | Local Transaction |
| 4 | `feature_90` | 0.5618 | Local Transaction |
| 5 | `feature_2` | 0.4287 | Local Transaction |
| 6 | `feature_76` | 0.3740 | Local Transaction |
| 7 | `feature_58` | 0.3337 | Local Transaction |
| 8 | `feature_46` | 0.2877 | Local Transaction |
| 9 | `feature_5` | 0.2356 | Local Transaction |
| 10 | `feature_3` | 0.2347 | Local Transaction |
| 11 | `feature_161` | 0.2202 | Neighbourhood |
| 12 | `feature_125` | 0.2180 | Neighbourhood |
| 13 | `feature_52` | 0.2098 | Local Transaction |
| 14 | `feature_127` | 0.1625 | Neighbourhood |
| 15 | `feature_163` | 0.1613 | Neighbourhood |

**Observation:** The engineered `local_feature_std` ranks first, indicating that the statistical dispersion of local transaction metrics is the single most discriminating feature. Neighbourhood aggregation features (94–165 range) occupy 4 of the top 15 slots, confirming that 1-hop context meaningfully supplements transaction-level features.

### 8.3 Visualization Artifacts

| Artifact | Path |
|---------|------|
| XGBoost Global Bar Plot | `results/elliptic/graphs/shap_global_bar_xgboost.png` |
| XGBoost Beeswarm | `results/elliptic/graphs/shap_global_beeswarm_xgboost.png` |
| Random Forest Global Bar Plot | `results/elliptic/graphs/shap_global_bar_random_forest.png` |
| Random Forest Beeswarm | `results/elliptic/graphs/shap_global_beeswarm_random_forest.png` |
| IBM XGBoost Bar Plot | `results/graphs/shap_global_bar_xgboost.png` |
| IBM Random Forest Bar Plot | `results/graphs/shap_global_bar_random_forest.png` |
| IBM Local Waterfall (sample) | `results/graphs/interactive_waterfall.png` |

### 8.4 Local (Individual Transaction) Explanations

For the IBM pipeline, the `src/explainability.py` module provides `explain_transaction()` which:
1. Applies the SHAP TreeExplainer to a single transaction.
2. Generates a waterfall plot showing the cumulative feature contributions from the base expected value to the final prediction.
3. Generates natural language reasoning bullets identifying the top positive and negative contributing features.

For the Elliptic pipeline, only global explanations are implemented. Local waterfall explanations for individual cryptocurrency transactions are not yet implemented.

---

## Chapter 9: Cryptocurrency Graph Analytics

### 9.1 Bitcoin Transaction Graph Structure

Bitcoin uses an Unspent Transaction Output (UTXO) model where each transaction consumes one or more UTXOs as inputs and creates one or more new UTXOs as outputs. This naturally forms a **Directed Acyclic Graph (DAG)** where:
- **Nodes** represent individual Bitcoin transactions.
- **Directed edges** represent payment flows: an edge (txId1 → txId2) means that an output of transaction txId1 is consumed as an input to transaction txId2.

The Elliptic dataset provides this graph as a directed edgelist with 234,355 edges connecting 203,769 transaction nodes across 49 timesteps.

### 9.2 Graph Construction

The graph analytics module (`src/elliptic/graph_analysis.py`) uses **NetworkX DiGraph**:

```python
G = nx.DiGraph()
for _, row in sub_edges.iterrows():
    G.add_edge(int(row['txId1']), int(row['txId2']))
```

**Degree Computation (for all 203,769 nodes):**
In-degree and out-degree for all nodes are computed using efficient Pandas `value_counts()` on the full edgelist, without constructing the full NetworkX graph in memory:

```python
in_degrees = df_edges['txId2'].value_counts()
out_degrees = df_edges['txId1'].value_counts()
```

**Subgraph Extraction for Visualization:**
An illicit-centered subgraph is extracted by selecting edges incident to the top 50 confirmed illicit transactions and their 2-hop neighbours, up to 250 edges (260 nodes).

### 9.3 Node Metric Computation

| Metric | Computation | Description |
|--------|------------|-------------|
| Degree | `G.degree()` | Total edge count per node |
| In-Degree | `G.in_degree()` | Incoming payment count |
| Out-Degree | `G.out_degree()` | Outgoing payment count |
| PageRank | `nx.pagerank(G, alpha=0.85)` | Node importance by payment-flow propagation |

### 9.4 Layout and Visualization

A **Spring Layout** (Fruchterman-Reingold force-directed algorithm) is used for 2D node placement:

```python
pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)
```

The interactive visualization is built with **Plotly** as a scatter/line trace combination:
- **Edge traces:** Gray lines connecting node positions.
- **Node traces:** Colored circles scaled by `max(8, min(25, 8 + degree × 2))`.
- **Node hover text:** txId, class, in-degree, out-degree, PageRank.

**Node Color Coding (Ground Truth Only):**
| Color | Class | Ground Truth |
|-------|-------|-------------|
| 🔴 `#EF4444` (Red) | Class 1 | Confirmed Illicit |
| 🟢 `#10B981` (Green) | Class 2 | Confirmed Licit |
| ⚪ `#9CA3AF` (Gray) | unknown | Ground Truth Unavailable |

> **Critical distinction:** Node colors represent **verified dataset labels** from blockchain forensic analysts, not model predictions. No predicted labels are presented as ground truth.

### 9.5 Topological Laundering Patterns Identified

1. **Peeling Chains:** Illicit transactions exhibit high out-degree chains (high `degree_diff` = out − in), consistent with the peeling chain typology where funds are progressively dispersed to fresh UTXO addresses.

2. **Consolidation Hubs:** High in-degree nodes (many incoming edges, `degree_ratio` near 1.0) are associated with funds consolidation before exchange deposit — a characteristic of profit extraction following a laundering cycle.

3. **Neighbourhood Homophily:** Illicit nodes cluster with other illicit or unknown nodes in the extracted subgraph, validating the predictive power of 1-hop neighbourhood aggregated features in the ML models.

---

## Chapter 10: Risk Scoring and Alert Management

### 10.1 IBM Banking Risk Scoring

Implemented in `src/risk_scoring.py`:

$$\text{base\_score} = P_{\text{launder}} \times 100$$

$$\text{risk\_score} = \text{Clamp}(\text{base\_score}, 0, 100)$$

| Tier | Score Range | Action |
|------|------------|--------|
| 🟢 Low Risk | 0–29 | Auto-clear |
| 🟡 Medium Risk | 30–59 | Standard audit queue |
| 🟠 High Risk | 60–79 | Level-1 compliance review |
| 🔴 Critical Risk | 80–100 | Immediate SAR consideration |

An `Alert_Triggered` boolean flag is set when the model probability $\geq$ optimal threshold.

### 10.2 Cryptocurrency Risk Scoring

Implemented in `src/elliptic/risk_scoring.py` with an additional **network context penalty**:

$$\text{base\_score} = \text{Clamp}(P_{\text{illicit}}, 0, 1) \times 100$$

$$\text{network\_penalty} = \min(5.0, \text{total\_degree} \times 0.2) + \begin{cases} 3.0 & \text{if degree\_ratio} \notin [0.15, 0.85] \\ 0 & \text{otherwise} \end{cases}$$

$$\text{risk\_score} = \text{Clamp}(\text{base\_score} + \text{network\_penalty}, 0, 100)$$

The network penalty adds up to 8 points for transactions with high connectivity (peeling/consolidation hubs) and asymmetric flow patterns that are topologically associated with laundering.

| Tier | Score Range |
|------|------------|
| Low | 0–29 |
| Medium | 30–69 |
| High | 70–89 |
| Critical | 90–100 |

### 10.3 Compliance Disclaimer

> All risk scores and ML predictions produced by this system are **probabilistic indicators for compliance triage and enhanced due diligence only**. They do not constitute definitive legal proof of money laundering or any other illicit financial activity. All flagged transactions must be reviewed by qualified compliance professionals before any regulatory or legal action is taken.

---

## Chapter 11: System Architecture

### 11.1 High-Level Architecture

The platform implements complete **pipeline isolation** between the two AML domains. No feature data, preprocessing artifacts, model weights, or risk scoring logic crosses between the IBM and Elliptic pipelines.

```
USER
 │
 ▼
STREAMLIT APPLICATION (app/streamlit_app.py)
 │
 ├── SIDEBAR: Active Surveillance Domain selector
 │            Active ML Classifier selector
 │            Operating Threshold display
 │
 ├── MODE 1: Traditional Banking AML
 │    │
 │    ├── Data: data/raw/HI-Small_Trans.csv
 │    ├── Preprocessing: src/preprocessing.py (AMLPreprocessingPipeline)
 │    │   models/aml_pipeline_imputer.pkl
 │    │   models/aml_pipeline_scaler.pkl
 │    │   models/aml_pipeline_freq_encoders.json
 │    ├── Feature Engineering: src/feature_engineering.py (24 features)
 │    ├── Models: models/xgboost_model.pkl, models/random_forest_model.pkl
 │    │          models/isolation_forest_model.pkl, models/aml_autoencoder_v2.keras
 │    ├── Inference: src/predict.py (AMLPredictor)
 │    ├── Risk Scoring: src/risk_scoring.py
 │    └── XAI: src/explainability.py (AMLExplainer)
 │
 └── MODE 2: Cryptocurrency AML
      │
      ├── Data: data/elliptic/raw/*.csv
      ├── Preprocessing: src/elliptic/feature_engineering.py (183 features)
      │   models/elliptic/elliptic_scaler.pkl
      │   models/elliptic/elliptic_feature_names.json
      ├── Models: models/elliptic/elliptic_xgboost.pkl
      │          models/elliptic/elliptic_random_forest.pkl
      │          models/elliptic/elliptic_logistic_regression.pkl
      │          models/elliptic/elliptic_isolation_forest.pkl
      │          models/elliptic/elliptic_autoencoder.keras
      ├── Inference: src/elliptic/predict.py (EllipticPredictor)
      ├── Risk Scoring: src/elliptic/risk_scoring.py (with network penalty)
      ├── XAI: src/elliptic/explainability.py (EllipticExplainer)
      └── Graph Analytics: src/elliptic/graph_analysis.py
                           results/elliptic/graphs/transaction_network.html
```

---

## Chapter 12: Streamlit Dashboard

The unified dashboard is implemented in `app/streamlit_app.py` with 7 functional pages.

### 12.1 Sidebar Controls
- **Active Surveillance Domain:** Toggles the entire pipeline between IBM Banking and Elliptic Cryptocurrency.
- **Active ML Classifier:** IBM offers XGBoost and Random Forest; Elliptic offers XGBoost, Random Forest, and Logistic Regression.
- **Operating Threshold Badge:** Displays the F2-optimized threshold for the selected model.

### 12.2 Page 1 — Executive Dashboard
Displays KPI cards (Total Analyzed, Legitimate, Suspicious, High Risk, Critical Risk) computed from 1,000 demonstration records. Shows classification donut chart and risk tier bar chart. For the crypto domain, displays a scatter plot of blockchain degree connectivity vs. computed risk score.

### 12.3 Page 2 — Transaction Analysis
Two tabs:
- **Batch CSV Ingestion:** File upload widget accepts any CSV conforming to the active domain schema. Batch predictions are run on all uploaded transactions. Falls back to 100 held-out demo transactions if no file is uploaded.
- **Single Transaction Evaluator:**
  - IBM: Interactive widgets for all 10 raw transaction fields. Builds a full raw transaction dictionary → processes through the complete feature engineering pipeline → generates probability, risk score, and SHAP explanation.
  - Crypto: Row selector from the processed labelled dataset with inline risk metrics display.

### 12.4 Page 3 — Explainable AI (SHAP)
Two subtabs:
- **Global Feature Importance:** Displays the domain- and model-specific global SHAP bar plot and beeswarm plot (four model-specific PNGs for Elliptic; four for IBM).
- **Local Waterfall (IBM only):** Index slider selects a demonstration transaction; on-demand waterfall plot generation with top contributing features listed as natural language bullets. *Not yet implemented for the Elliptic cryptocurrency domain.*

### 12.5 Page 4 — Anomaly Detection
Comparative metrics display for Isolation Forest and Deep Autoencoder. Shows training sample counts, detection thresholds, recall, and precision. Includes a methodology note distinguishing unsupervised anomaly detection from supervised classification.

### 12.6 Page 5 — Alert Management
Risk-tier multiselect filter, prediction status filter. Displays filtered transactions in a dataframe. One-click CSV export of filtered alerts for compliance investigation handoff.

### 12.7 Page 6 — Model Performance
Full benchmark metric table loaded from `results/[domain]/metrics/model_comparison.csv`. IBM mode additionally shows PR curves, ROC curves, and confusion matrix images.

### 12.8 Page 7 — Blockchain Graph Analytics (Crypto Only)
Embeds the interactive 4.9 MB Plotly HTML network visualization directly in the Streamlit page using `st.components.v1.html()`. Displays explanatory text distinguishing ground truth classes from model predictions and explaining Bitcoin laundering topologies (peeling chains, consolidation hubs).

---

## Chapter 13: Implementation Details

### 13.1 Technical Stack
| Component | Technology | Version |
|-----------|-----------|---------|
| Programming Language | Python | 3.12 (venv in `.venv/`) |
| ML Framework | scikit-learn | 1.9.1 |
| Gradient Boosting | XGBoost | 3.4.1 |
| Deep Learning | TensorFlow/Keras | 2.21.0 |
| Explainable AI | SHAP | 0.54.1 |
| Graph Analytics | NetworkX | 3.7 |
| Web Framework | Streamlit | 1.67.0 |
| Visualization | Plotly | 7.1.0 |
| Visualization | Matplotlib / Seaborn | 3.10.3 / 0.13.2 |
| Data Processing | Pandas | 2.3.1 |
| Numerical Computing | NumPy | 2.3.1 |
| Model Persistence | Joblib | 1.6.0 |
| Operating System | macOS | — |

### 13.2 Key Source Modules

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `src/preprocessing.py` | — | IBM preprocessing pipeline class |
| `src/feature_engineering.py` | — | 24 IBM feature generation |
| `src/predict.py` (AMLPredictor) | — | IBM unified inference engine |
| `src/risk_scoring.py` | — | IBM 0–100 risk scoring |
| `src/explainability.py` (AMLExplainer) | — | IBM SHAP global + local |
| `src/elliptic/data_acquisition.py` | 143 | Phase 1 validation |
| `src/elliptic/feature_engineering.py` | 148 | 183 crypto feature generation |
| `src/elliptic/train.py` | 229 | 5-model training + eval |
| `src/elliptic/predict.py` | 92 | Crypto inference engine |
| `src/elliptic/risk_scoring.py` | 68 | Crypto network-aware risk scoring |
| `src/elliptic/explainability.py` | 155 | Crypto SHAP global explanations |
| `src/elliptic/graph_analysis.py` | 165 | NetworkX + Plotly network HTML |
| `app/streamlit_app.py` | 400+ | Unified 7-page web dashboard |

### 13.3 Installation and Execution

```bash
# Navigate to project root
cd /Users/pushpasrisindhuganji/AML_Project

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete test suite
python -m unittest discover -s tests -p "test_*.py"

# Run Elliptic pipeline phases (if re-running)
python src/elliptic/data_acquisition.py
python src/elliptic/feature_engineering.py
python src/elliptic/train.py
python src/elliptic/explainability.py
python src/elliptic/graph_analysis.py

# Launch unified web application
streamlit run app/streamlit_app.py
```

---

## Chapter 14: Results and Discussion

### 14.1 IBM Banking AML Results

XGBoost with F2-optimized threshold (0.8989) achieves the best balance of precision (44.21%) and recall (67.54%) among supervised models on the IBM dataset, with a PR-AUC of 0.6164 and ROC-AUC of 0.9812. The false positive rate of 0.89% (882 false alerts per 100,000 transactions) represents a substantial improvement over both the default-threshold XGBoost (5.69% FPR) and typical rule-based AML systems, which commonly produce 90%–99% false positive rates.

Random Forest achieves comparable recall (67.05%) with slightly lower precision (33.09%) and a higher FPR of 1.42%. Both supervised models significantly outperform the unsupervised baselines.

The SHAP analysis of the IBM models reveals that `Payment Format_Frequency` (unusual payment channels), `Source_Transaction_Frequency` (velocity anomalies), and `Amount_Ratio` (flow conversion discrepancies) are the primary drivers of laundering predictions.

### 14.2 Elliptic Cryptocurrency AML Results

Both XGBoost and Random Forest achieve near-identical F1 scores (~0.58) and PR-AUC values (~0.55–0.56) on the cryptocurrency test partition. The Elliptic dataset's 10:1 imbalance (within labelled records) is markedly less severe than IBM's 980:1, resulting in higher model precision for both classifiers.

The strict temporal split reveals an important characteristic of the dataset: model recall on the test partition (timesteps 42–49) is approximately 47–48%, compared to the validation recall of approximately 60–65%. This performance degradation across time steps suggests temporal distribution shift—illicit transaction patterns in later timesteps may differ from those in early timesteps, consistent with the documented evolution of darknet marketplace activity.

Logistic Regression achieves the highest recall (63.73%) among supervised Elliptic models at the cost of very high false positive rate (14.11%), confirming that more complex non-linear models are necessary for effective cryptocurrency AML.

### 14.3 Comparative Analysis of Two Domains

| Dimension | IBM Banking | Elliptic Bitcoin |
|-----------|------------|-----------------|
| Class Imbalance | ~980:1 | ~10:1 (within labelled) |
| Feature Space | 24 handcrafted financial features | 165 anonymized + 18 graph-engineered = 183 |
| Label Coverage | 100% labelled | 22.85% labelled (77.15% unknown) |
| Best Model F1 | 0.5344 (XGBoost) | 0.5800 (XGBoost) |
| Best PR-AUC | 0.6164 (XGBoost) | 0.5630 (XGBoost) |
| Best ROC-AUC | 0.9812 (XGBoost) | 0.8730 (XGBoost) |
| Unsupervised Performance | Very low (Recall ≤ 28%) | Very low (Recall ≤ 70% at 99% FPR) |
| Graph Analytics | Not applicable | Full NetworkX + Plotly visualization |
| Temporal Complexity | Month-long single cohort | 49 discrete timesteps with drift |

---

## Chapter 15: Limitations and Future Scope

### 15.1 Current Limitations

| Limitation | Description |
|-----------|-------------|
| Autoencoder performance | Zero recall on Elliptic test partition due to temporal distribution drift |
| Crypto local SHAP | Individual transaction waterfall explanations not implemented for Elliptic |
| Community detection | Graph modularity community detection not implemented |
| Betweenness centrality as ML feature | Computed only in visualization subgraph, not propagated to full ML pipeline |
| Logistic Regression convergence | LR on Elliptic shows convergence warning at 1,000 iterations |
| Real-time inference | No streaming or real-time transaction ingestion implemented |
| Cross-chain analysis | Only Bitcoin (Elliptic) supported; Ethereum, Monero not covered |

### 15.2 Proposed Future Enhancements

1. **Graph Neural Networks (GNNs):** Replace manual graph feature engineering with GNN architectures (GraphSAGE, GCN, GAT) that natively learn node representations from graph structure, capturing multi-hop laundering chains beyond the 1-hop aggregations implemented here.

2. **Temporal Graph Neural Networks:** Apply Temporal Graph Networks (TGN) to model the dynamic evolution of the transaction graph across the 49 Elliptic timesteps, addressing the temporal distribution shift observed in model performance.

3. **Local SHAP for Cryptocurrency:** Extend `src/elliptic/explainability.py` to generate individual transaction waterfall explanations for crypto transactions, matching the IBM pipeline capability.

4. **Community Detection:** Implement NetworkX `greedy_modularity_communities()` to identify suspicious transaction clusters, enabling investigation of laundering rings rather than individual transactions.

5. **Bitcoin Address Clustering:** Extend the platform to cluster wallet addresses using common-input-ownership heuristics, enabling entity-level (rather than transaction-level) risk profiling.

6. **Model Drift Detection:** Implement statistical drift detection (e.g., Population Stability Index, PSI) to flag when model performance has degraded due to concept drift in live deployment.

7. **Real-Time Blockchain Monitoring:** Integrate with a Bitcoin node API (e.g., BlockCypher, Blockchain.com) to ingest and score live transactions in real time.

8. **Improved Autoencoder Architecture:** Experiment with Variational Autoencoders (VAEs) or LSTM-based autoencoders that capture temporal transaction sequences, potentially overcoming the reconstruction threshold instability observed across timesteps.

9. **Cloud Deployment:** Package the application as a Docker container deployable on Google Cloud Run or AWS Fargate, enabling scalable, production-grade compliance monitoring.

---

## Chapter 16: Conclusion

This project presents a comprehensive **Intelligent Anti-Money Laundering Detection Platform** that addresses the dual surveillance requirements of modern financial crime detection: traditional fiat banking transaction monitoring and cryptocurrency blockchain network analysis.

### Key Contributions

1. **Dual-Domain Architecture:** Two completely isolated ML pipelines (IBM HI-Small and Elliptic Bitcoin) are integrated within a single unified Streamlit dashboard without cross-contamination of features, models, or preprocessing artifacts.

2. **Rigorous ML Implementation:** Five models per domain (Logistic Regression, Random Forest, XGBoost, Isolation Forest, Autoencoder) are implemented with appropriate class-imbalance handling, F2-score threshold optimization on validation partitions, and leakage-free evaluation on held-out test sets.

3. **Best Performance — XGBoost:**
   - **IBM Banking:** F1=0.5344, PR-AUC=0.6164, ROC-AUC=0.9812, FPR=0.89%.
   - **Elliptic Crypto:** F1=0.5800, PR-AUC=0.5630, ROC-AUC=0.8730, FPR=0.79%.

4. **Explainable AI:** TreeSHAP provides both global and local explanations for all supervised IBM models, and global explanations for all supervised Elliptic models, fulfilling the interpretability requirements of financial regulators.

5. **Graph Analytics:** An interactive Bitcoin transaction network visualization built with NetworkX and Plotly enables visual inspection of laundering topologies, distinguishing confirmed illicit, licit, and unknown nodes from verified dataset labels.

6. **Risk Scoring:** A calibrated 0–100 risk scoring system with four compliance tiers maps model probabilities to actionable compliance decisions, with a network-topology penalty for the cryptocurrency domain.

7. **Reliability:** All 31 unit tests pass (100%), covering data acquisition, feature alignment, temporal splitting, model inference, risk scoring, dashboard functionality, and dataset domain switching.

### Practical Significance

This platform demonstrates that machine learning can meaningfully improve AML detection outcomes beyond rule-based systems, achieving false positive rates below 1% while maintaining recall above 65% for traditional banking transactions. The cryptocurrency pipeline, constrained by temporal distribution shift and the 77% unlabelled transaction challenge, still achieves F1>0.58 on the most difficult temporal test partition.

The unified Streamlit dashboard provides compliance teams with an accessible, interpretable interface for investigation triage, alert management, and evidence-based compliance decision-making.

---

## References

[1] M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," *KDD Workshop on Anomaly Detection in Finance*, 2019.

[2] M. Weber et al., "Scalable Graph Learning for Anti-Money Laundering: A First Look," *arXiv:1812.00076*, 2018.

[3] Elliptic Co., "Elliptic Bitcoin Transaction Dataset," Kaggle, 2019. Available: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

[4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD 2016*, pp. 785–794. DOI: 10.1145/2939672.2939785.

[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

[6] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS 2017*, pp. 4765–4774.

[7] S. M. Lundberg, G. G. Erion, and S.-I. Lee, "Consistent Individualized Feature Attribution for Tree Ensembles," *arXiv:1802.03888*, 2018.

[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," *ICDM 2008*, pp. 413–422. DOI: 10.1109/ICDM.2008.17.

[9] A. Aric, D. A. Schult, and P. J. Swart, "Exploring Network Structure, Dynamics, and Function using NetworkX," *SciPy 2008*, pp. 11–15.

[10] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[11] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-Sampling Technique," *JAIR*, vol. 16, pp. 321–357, 2002.

[12] M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems," 2015. Available: https://www.tensorflow.org

[13] Streamlit Inc., "Streamlit: The Fastest Way to Build Data Apps," 2024. Available: https://streamlit.io

[14] Plotly Technologies Inc., "Plotly Python Graphing Library," 2024. Available: https://plotly.com/python

[15] FATF, "Virtual Assets and Virtual Asset Service Providers," FATF Guidance, Paris, 2019.

[16] C. Molnar, "Interpretable Machine Learning," 2nd ed., 2022. Available: https://christophm.github.io/interpretable-ml-book/

[17] L. Page et al., "The PageRank Citation Ranking: Bringing Order to the Web," Stanford InfoLab, 1999.

[18] D. P. Kingma and J. L. Ba, "Adam: A Method for Stochastic Optimization," *ICLR 2015*. arXiv:1412.6980.

[19] E. Kirkos et al., "Data Mining Techniques for Detection of Fraudulent Financial Statements," *Expert Systems with Applications*, vol. 32, no. 4, pp. 995–1003, 2007.

[20] R. Baniasadi et al., "Detecting Money Laundering in Bitcoin Using Machine Learning," *IIAI-AAI 2021*.
