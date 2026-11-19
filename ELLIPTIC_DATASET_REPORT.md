# Elliptic Bitcoin Transaction Dataset Acquisition and Validation Report

## Executive Summary
This report documents the validation and acquisition of the **Elliptic Bitcoin Transaction Dataset** acquired for cryptocurrency anti-money laundering (AML) detection. The dataset consists of node transactions, directed payment flow edges, and temporal timestamps across 49 distinct time steps.

---

## 1. File Inventory and Integrity
| File Identifier | Path | Size (MB) | Raw Row Count | Column Count | Header Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classes** | `data/elliptic/raw/elliptic_txs_classes.csv` | 3.15 MB | 203,769 | 2 | Yes (`txId`, `class`) |
| **Edgelist** | `data/elliptic/raw/elliptic_txs_edgelist.csv` | 4.26 MB | 234,355 | 2 | Yes (`txId1`, `txId2`) |
| **Features** | `data/elliptic/raw/elliptic_txs_features.csv` | 657.73 MB | 203,769 | 167 | No (Mapped to `txId`, `time_step`, `feature_1`..`feature_165`) |

---

## 2. Schema and Column Specifications
1. **`elliptic_txs_classes.csv`**:
   - `txId`: Unique transaction ID (int64).
   - `class`: Classification label string:
     - `'1'`: **Illicit** (4,545 transactions, 2.23%)
     - `'2'`: **Licit** (42,019 transactions, 20.62%)
     - `'unknown'`: **Unknown** (157,205 transactions, 77.15%)
2. **`elliptic_txs_edgelist.csv`**:
   - `txId1`: Source transaction node ID (int64).
   - `txId2`: Destination transaction node ID (int64).
   - Unique source nodes: 166,345
   - Unique destination nodes: 148,447
   - Total distinct transaction nodes participating in graph edges: 203,769
3. **`elliptic_txs_features.csv`**:
   - `txId`: Unique transaction node identifier.
   - `time_step`: Discrete temporal timestep index (1 to 49).
   - `feature_1` to `feature_93`: Local transaction features (e.g. fees, transaction volume, transaction inputs/outputs).
   - `feature_94` to `feature_165`: Aggregated 1-hop neighborhood features (mean, std, min, max of neighbor transactions).

---

## 3. Data Quality, Missing Values, and Duplicates
- **Missing / Null Values**:
  - `classes`: 0 null entries.
  - `edgelist`: 0 null entries.
  - `features`: 0 null entries across all 167 columns.
- **Duplicate Transaction IDs**:
  - `classes.csv`: 0 duplicate IDs.
  - `features.csv`: 0 duplicate IDs.
  - `edgelist.csv`: 0 exact duplicate directed edges.

---

## 4. Cross-File Transaction ID Consistency
- **Classes vs Features Matching**:
  - Transactions in `classes` missing in `features`: **0**
  - Transactions in `features` missing in `classes`: **0**
  - Perfect 1:1 correspondence across all 203,769 nodes confirmed.
- **Graph Edges vs Features Matching**:
  - Graph nodes in `edgelist` missing from `features`: **0**
  - Every transaction appearing in the directed edge network is present in the feature matrix.

---

## 5. Architectural & Design Conclusions
1. **Ground-Truth Label Strictness**: In accordance with the Anti-Money Laundering specifications, the 157,205 `unknown` transactions are **strictly excluded** from supervised training ground truth. They are preserved for unsupervised graph propagation, network context, and anomaly scoring.
2. **Temporal Integrity**: The transactions span 49 discrete time steps. All model validation must adhere to strict temporal splits (e.g., training on early time steps, evaluating on later time steps) without future-to-past data leakage.
3. **Pipeline Isolation**: The Elliptic cryptocurrency schema is completely decoupled from the IBM banking transaction schema, operating inside `src/elliptic/` with dedicated model artifacts in `models/elliptic/`.
