# AML Project - Dataset Documentation

This project uses two datasets for intelligent anti-money laundering detection.

## 1. IBM AML Transaction Dataset

**Source:** IBM Transactions for Anti-Money Laundering

**Kaggle:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

**License:** CDLA-Sharing-1.0

### Dataset file

- HI-Small_Trans.csv

### Download

```bash
mkdir -p data/raw

kaggle datasets download \
  -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  -f HI-Small_Trans.csv \
  -p data/raw \
  --unzip
## 2. Elliptic Bitcoin Transaction Dataset

Used for cryptocurrency transaction analysis, graph-based learning and illicit transaction detection.

### Raw files

- elliptic_txs_classes.csv
- elliptic_txs_edgelist.csv
- elliptic_txs_features.csv

### Processed files

- elliptic_labelled.csv
- elliptic_unknown.csv
- elliptic_features_engineered.csv

## Dataset Storage

Large datasets are not stored directly in this GitHub repository because of file-size restrictions.

Refer to the original dataset sources for licensing and usage conditions.

## Purpose

Both datasets are used for academic research and experimentation in machine learning, anomaly detection, graph analytics and explainable AI.
