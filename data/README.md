# AML Project - Dataset Documentation

This project uses two datasets for intelligent anti-money laundering detection.

## 1. IBM AML Transaction Dataset

**Source:** IBM Transactions for Anti-Money Laundering

**Original Kaggle Dataset:**
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

**License:** CDLA-Sharing-1.0

### Dataset File
- HI-Small_Trans.csv

### Download

```bash
mkdir -p data/raw

kaggle datasets download \
  -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  -f HI-Small_Trans.csv \
  -p data/raw \
  --unzip
### 2. Verify formatting

```bash
cat data/README.md
