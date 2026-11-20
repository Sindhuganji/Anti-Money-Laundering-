AML Project – Dataset Documentation

This project uses two datasets for Intelligent Anti-Money Laundering Detection Using Explainable AI:

1. IBM Transactions for Anti-Money Laundering Dataset
2. Elliptic Bitcoin Transaction Dataset

These datasets support the development and evaluation of machine learning models for detecting suspicious financial transactions and cryptocurrency-related illicit activities.

⸻

1. IBM Anti-Money Laundering Transaction Dataset

Dataset Information

* Dataset Name: IBM Transactions for Anti-Money Laundering (AML)
* Original Source: IBM
* Dataset Platform: Kaggle
* Kaggle Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
* License: CDLA-Sharing-1.0 (refer to the original dataset terms)

Dataset Files

File Name	Description
HI-Small_Trans.csv	Transaction-level financial data used for AML detection.

Download Instructions

Step 1: Install Kaggle CLI

pip install kaggle

Step 2: Authenticate with Kaggle

kaggle auth login

Step 3: Download the Dataset

mkdir -p data/raw
kaggle datasets download \
  -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  -p data/raw \
  --unzip

Refer to the original Kaggle dataset page for available files and download details.

Project Usage

The IBM AML dataset is used for:

* Transaction preprocessing and data cleaning.
* Handling missing values and categorical features.
* Feature engineering.
* Historical transaction behaviour analysis.
* Supervised machine learning classification.
* Anomaly detection.
* Model evaluation and explainability.

⸻

2. Elliptic Bitcoin Transaction Dataset

Dataset Information

* Dataset Name: Elliptic Bitcoin Transaction Dataset
* Original Publisher: Elliptic
* Dataset Platform: Kaggle
* Domain: Cryptocurrency Transaction Analysis
* Purpose: Identification of licit and illicit Bitcoin transactions.

Original Dataset Sources

* Elliptic Official Website: https://www.elliptic.co/insights/elliptic-dataset-cryptocurrency-financial-crime/
* Kaggle Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
* Dataset License: CC BY-NC-ND 4.0 (verify the current license and terms on the original Kaggle page).

Raw Dataset Files

The original dataset consists of three main files:

File Name	Description
elliptic_txs_classes.csv	Contains transaction IDs and their corresponding class labels.
elliptic_txs_edgelist.csv	Represents relationships and connections between Bitcoin transactions.
elliptic_txs_features.csv	Contains transaction features used for machine learning.

Dataset Components

1. Transaction Classes

Contains labels associated with Bitcoin transactions:

* 1 – Illicit transactions.
* 2 – Licit transactions.
* unknown – Transactions without an available class label.

2. Transaction Edgelist

Represents relationships between transactions.

Each edge indicates a relationship representing the flow of Bitcoin between transactions.

3. Transaction Features

Contains transaction-level and aggregated features used for classification and analysis.

Download Instructions

Step 1: Create the Dataset Directory

mkdir -p data/elliptic/raw

Step 2: Download the Dataset Using Kaggle CLI

kaggle datasets download \
  -d ellipticco/elliptic-data-set \
  -p data/elliptic/raw \
  --unzip

Step 3: Verify Downloaded Files

ls -lh data/elliptic/raw/

The directory should contain:

elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
elliptic_txs_features.csv

Processed Dataset Files

The following files are generated during the project’s preprocessing and feature engineering stages.

File Name	Description
elliptic_labelled.csv	Processed transactions with available labels.
elliptic_unknown.csv	Transactions with unknown or unavailable labels.
elliptic_features_engineered.csv	Processed dataset containing engineered features.
elliptic_metadata.json	Metadata associated with the processed dataset.

These are project-generated files and are not original files from the Kaggle dataset.

Project Usage

The Elliptic dataset is used for:

* Cryptocurrency transaction analysis.
* Transaction graph construction.
* Graph-based feature analysis.
* Licit and illicit transaction detection.
* Machine learning experimentation.
* Analysis of relationships between transactions.
* Identification of suspicious Bitcoin transaction patterns.

⸻

3. Dataset Directory Structure

The project maintains separate directories for raw and processed datasets.

data/
│
├── raw/
│   └── HI-Small_Trans.csv
│
├── processed/
│   ├── aml_preprocessed.csv
│   ├── aml_preprocessed_sample_v2.csv
│   ├── aml_feature_engineered.csv
│   ├── aml_feature_engineered_v2.csv
│   ├── aml_feature_engineered_version2.csv
│   ├── aml_train.csv
│   ├── aml_val.csv
│   ├── aml_test.csv
│   └── duplicate_check.db
│
└── elliptic/
    │
    ├── raw/
    │   ├── elliptic_txs_classes.csv
    │   ├── elliptic_txs_edgelist.csv
    │   └── elliptic_txs_features.csv
    │
    └── processed/
        ├── elliptic_labelled.csv
        ├── elliptic_unknown.csv
        ├── elliptic_features_engineered.csv
        └── elliptic_metadata.json

Note: The directory structure represents the dataset organization used in this project. Intermediate files may vary depending on preprocessing and experimentation.

⸻

4. Dataset Storage and GitHub Policy

The datasets and intermediate files are relatively large. Therefore, raw and processed dataset files are excluded from the GitHub repository.

The repository contains source code, documentation, model configurations, and other project files required to understand and reproduce the project workflow.

Storage Guidelines

* Large CSV files and database files are not committed to GitHub.
* Original datasets should be downloaded from their respective sources.
* Processed datasets can be generated by executing the project’s preprocessing scripts.
* Dataset metadata and documentation can be maintained separately.
* Users must follow the original dataset licenses and usage conditions.
* The Elliptic dataset and its derivatives should not be redistributed without checking the applicable license permissions.

⸻

5. Project Applications

Both datasets are used in the project to explore:

* Machine Learning
* Supervised Learning
* Unsupervised Learning
* Anomaly Detection
* Feature Engineering
* Transaction Behaviour Analysis
* Graph Analytics
* Cryptocurrency Transaction Analysis
* Financial Fraud Detection
* Explainable Artificial Intelligence (XAI)

⸻

6. Reproducibility

To reproduce the project:

1. Clone the GitHub repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Configure Kaggle authentication.
5. Download the original datasets from their respective sources.
6. Place the raw datasets in the appropriate directories.
7. Execute the preprocessing and feature engineering scripts.
8. Train and evaluate the machine learning models.
9. Run the application to explore AML detection results.

Install Dependencies

pip install -r requirements.txt

Create Virtual Environment

python3 -m venv .venv

Activate Virtual Environment

source .venv/bin/activate

⸻

7. Research Reference

The Elliptic dataset was introduced in the following research paper:

Weber et al. (2019)

Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.

The research explores the use of graph convolutional networks to identify illicit Bitcoin transactions.

Original Research Paper:
https://arxiv.org/abs/1908.02591

Dataset Publication:
https://www.elliptic.co/insights/elliptic-dataset-cryptocurrency-financial-crime/

⸻

8. Project Repository

GitHub Repository:

https://github.com/Sindhuganji/Anti-Money-Laundering-

⸻

Project Title: Intelligent Anti-Money Laundering Detection Using Explainable AI

Technologies: Python, Machine Learning, Anomaly Detection, Graph Analytics, Explainable AI, Streamlit.
