"""
Script to generate leakage-free Train/Val/Test datasets and persist the AML Preprocessing Pipeline.

Process:
1. Loads the feature-engineered dataset (or raw data).
2. Performs stratified split (64% train, 16% validation, 20% test, random_state=42).
3. Fits AMLPreprocessingPipeline (median imputer + RobustScaler) strictly on training features.
4. Generates:
   - data/processed/aml_train.csv
   - data/processed/aml_val.csv
   - data/processed/aml_test.csv
5. Persists pipeline artifacts to models/:
   - models/aml_pipeline_imputer.pkl
   - models/aml_pipeline_scaler.pkl
   - models/aml_pipeline_metadata.json
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

from src.data_loader import load_processed_data, load_raw_data
from src.preprocessing import split_data, AMLPreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.path.join("data", "processed", "aml_feature_engineered_v2.csv")
DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"


def run_pipeline_generation(sample_size: int = None):
    logger.info("Starting pipeline generation and dataset splitting...")
    
    # Load dataset
    df = load_processed_data(file_path=PROCESSED_DATA_PATH, nrows=sample_size)
    logger.info(f"Loaded dataset: {df.shape}")
    logger.info(f"Target distribution:\n{df['Is Laundering'].value_counts()}")

    # Perform stratified Train/Val/Test split (64% / 16% / 20%)
    train_df, val_df, test_df = split_data(
        df,
        target_col="Is Laundering",
        test_size=0.20,
        val_size=0.20,
        random_state=42,
        stratify=True
    )

    logger.info(f"Train set: {train_df.shape}, positives: {train_df['Is Laundering'].sum()}")
    logger.info(f"Val set:   {val_df.shape}, positives: {val_df['Is Laundering'].sum()}")
    logger.info(f"Test set:  {test_df.shape}, positives: {test_df['Is Laundering'].sum()}")

    # Fit pipeline exclusively on train_df
    pipeline = AMLPreprocessingPipeline(scaler_type="robust", impute_strategy="median")
    pipeline.fit(train_df)

    # Save pipeline artifacts
    pipeline.save(model_dir=MODELS_DIR, prefix="aml_pipeline")

    # Save train/val/test splits
    train_path = os.path.join(DATA_DIR, "aml_train.csv")
    val_path = os.path.join(DATA_DIR, "aml_val.csv")
    test_path = os.path.join(DATA_DIR, "aml_test.csv")

    logger.info(f"Saving train dataset to {train_path}...")
    train_df.to_csv(train_path, index=False)

    logger.info(f"Saving val dataset to {val_path}...")
    val_df.to_csv(val_path, index=False)

    logger.info(f"Saving test dataset to {test_path}...")
    test_df.to_csv(test_path, index=False)

    logger.info("Pipeline generation and dataset splitting completed successfully!")


if __name__ == "__main__":
    sample_size = None
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
    run_pipeline_generation(sample_size=sample_size)
