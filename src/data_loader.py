"""
Data loading utilities for AML Detection System.
Provides functions to load raw and processed transaction datasets reliably.
"""

import os
import logging
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw", "HI-Small_Trans.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "aml_feature_engineered_v2.csv")


def load_raw_data(
    file_path: Optional[str] = None,
    nrows: Optional[int] = None,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Load the raw IBM AML transaction dataset (HI-Small_Trans.csv).
    
    Parameters
    ----------
    file_path : Optional[str]
        Path to raw CSV file. Defaults to 'data/raw/HI-Small_Trans.csv'.
    nrows : Optional[int]
        Number of rows to read (useful for testing and development).
    drop_duplicates : bool
        Whether to drop exact duplicate transaction rows.
        
    Returns
    -------
    pd.DataFrame
        Loaded raw transaction dataframe.
    """
    path = file_path or RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    logger.info(f"Loading raw dataset from {path} (nrows={nrows})...")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    logger.info(f"Loaded raw dataset shape: {df.shape}")

    if drop_duplicates:
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows.")

    return df


def load_processed_data(
    file_path: Optional[str] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load pre-engineered transaction dataset (aml_feature_engineered_v2.csv).
    
    Parameters
    ----------
    file_path : Optional[str]
        Path to processed CSV. Defaults to 'data/processed/aml_feature_engineered_v2.csv'.
    nrows : Optional[int]
        Number of rows to read.
        
    Returns
    -------
    pd.DataFrame
        Loaded feature engineered dataframe.
    """
    path = file_path or PROCESSED_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset file not found at: {path}")

    logger.info(f"Loading processed dataset from {path} (nrows={nrows})...")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    logger.info(f"Loaded processed dataset shape: {df.shape}")
    return df


def validate_raw_schema(df: pd.DataFrame) -> bool:
    """
    Verify that raw DataFrame has the expected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
        
    Returns
    -------
    bool
        True if schema is valid, raises ValueError otherwise.
    """
    expected_columns = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering"
    ]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataframe missing required columns: {missing}")
    return True
