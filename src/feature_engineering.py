"""
Feature engineering module for Anti-Money Laundering (AML) detection.
Builds domain-specific features without data leakage:
- Financial transaction metrics (difference, ratio, log amounts)
- Temporal features (hour, day, weekday, is_weekend, is_night)
- Relational features (same_currency, same_bank)
- Account activity and frequency features (fitted on train or pre-split mapping)
- Historical cumulative activity features (time-ordered)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENGINEERED_FEATURE_NAMES = [
    'Amount Received',
    'Amount Paid',
    'Amount_Difference',
    'Absolute_Amount_Difference',
    'Amount_Ratio',
    'Log_Amount_Received',
    'Log_Amount_Paid',
    'Transaction_Hour',
    'Transaction_Day',
    'Transaction_Month',
    'Transaction_Weekday',
    'Is_Weekend',
    'Is_Night_Transaction',
    'Same_Currency',
    'Same_Bank',
    'Source_Transaction_Frequency',
    'Destination_Transaction_Frequency',
    'Previous_Source_Transactions',
    'Previous_Source_Amount',
    'Previous_Destination_Transactions',
    'Previous_Destination_Amount',
    'Receiving Currency_Frequency',
    'Payment Currency_Frequency',
    'Payment Format_Frequency'
]


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date and time features from Timestamp column.
    """
    df = df.copy()
    if 'Timestamp' in df.columns:
        dt = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Transaction_Hour'] = dt.dt.hour.fillna(0).astype('int8')
        df['Transaction_Day'] = dt.dt.day.fillna(1).astype('int8')
        df['Transaction_Month'] = dt.dt.month.fillna(9).astype('int8')
        df['Transaction_Weekday'] = dt.dt.dayofweek.fillna(0).astype('int8')
        df['Is_Weekend'] = (df['Transaction_Weekday'] >= 5).astype('int8')
        df['Is_Night_Transaction'] = (
            (df['Transaction_Hour'] >= 22) | (df['Transaction_Hour'] < 6)
        ).astype('int8')
    return df


def extract_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate financial transaction amount relationship features.
    """
    df = df.copy()
    paid = pd.to_numeric(df['Amount Paid'], errors='coerce').fillna(0.0)
    received = pd.to_numeric(df['Amount Received'], errors='coerce').fillna(0.0)

    df['Amount Paid'] = paid
    df['Amount Received'] = received
    df['Amount_Difference'] = paid - received
    df['Absolute_Amount_Difference'] = df['Amount_Difference'].abs()
    df['Amount_Ratio'] = paid / (received + 1e-8)
    df['Log_Amount_Received'] = np.log1p(np.maximum(0.0, received))
    df['Log_Amount_Paid'] = np.log1p(np.maximum(0.0, paid))
    return df


def extract_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relational features: same currency, same bank.
    """
    df = df.copy()
    if 'Receiving Currency' in df.columns and 'Payment Currency' in df.columns:
        df['Same_Currency'] = (
            df['Receiving Currency'] == df['Payment Currency']
        ).astype('int8')
    else:
        df['Same_Currency'] = 0

    if 'From Bank' in df.columns and 'To Bank' in df.columns:
        df['Same_Bank'] = (
            df['From Bank'] == df['To Bank']
        ).astype('int8')
    else:
        df['Same_Bank'] = 0

    return df


def extract_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-ordered historical activity per account:
    - Previous_Source_Transactions
    - Previous_Source_Amount
    - Previous_Destination_Transactions
    - Previous_Destination_Amount
    
    Operates strictly chronologically to avoid future look-ahead.
    """
    df = df.copy()
    df['_orig_order'] = np.arange(len(df))

    if 'Timestamp' in df.columns:
        df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values(['Timestamp_dt', '_orig_order']).reset_index(drop=True)

    # Source account historical activity
    if 'Account' in df.columns:
        source_group = df.groupby('Account', sort=False)
        df['Previous_Source_Transactions'] = source_group.cumcount().astype('float32')
        df['Previous_Source_Amount'] = (
            source_group['Amount Paid'].cumsum() - df['Amount Paid']
        ).astype('float32')
    else:
        df['Previous_Source_Transactions'] = 0.0
        df['Previous_Source_Amount'] = 0.0

    # Destination account historical activity
    if 'Account.1' in df.columns:
        dest_group = df.groupby('Account.1', sort=False)
        df['Previous_Destination_Transactions'] = dest_group.cumcount().astype('float32')
        df['Previous_Destination_Amount'] = (
            dest_group['Amount Received'].cumsum() - df['Amount Received']
        ).astype('float32')
    else:
        df['Previous_Destination_Transactions'] = 0.0
        df['Previous_Destination_Amount'] = 0.0

    # Restore original ordering
    df = df.sort_values('_orig_order').reset_index(drop=True)
    df.drop(columns=['_orig_order'], inplace=True, errors='ignore')
    if 'Timestamp_dt' in df.columns:
        df.drop(columns=['Timestamp_dt'], inplace=True, errors='ignore')

    return df


def fit_frequency_encoders(
    train_df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    identifier_cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Fit frequency encodings exclusively on training data to prevent data leakage.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    categorical_cols : Optional[List[str]]
        Categorical columns to compute normalized frequencies for.
    identifier_cols : Optional[List[str]]
        Account ID columns to compute raw counts for.
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of mappings {col_name: {category: frequency/count}}.
    """
    cat_cols = categorical_cols or [
        'Receiving Currency',
        'Payment Currency',
        'Payment Format'
    ]
    id_cols = identifier_cols or [
        'Account',
        'Account.1'
    ]

    encoders: Dict[str, Dict[str, float]] = {}

    for col in cat_cols:
        if col in train_df.columns:
            freq = train_df[col].value_counts(normalize=True).to_dict()
            encoders[col] = freq

    for col in id_cols:
        if col in train_df.columns:
            counts = train_df[col].value_counts().to_dict()
            encoders[col] = counts

    return encoders


def apply_frequency_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, Dict[str, float]],
    default_cat_val: float = 0.0,
    default_id_val: float = 1.0
) -> pd.DataFrame:
    """
    Apply pre-computed training frequency mappings to any dataframe (train/val/test/inference).
    Unseen categories/accounts safely default to 0.0 or 1.0 without leakage.
    """
    df = df.copy()

    # Categorical normalized frequencies
    cat_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in cat_cols:
        if col in encoders:
            mapping = encoders[col]
            df[col + '_Frequency'] = df[col].map(mapping).fillna(default_cat_val).astype('float32')

    # Account frequencies
    if 'Account' in encoders:
        df['Source_Transaction_Frequency'] = (
            df['Account'].map(encoders['Account']).fillna(default_id_val).astype('float32')
        )
    if 'Account.1' in encoders:
        df['Destination_Transaction_Frequency'] = (
            df['Account.1'].map(encoders['Account.1']).fillna(default_id_val).astype('float32')
        )

    return df


def build_features(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    End-to-end feature engineering pipeline for a single DataFrame.
    If encoders are None, they are fitted on this DataFrame (for training).
    If encoders are provided, they are applied (for val/test/inference).
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]
        Transformed DataFrame with engineered features, and fitted encoders.
    """
    logger.info("Executing feature engineering pipeline...")
    df_feat = extract_temporal_features(df)
    df_feat = extract_amount_features(df_feat)
    df_feat = extract_relational_features(df_feat)
    df_feat = extract_historical_features(df_feat)

    if encoders is None:
        logger.info("Fitting frequency encoders on provided DataFrame...")
        encoders = fit_frequency_encoders(df_feat)

    df_feat = apply_frequency_encoders(df_feat, encoders)

    # Retain final engineered feature columns + Target if present
    target_col = 'Is Laundering'
    keep_cols = [c for c in ENGINEERED_FEATURE_NAMES if c in df_feat.columns]
    if target_col in df_feat.columns:
        keep_cols.append(target_col)

    output_df = df_feat[keep_cols].copy()
    logger.info(f"Feature engineering completed. Output shape: {output_df.shape}")
    return output_df, encoders
