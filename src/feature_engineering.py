"""Leakage-aware feature engineering for IBM HI-Small AML transactions.

Historical account features are calculated strictly from earlier transactions.
For inference, pass history_df containing transactions strictly earlier than the
transactions being scored. Without history, historical values start at zero.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
TARGET = "Is Laundering"
ENGINEERED_FEATURE_NAMES = [
    "Amount Received", "Amount Paid", "Amount_Difference",
    "Absolute_Amount_Difference", "Amount_Ratio", "Log_Amount_Received",
    "Log_Amount_Paid", "Transaction_Hour", "Transaction_Day",
    "Transaction_Month", "Transaction_Weekday", "Is_Weekend",
    "Is_Night_Transaction", "Same_Currency", "Same_Bank",
    "Source_Transaction_Frequency", "Destination_Transaction_Frequency",
    "Previous_Source_Transactions", "Previous_Source_Amount",
    "Previous_Destination_Transactions", "Previous_Destination_Amount",
    "Receiving Currency_Frequency", "Payment Currency_Frequency",
    "Payment Format_Frequency",
]
CAT_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]


def _numeric_amounts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("Amount Paid", "Amount Received"):
        if col not in out:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float64")
    return out


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Timestamp" not in out:
        raise ValueError("Raw IBM transactions must contain 'Timestamp'.")
    dt = pd.to_datetime(out["Timestamp"], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Timestamp contains {int(dt.isna().sum())} invalid values.")
    out["Transaction_Hour"] = dt.dt.hour.astype("int8")
    out["Transaction_Day"] = dt.dt.day.astype("int8")
    out["Transaction_Month"] = dt.dt.month.astype("int8")
    out["Transaction_Weekday"] = dt.dt.dayofweek.astype("int8")
    out["Is_Weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    out["Is_Night_Transaction"] = ((dt.dt.hour >= 22) | (dt.dt.hour < 6)).astype("int8")
    return out


def extract_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _numeric_amounts(df)
    paid, received = out["Amount Paid"], out["Amount Received"]
    out["Amount_Difference"] = paid - received
    out["Absolute_Amount_Difference"] = (paid - received).abs()
    out["Amount_Ratio"] = paid / received.replace(0, np.nan)
    out["Amount_Ratio"] = out["Amount_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["Log_Amount_Received"] = np.log1p(received.clip(lower=0))
    out["Log_Amount_Paid"] = np.log1p(paid.clip(lower=0))
    return out


def extract_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Same_Currency"] = (
        out["Receiving Currency"].astype(str).eq(out["Payment Currency"].astype(str)).astype("int8")
        if {"Receiving Currency", "Payment Currency"}.issubset(out.columns) else 0
    )
    out["Same_Bank"] = (
        out["From Bank"].astype(str).eq(out["To Bank"].astype(str)).astype("int8")
        if {"From Bank", "To Bank"}.issubset(out.columns) else 0
    )
    return out


def extract_historical_features(df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """For each row, calculate account counts/amounts before that row only.

    history_df is optional prior context and must contain no transaction at or
    after the earliest transaction in df. Ties are ordered by input row order.
    """
    current = _numeric_amounts(df)
    current = current.copy()
    current["__is_current"] = True
    current["__row_order"] = np.arange(len(current))
    if history_df is not None and len(history_df):
        hist = _numeric_amounts(history_df.copy())
        if "Timestamp" not in hist:
            raise ValueError("history_df must contain Timestamp.")
        hist_times = pd.to_datetime(hist["Timestamp"], errors="coerce")
        current_times = pd.to_datetime(current["Timestamp"], errors="coerce")
        if hist_times.isna().any() or current_times.isna().any():
            raise ValueError("history_df/current transactions contain invalid timestamps.")
        if hist_times.max() >= current_times.min():
            raise ValueError("history_df must contain only transactions strictly earlier than the scored batch.")
        hist["__is_current"] = False
        hist["__row_order"] = -1 - np.arange(len(hist))
        combined = pd.concat([hist, current], ignore_index=True, sort=False)
    else:
        combined = current

    if "Timestamp" not in combined:
        raise ValueError("Historical feature calculation requires Timestamp.")
    combined["__time"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
    if combined["__time"].isna().any():
        raise ValueError("Invalid Timestamp encountered while building historical features.")
    combined = combined.sort_values(["__time", "__is_current", "__row_order"], kind="mergesort").reset_index(drop=True)

    for account_col, amount_col, count_name, sum_name in [
        ("Account", "Amount Paid", "Previous_Source_Transactions", "Previous_Source_Amount"),
        ("Account.1", "Amount Received", "Previous_Destination_Transactions", "Previous_Destination_Amount"),
    ]:
        if account_col in combined:
            grp = combined.groupby(account_col, sort=False, dropna=False)
            combined[count_name] = grp.cumcount().astype("float32")
            combined[sum_name] = (grp[amount_col].cumsum() - combined[amount_col]).astype("float32")
        else:
            combined[count_name] = np.float32(0)
            combined[sum_name] = np.float32(0)

    # Past-only account activity replaces full-period account frequency counts.
    combined["Source_Transaction_Frequency"] = np.log1p(combined["Previous_Source_Transactions"].clip(lower=0)).astype("float32")
    combined["Destination_Transaction_Frequency"] = np.log1p(combined["Previous_Destination_Transactions"].clip(lower=0)).astype("float32")
    result = combined.loc[combined["__is_current"]].sort_values("__row_order")
    result = result.drop(columns=["__is_current", "__row_order", "__time"], errors="ignore")
    return result.reset_index(drop=True)


def fit_frequency_encoders(train_df: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Fit normalized category frequencies on training data only."""
    encoders: Dict[str, Dict[str, float]] = {}
    for col in (categorical_cols or CAT_COLS):
        if col in train_df:
            vals = train_df[col].fillna("__MISSING__").astype(str)
            encoders[col] = vals.value_counts(normalize=True).to_dict()
    return encoders


def apply_frequency_encoders(df: pd.DataFrame, encoders: Dict[str, Dict[str, float]], default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for col in CAT_COLS:
        if col in out:
            mapping = encoders.get(col, {})
            out[f"{col}_Frequency"] = out[col].fillna("__MISSING__").astype(str).map(mapping).fillna(default).astype("float32")
        else:
            out[f"{col}_Frequency"] = np.float32(default)
    # These are past-only counts generated by extract_historical_features.
    for col in ("Source_Transaction_Frequency", "Destination_Transaction_Frequency"):
        if col not in out:
            out[col] = np.float32(0)
    return out


def build_features(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, Dict[str, float]]] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Build model features. `history_df` supplies transactions before this batch."""
    if df.empty:
        raise ValueError("Cannot build features for an empty transaction DataFrame.")
    raw = df.copy().reset_index(drop=True)
    feat = extract_temporal_features(raw)
    feat = extract_amount_features(feat)
    feat = extract_relational_features(feat)
    feat = extract_historical_features(feat, history_df=history_df)
    if encoders is None:
        encoders = fit_frequency_encoders(raw)
    feat = apply_frequency_encoders(feat, encoders)
    missing = [c for c in ENGINEERED_FEATURE_NAMES if c not in feat]
    if missing:
        raise ValueError(f"Feature engineering failed to create: {missing}")
    return feat[ENGINEERED_FEATURE_NAMES + ([TARGET] if TARGET in feat else [])].copy(), encoders
