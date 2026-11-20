"""
Leakage-aware preprocessing for the IBM AML HI-Small dataset.

The raw data is split chronologically before fitting any learned preprocessing
components. Imputer/scaler are fitted on training data only.
"""
import os
import json
import logging
from typing import Optional, Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Is Laundering"
MODELS_DIR = "models"


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    train_fraction: float = 0.64,
    val_fraction: float = 0.16,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time, preserving row order within equal timestamps.

    train_fraction and val_fraction are fractions of the complete dataset.
    The remaining rows form the test set.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing required timestamp column: {timestamp_col}")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    result = df.copy()
    result["_source_order"] = np.arange(len(result))
    result["_parsed_timestamp"] = pd.to_datetime(result[timestamp_col], errors="coerce")
    if result["_parsed_timestamp"].isna().any():
        bad = int(result["_parsed_timestamp"].isna().sum())
        raise ValueError(f"{bad} rows have invalid timestamps; fix/drop them before splitting.")

    result = result.sort_values(
        ["_parsed_timestamp", "_source_order"], kind="mergesort"
    ).reset_index(drop=True)

    n = len(result)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    # Avoid splitting a timestamp group across partitions. This prevents
    # transactions at the same recorded time being split across train/val/test.
    def move_boundary_to_timestamp_end(boundary: int) -> int:
        if boundary <= 0 or boundary >= n:
            return boundary
        boundary_time = result.loc[boundary - 1, "_parsed_timestamp"]
        while boundary < n and result.loc[boundary, "_parsed_timestamp"] == boundary_time:
            boundary += 1
        return boundary

    train_end = move_boundary_to_timestamp_end(train_end)
    val_end = move_boundary_to_timestamp_end(max(val_end, train_end + 1))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Chronological split produced an empty partition; inspect timestamp distribution.")

    train = result.iloc[:train_end].copy()
    val = result.iloc[train_end:val_end].copy()
    test = result.iloc[val_end:].copy()

    for part in (train, val, test):
        part.drop(columns=["_source_order", "_parsed_timestamp"], inplace=True)

    def report(name: str, part: pd.DataFrame) -> None:
        positives = int(pd.to_numeric(part[TARGET_COL], errors="coerce").sum())
        logger.info("%s: rows=%s, positives=%s, positive_rate=%.6f",
                    name, len(part), positives, positives / max(1, len(part)))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    report("Train", train)
    report("Validation", val)
    report("Test", test)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class AMLPreprocessingPipeline:
    """Numeric imputation and scaling fitted exclusively on training features."""

    def __init__(
        self,
        scaler_type: str = "robust",
        impute_strategy: str = "median",
        target_col: str = TARGET_COL,
        feature_names: Optional[List[str]] = None,
    ):
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.target_col = target_col
        self.feature_names = feature_names
        self.imputer = SimpleImputer(strategy=impute_strategy)
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'robust' or 'minmax'.")
        self.frequency_encoders: Dict = {}
        self.is_fitted = False

    @staticmethod
    def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=["Is Laundering"], errors="ignore")
        # Raw identifiers and string categoricals must be encoded/removed upstream.
        X = X.select_dtypes(include=[np.number])
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, X_train: pd.DataFrame, frequency_encoders: Optional[Dict] = None):
        X_num = self._numeric_frame(X_train)
        if X_num.empty:
            raise ValueError("No numeric feature columns remain after preprocessing.")
        self.feature_names = list(X_num.columns)
        self.imputer.fit(X_num)
        self.scaler.fit(self.imputer.transform(X_num))
        self.frequency_encoders = frequency_encoders or {}
        self.is_fitted = True
        logger.info("Preprocessing fitted on %d training-only features.", len(self.feature_names))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        X_num = self._numeric_frame(X)
        missing = [c for c in self.feature_names if c not in X_num.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")
        X_num = X_num.reindex(columns=self.feature_names)
        transformed = self.scaler.transform(self.imputer.transform(X_num))
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame, frequency_encoders: Optional[Dict] = None):
        self.fit(X_train, frequency_encoders=frequency_encoders)
        return self.transform(X_train)

    def save(self, model_dir: str = MODELS_DIR, prefix: str = "aml_pipeline"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.imputer, os.path.join(model_dir, f"{prefix}_imputer.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, f"{prefix}_scaler.pkl"))
        with open(os.path.join(model_dir, f"{prefix}_freq_encoders.json"), "w") as f:
            json.dump(self.frequency_encoders, f, indent=2, default=str)
        metadata = {
            "scaler_type": self.scaler_type,
            "impute_strategy": self.impute_strategy,
            "target_col": self.target_col,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        with open(os.path.join(model_dir, f"{prefix}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: str = MODELS_DIR, prefix: str = "aml_pipeline"):
        paths = {
            "imputer": os.path.join(model_dir, f"{prefix}_imputer.pkl"),
            "scaler": os.path.join(model_dir, f"{prefix}_scaler.pkl"),
            "metadata": os.path.join(model_dir, f"{prefix}_metadata.json"),
        }
        for path in paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing preprocessing artifact: {path}")
        with open(paths["metadata"]) as f:
            meta = json.load(f)
        obj = cls(
            scaler_type=meta["scaler_type"],
            impute_strategy=meta["impute_strategy"],
            target_col=meta["target_col"],
            feature_names=meta["feature_names"],
        )
        obj.imputer = joblib.load(paths["imputer"])
        obj.scaler = joblib.load(paths["scaler"])
        enc_path = os.path.join(model_dir, f"{prefix}_freq_encoders.json")
        if os.path.exists(enc_path):
            with open(enc_path) as f:
                obj.frequency_encoders = json.load(f)
        obj.is_fitted = bool(meta.get("is_fitted", True))
        return obj
