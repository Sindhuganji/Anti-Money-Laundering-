"""Chronological splitting and train-fitted preprocessing for IBM AML."""
import os, json, logging
from typing import Optional, Tuple, List, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)
TARGET_COL = "Is Laundering"
MODELS_DIR = "models"


def chronological_split(df: pd.DataFrame, timestamp_col: str = "Timestamp", train_fraction: float = .64, val_fraction: float = .16):
    if timestamp_col not in df or TARGET_COL not in df:
        raise ValueError(f"Dataset must contain {timestamp_col!r} and {TARGET_COL!r}.")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Invalid chronological split fractions.")
    x = df.copy()
    x["__order"] = np.arange(len(x))
    x["__time"] = pd.to_datetime(x[timestamp_col], errors="coerce")
    if x["__time"].isna().any():
        raise ValueError(f"Found {int(x['__time'].isna().sum())} invalid timestamps.")
    x = x.sort_values(["__time", "__order"], kind="mergesort").reset_index(drop=True)
    n = len(x)
    a, b = int(n * train_fraction), int(n * (train_fraction + val_fraction))
    # Do not split identical timestamp groups across partitions.
    def boundary_end(i):
        if i <= 0 or i >= n: return i
        t = x.loc[i - 1, "__time"]
        while i < n and x.loc[i, "__time"] == t: i += 1
        return i
    a = boundary_end(a); b = boundary_end(max(b, a + 1))
    if not 0 < a < b < n:
        raise ValueError("Chronological split produced an empty partition.")
    parts = [x.iloc[:a].copy(), x.iloc[a:b].copy(), x.iloc[b:].copy()]
    for p in parts: p.drop(columns=["__order", "__time"], inplace=True)
    for name, p in zip(("Train", "Validation", "Test"), parts):
        positives = int(pd.to_numeric(p[TARGET_COL]).sum())
        logger.info("%s: rows=%d positives=%d positive_rate=%.6f", name, len(p), positives, positives / len(p))
    return tuple(p.reset_index(drop=True) for p in parts)


class AMLPreprocessingPipeline:
    def __init__(self, scaler_type="robust", impute_strategy="median", target_col=TARGET_COL, feature_names: Optional[List[str]]=None):
        self.scaler_type, self.impute_strategy, self.target_col = scaler_type, impute_strategy, target_col
        self.feature_names = feature_names
        self.imputer = SimpleImputer(strategy=impute_strategy)
        if scaler_type == "robust": self.scaler = RobustScaler()
        elif scaler_type == "minmax": self.scaler = MinMaxScaler()
        else: raise ValueError("scaler_type must be 'robust' or 'minmax'.")
        self.frequency_encoders: Dict = {}
        self.is_fitted = False

    @staticmethod
    def _numeric_frame(X):
        X = X.copy().drop(columns=[TARGET_COL], errors="ignore")
        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, X_train, frequency_encoders=None):
        X = self._numeric_frame(X_train)
        if X.empty: raise ValueError("No numeric features available for preprocessing.")
        self.feature_names = list(X.columns)
        self.imputer.fit(X)
        self.scaler.fit(self.imputer.transform(X))
        self.frequency_encoders = frequency_encoders or {}
        self.is_fitted = True
        logger.info("Preprocessing fitted on %d training-only features.", len(self.feature_names))
        return self

    def transform(self, X):
        if not self.is_fitted: raise RuntimeError("Preprocessing pipeline is not fitted.")
        numeric = self._numeric_frame(X)
        missing = [c for c in self.feature_names if c not in numeric]
        if missing: raise ValueError(f"Input DataFrame is missing required features: {missing}")
        numeric = numeric.reindex(columns=self.feature_names)
        values = self.scaler.transform(self.imputer.transform(numeric))
        return pd.DataFrame(values, columns=self.feature_names, index=X.index)

    def fit_transform(self, X_train, frequency_encoders=None):
        return self.fit(X_train, frequency_encoders).transform(X_train)

    def save(self, model_dir=MODELS_DIR, prefix="aml_pipeline"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.imputer, os.path.join(model_dir, f"{prefix}_imputer.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, f"{prefix}_scaler.pkl"))
        with open(os.path.join(model_dir, f"{prefix}_freq_encoders.json"), "w") as f: json.dump(self.frequency_encoders, f, indent=2)
        with open(os.path.join(model_dir, f"{prefix}_metadata.json"), "w") as f:
            json.dump({"scaler_type": self.scaler_type, "impute_strategy": self.impute_strategy,
                       "target_col": self.target_col, "feature_names": self.feature_names,
                       "is_fitted": self.is_fitted}, f, indent=2)

    @classmethod
    def load(cls, model_dir=MODELS_DIR, prefix="aml_pipeline"):
        meta_path = os.path.join(model_dir, f"{prefix}_metadata.json")
        if not os.path.exists(meta_path): raise FileNotFoundError(f"Missing preprocessing metadata: {meta_path}")
        with open(meta_path) as f: meta = json.load(f)
        obj = cls(meta["scaler_type"], meta["impute_strategy"], meta["target_col"], meta["feature_names"])
        for attr, suffix in (("imputer", "imputer.pkl"), ("scaler", "scaler.pkl")):
            path = os.path.join(model_dir, f"{prefix}_{suffix}")
            if not os.path.exists(path): raise FileNotFoundError(f"Missing preprocessing artifact: {path}")
            setattr(obj, attr, joblib.load(path))
        enc_path = os.path.join(model_dir, f"{prefix}_freq_encoders.json")
        if os.path.exists(enc_path):
            with open(enc_path) as f: obj.frequency_encoders = json.load(f)
        obj.is_fitted = bool(meta.get("is_fitted", True))
        return obj


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
):
    """Backward-compatible stratified split for existing project modules.

    New IBM AML training should use chronological_split() instead.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present.")

    y = df[target_col]
    stratify_target = y if stratify else None

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    train_stratify = train_val[target_col] if stratify else None
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_stratify,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
