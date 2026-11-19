"""
Preprocessing and data splitting module for Anti-Money Laundering (AML) detection.
Provides leakage-free train/validation/test splits, median imputation,
RobustScaler/MinMaxScaler transformations, and persistence of all preprocessing artifacts.
"""

import os
import json
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = 'Is Laundering'
MODELS_DIR = "models"
DATA_PROCESSED_DIR = os.path.join("data", "processed")


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into Train, Validation, and Test sets using stratified sampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    target_col : str
        Target variable name.
    test_size : float
        Proportion of total dataset for test set (default 0.20).
    val_size : float
        Proportion of remaining training data for validation set (default 0.20 -> 0.16 of total).
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to stratify by target variable.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df (64%), val_df (16%), test_df (20%)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in dataframe.")

    y = df[target_col]
    stratify_target = y if stratify else None

    # First split: train+val (80%) vs test (20%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    # Second split: train (80% of 80% = 64%) vs val (20% of 80% = 16%)
    stratify_train = train_val_df[target_col] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_train
    )

    logger.info(
        f"Data split summary:\n"
        f" - Train: {train_df.shape} (Laundering: {train_df[target_col].sum()})\n"
        f" - Val:   {val_df.shape} (Laundering: {val_df[target_col].sum()})\n"
        f" - Test:  {test_df.shape} (Laundering: {test_df[target_col].sum()})"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


class AMLPreprocessingPipeline:
    """
    Production-grade preprocessing pipeline for AML tabular data.
    Fits imputer and scaler strictly on training features to prevent leakage.
    Provides transform and inverse_transform methods.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        impute_strategy: str = "median",
        target_col: str = TARGET_COL,
        feature_names: Optional[List[str]] = None
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
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'minmax'.")

        self.frequency_encoders: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, frequency_encoders: Optional[Dict] = None):
        """
        Fit imputer and scaler on training feature matrix.
        """
        # Ensure target column is dropped if present
        if self.target_col in X_train.columns:
            X_train = X_train.drop(columns=[self.target_col])

        # Exclude non-numeric or raw identifier columns if any remain
        X_num = X_train.select_dtypes(include=[np.number])
        if 'From Bank' in X_num.columns:
            X_num = X_num.drop(columns=['From Bank'], errors='ignore')
        if 'To Bank' in X_num.columns:
            X_num = X_num.drop(columns=['To Bank'], errors='ignore')

        self.feature_names = list(X_num.columns)
        logger.info(f"Fitting AMLPreprocessingPipeline on {len(self.feature_names)} features...")

        X_imputed = self.imputer.fit_transform(X_num)
        self.scaler.fit(X_imputed)

        if frequency_encoders:
            self.frequency_encoders = frequency_encoders

        self.is_fitted = True
        logger.info("AMLPreprocessingPipeline fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix using fitted imputer and scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        # Exclude target if present
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        # Align with fitted feature columns
        missing_cols = [c for c in self.feature_names if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

        X_ordered = X[self.feature_names].copy()
        # Clean infinite values
        X_ordered = X_ordered.replace([np.inf, -np.inf], np.nan)

        X_imputed = self.imputer.transform(X_ordered)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.feature_names)

    def fit_transform(self, X_train: pd.DataFrame, frequency_encoders: Optional[Dict] = None) -> pd.DataFrame:
        """Fit on training data and return scaled DataFrame."""
        return self.fit(X_train, frequency_encoders).transform(X_train)

    def save(
        self,
        model_dir: str = MODELS_DIR,
        prefix: str = "aml_pipeline"
    ):
        """
        Persist pipeline components (imputer, scaler, feature names, frequency encoders).
        """
        os.makedirs(model_dir, exist_ok=True)
        imputer_path = os.path.join(model_dir, f"{prefix}_imputer.pkl")
        scaler_path = os.path.join(model_dir, f"{prefix}_scaler.pkl")
        encoders_path = os.path.join(model_dir, f"{prefix}_freq_encoders.json")
        metadata_path = os.path.join(model_dir, f"{prefix}_metadata.json")

        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.scaler, scaler_path)

        with open(encoders_path, "w") as f:
            json.dump(self.frequency_encoders, f, indent=4)

        metadata = {
            "scaler_type": self.scaler_type,
            "impute_strategy": self.impute_strategy,
            "target_col": self.target_col,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Pipeline persisted successfully to directory '{model_dir}' with prefix '{prefix}'.")

    @classmethod
    def load(
        cls,
        model_dir: str = MODELS_DIR,
        prefix: str = "aml_pipeline"
    ) -> "AMLPreprocessingPipeline":
        """
        Load persisted preprocessing pipeline from disk.
        """
        imputer_path = os.path.join(model_dir, f"{prefix}_imputer.pkl")
        scaler_path = os.path.join(model_dir, f"{prefix}_scaler.pkl")
        encoders_path = os.path.join(model_dir, f"{prefix}_freq_encoders.json")
        metadata_path = os.path.join(model_dir, f"{prefix}_metadata.json")

        for p in [imputer_path, scaler_path, metadata_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing pipeline artifact: {p}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        pipe = cls(
            scaler_type=meta["scaler_type"],
            impute_strategy=meta["impute_strategy"],
            target_col=meta["target_col"],
            feature_names=meta["feature_names"]
        )

        pipe.imputer = joblib.load(imputer_path)
        pipe.scaler = joblib.load(scaler_path)

        if os.path.exists(encoders_path):
            with open(encoders_path, "r") as f:
                pipe.frequency_encoders = json.load(f)

        pipe.is_fitted = meta.get("is_fitted", True)
        logger.info(f"Loaded AMLPreprocessingPipeline from {model_dir}/{prefix}_*")
        return pipe
