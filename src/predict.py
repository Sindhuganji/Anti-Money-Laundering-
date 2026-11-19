"""
Unified Inference Engine for Anti-Money Laundering (AML) Detection.

Features:
- Dynamically loads trained Random Forest or XGBoost models along with validation-tuned thresholds.
- Integrates Isolation Forest and Autoencoder V2 for unsupervised anomaly detection.
- Applies consistent preprocessing and feature engineering without leakage.
- Generates probability, risk score (0-100), and discrete risk categories (Low, Medium, High, Critical).
- Generates SHAP local waterfall plots and natural language compliance explanations.
- Supports single-transaction, batch, and CSV file inference.
"""

import os
import json
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Safe environment paths for sandbox
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from src.data_loader import validate_raw_schema
from src.feature_engineering import build_features, apply_frequency_encoders
from src.preprocessing import AMLPreprocessingPipeline
from src.risk_scoring import calculate_risk_score, assign_risk_category, evaluate_transaction_risk
from src.explainability import AMLExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"


class AMLPredictor:
    """
    Unified predictor supporting model selection (Random Forest vs XGBoost),
    anomaly scoring (Isolation Forest & Autoencoder V2), and SHAP explanations.
    """

    def __init__(
        self,
        model_name: str = "XGBoost",
        models_dir: str = MODELS_DIR
    ):
        self.models_dir = models_dir
        self.pipeline = AMLPreprocessingPipeline.load(model_dir=models_dir, prefix="aml_pipeline")
        self.feature_names = self.pipeline.feature_names
        self.set_model(model_name)

        # Lazy loaded components
        self._explainer: Optional[AMLExplainer] = None
        self._if_model = None
        self._if_imputer = None
        self._ae_model = None
        self._ae_scaler = None
        self._ae_imputer = None

    def set_model(self, model_name: str):
        """Switch active supervised model between XGBoost and Random Forest."""
        self.model_name = model_name
        if model_name.lower().replace(" ", "") in ["xgboost", "xgb"]:
            model_file = os.path.join(self.models_dir, "xgboost_model.pkl")
            thresh_file = os.path.join(self.models_dir, "xgboost_threshold.json")
            self.model_key = "xgboost"
        elif model_name.lower().replace(" ", "") in ["randomforest", "rf"]:
            model_file = os.path.join(self.models_dir, "random_forest_model.pkl")
            thresh_file = os.path.join(self.models_dir, "random_forest_threshold.json")
            self.model_key = "random_forest"
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose 'XGBoost' or 'Random Forest'.")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = joblib.load(model_file)

        with open(thresh_file, "r") as f:
            thresh_meta = json.load(f)
            self.threshold = float(thresh_meta["optimal_threshold"])

        self._explainer = None  # Reset explainer for new model
        logger.info(f"Active model set to '{self.model_name}' (Threshold: {self.threshold:.4f})")

    @property
    def explainer(self) -> AMLExplainer:
        """Lazy load SHAP explainer for the active model."""
        if self._explainer is None:
            model_file = os.path.join(self.models_dir, f"{self.model_key}_model.pkl")
            thresh_file = os.path.join(self.models_dir, f"{self.model_key}_threshold.json")
            self._explainer = AMLExplainer(
                model_path=model_file,
                threshold_path=thresh_file,
                pipeline_prefix="aml_pipeline",
                model_name=self.model_name
            )
        return self._explainer

    def _prepare_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Check if input is raw transactions or already feature engineered,
        and apply the appropriate transformation pipeline.
        """
        df = df_input.copy()
        raw_cols = ["Timestamp", "Account", "Account.1"]
        is_raw = any(col in df.columns for col in raw_cols)

        if is_raw:
            # Build domain features and map training frequency encoders
            fe_df, _ = build_features(df, encoders=self.pipeline.frequency_encoders)
            X_transformed = self.pipeline.transform(fe_df)
        else:
            # Already feature-engineered table; align and transform
            X_transformed = self.pipeline.transform(df)

        return X_transformed

    def predict_single(
        self,
        transaction_dict: Dict[str, Any],
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Predict suspiciousness, probability, risk score and explanations for one transaction.
        """
        df_single = pd.DataFrame([transaction_dict])
        X = self._prepare_features(df_single)

        prob = float(self.model.predict_proba(X)[0, 1])
        risk_info = evaluate_transaction_risk(prob, threshold=self.threshold)

        result = {
            "model_used": self.model_name,
            "prediction": "Suspicious" if risk_info["is_suspicious"] else "Legitimate",
            "is_suspicious": risk_info["is_suspicious"],
            "probability": risk_info["probability"],
            "threshold": risk_info["threshold"],
            "risk_score": risk_info["risk_score"],
            "risk_category": risk_info["risk_category"],
            "risk_color": risk_info["risk_color"],
            "investigation_recommended": risk_info["investigation_recommended"]
        }

        if explain:
            explanation = self.explainer.explain_transaction(X)
            result["explanation"] = explanation

        return result

    def predict_batch(
        self,
        df_transactions: pd.DataFrame,
        include_anomaly_scores: bool = False
    ) -> pd.DataFrame:
        """
        Batch prediction for a dataframe of transactions.
        Returns original dataframe augmented with predictions, risk scores, and categories.
        """
        X = self._prepare_features(df_transactions)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        risk_scores = calculate_risk_score(probs, calibrated_threshold=self.threshold)
        categories = [assign_risk_category(s)[0] for s in risk_scores]

        output_df = df_transactions.copy()
        output_df["Predicted_Class"] = np.where(preds == 1, "Suspicious", "Legitimate")
        output_df["Laundering_Probability"] = np.round(probs, 4)
        output_df["Risk_Score"] = risk_scores
        output_df["Risk_Category"] = categories
        output_df["Alert_Triggered"] = preds == 1

        if include_anomaly_scores:
            if_scores = self.get_isolation_forest_scores(df_transactions)
            if if_scores is not None:
                output_df["Isolation_Forest_Anomaly"] = if_scores

        return output_df

    def get_isolation_forest_scores(self, df_input: pd.DataFrame) -> Optional[np.ndarray]:
        """Get anomaly flags from Isolation Forest."""
        if_path = os.path.join(self.models_dir, "isolation_forest_model.pkl")
        if_imp_path = os.path.join(self.models_dir, "isolation_forest_imputer.pkl")
        if not os.path.exists(if_path) or not os.path.exists(if_imp_path):
            return None

        if self._if_model is None:
            self._if_model = joblib.load(if_path)
            self._if_imputer = joblib.load(if_imp_path)

        try:
            X = self._prepare_features(df_input)
            X_imp = self._if_imputer.transform(X)
            # Isolation forest returns -1 for outlier, 1 for inlier
            preds = self._if_model.predict(X_imp)
            return np.where(preds == -1, "Anomaly", "Normal")
        except Exception as e:
            logger.warning(f"Failed to calculate Isolation Forest score: {e}")
            return None
