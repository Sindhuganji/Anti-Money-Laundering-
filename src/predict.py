"""IBM AML inference engine compatible with the corrected training pipeline.

For accurate historical features, pass prior transactions through history_df.
Without history_df, history-derived features are zero for a single transaction.
"""
import os, json, logging, joblib
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.feature_engineering import build_features
from src.preprocessing import AMLPreprocessingPipeline
from src.risk_scoring import calculate_risk_score, assign_risk_category, evaluate_transaction_risk
from src.explainability import AMLExplainer

logger = logging.getLogger(__name__)
MODELS_DIR = "models"

class AMLPredictor:
    def __init__(self, model_name="XGBoost", models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.pipeline = AMLPreprocessingPipeline.load(models_dir, "aml_pipeline")
        self.feature_names = self.pipeline.feature_names
        self._explainer = None
        self._if_model = self._if_imputer = None
        self.set_model(model_name)

    def set_model(self, model_name):
        normalized = model_name.lower().replace(" ", "")
        if normalized in ("xgboost", "xgb"): key = "xgboost"
        elif normalized in ("randomforest", "rf"): key = "random_forest"
        else: raise ValueError("Choose XGBoost or Random Forest.")
        model_path = os.path.join(self.models_dir, f"{key}_model.pkl")
        threshold_path = os.path.join(self.models_dir, f"{key}_threshold.json")
        if not os.path.exists(model_path) or not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Missing model or threshold artifact for {model_name}.")
        self.model = joblib.load(model_path)
        with open(threshold_path) as f: self.threshold = float(json.load(f)["optimal_threshold"])
        self.model_name, self.model_key, self._explainer = model_name, key, None

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = AMLExplainer(
                model_path=os.path.join(self.models_dir, f"{self.model_key}_model.pkl"),
                threshold_path=os.path.join(self.models_dir, f"{self.model_key}_threshold.json"),
                pipeline_prefix="aml_pipeline", model_name=self.model_name)
        return self._explainer

    def _prepare_features(self, df_input: pd.DataFrame, history_df: Optional[pd.DataFrame] = None):
        df = df_input.copy()
        raw_markers = {"Timestamp", "Account", "Account.1", "Amount Paid", "Amount Received"}
        if raw_markers.intersection(df.columns):
            features, _ = build_features(df, encoders=self.pipeline.frequency_encoders, history_df=history_df)
        else:
            features = df
        # Pipeline enforces exact training feature order and reports missing fields.
        return self.pipeline.transform(features)

    def predict_single(self, transaction_dict: Dict[str, Any], explain=True, history_df=None):
        X = self._prepare_features(pd.DataFrame([transaction_dict]), history_df=history_df)
        probability = float(self.model.predict_proba(X)[0, 1])
        risk = evaluate_transaction_risk(probability, threshold=self.threshold)
        result = {"model_used": self.model_name,
            "prediction": "Suspicious" if risk["is_suspicious"] else "Legitimate",
            "is_suspicious": risk["is_suspicious"], "probability": risk["probability"],
            "threshold": risk["threshold"], "risk_score": risk["risk_score"],
            "risk_category": risk["risk_category"], "risk_color": risk["risk_color"],
            "investigation_recommended": risk["investigation_recommended"],
            "historical_context_provided": history_df is not None and len(history_df) > 0}
        if explain: result["explanation"] = self.explainer.explain_transaction(X)
        return result

    def predict_batch(self, df_transactions: pd.DataFrame, include_anomaly_scores=False, history_df=None):
        X = self._prepare_features(df_transactions, history_df=history_df)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        scores = calculate_risk_score(probs, calibrated_threshold=self.threshold)
        output = df_transactions.copy()
        output["Predicted_Class"] = np.where(preds == 1, "Suspicious", "Legitimate")
        output["Laundering_Probability"] = np.round(probs, 6)
        output["Risk_Score"] = scores
        output["Risk_Category"] = [assign_risk_category(s)[0] for s in scores]
        output["Alert_Triggered"] = preds.astype(bool)
        if include_anomaly_scores:
            flags = self.get_isolation_forest_scores(df_transactions, history_df=history_df)
            if flags is not None: output["Isolation_Forest_Anomaly"] = flags
        return output

    def get_isolation_forest_scores(self, df_input, history_df=None):
        model_path = os.path.join(self.models_dir, "isolation_forest_model.pkl")
        imputer_path = os.path.join(self.models_dir, "isolation_forest_imputer.pkl")
        if not (os.path.exists(model_path) and os.path.exists(imputer_path)): return None
        if self._if_model is None:
            self._if_model, self._if_imputer = joblib.load(model_path), joblib.load(imputer_path)
        try:
            X = self._prepare_features(df_input, history_df=history_df)
            pred = self._if_model.predict(self._if_imputer.transform(X))
            return np.where(pred == -1, "Anomaly", "Normal")
        except Exception as exc:
            logger.warning("Isolation Forest scoring skipped: %s", exc)
            return None
