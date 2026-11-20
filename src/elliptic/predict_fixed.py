"""Unified Elliptic cryptocurrency inference engine.

Loads the preprocessing artifacts and available supervised classifiers, then
supports single-transaction and batch inference while preserving the original
Elliptic ground-truth label when it is present.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.elliptic.risk_scoring import (
    calculate_crypto_risk_score,
    assign_crypto_risk_tier,
)


class EllipticPredictor:
    """Prediction wrapper for the trained Elliptic AML classifiers."""

    MODEL_FILES = {
        "XGBoost": (
            "elliptic_xgboost.pkl",
            "elliptic_xgboost_threshold.json",
        ),
        "Random Forest": (
            "elliptic_random_forest.pkl",
            "elliptic_random_forest_threshold.json",
        ),
        "Logistic Regression": (
            "elliptic_logistic_regression.pkl",
            "elliptic_logistic_regression_threshold.json",
        ),
    }

    def __init__(self, models_dir: str = "models/elliptic") -> None:
        self.models_dir = models_dir

        scaler_path = os.path.join(models_dir, "elliptic_scaler.pkl")
        features_path = os.path.join(models_dir, "elliptic_feature_names.json")

        missing = [p for p in (scaler_path, features_path) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Missing Elliptic preprocessing artifact(s): " + ", ".join(missing)
            )

        self.scaler = joblib.load(scaler_path)
        with open(features_path, "r", encoding="utf-8") as file:
            self.feature_names = json.load(file)

        if not isinstance(self.feature_names, list) or not self.feature_names:
            raise ValueError("elliptic_feature_names.json must contain a non-empty list.")

        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.model_load_warnings: list[str] = []

        for name, (model_file, threshold_file) in self.MODEL_FILES.items():
            model_path = os.path.join(models_dir, model_file)
            threshold_path = os.path.join(models_dir, threshold_file)

            if not os.path.isfile(model_path) or not os.path.isfile(threshold_path):
                self.model_load_warnings.append(
                    f"{name} skipped: model or threshold artifact is missing."
                )
                continue

            try:
                model = joblib.load(model_path)
                with open(threshold_path, "r", encoding="utf-8") as file:
                    threshold_data = json.load(file)

                threshold = float(threshold_data["threshold"])
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("threshold must be between 0 and 1")
                if not hasattr(model, "predict_proba"):
                    raise TypeError("model does not provide predict_proba()")

                self.models[name] = model
                self.thresholds[name] = threshold
            except Exception as exc:
                self.model_load_warnings.append(f"{name} skipped: {exc}")

        if not self.models:
            details = "\n".join(self.model_load_warnings)
            raise FileNotFoundError(
                f"No usable Elliptic classifiers found in '{models_dir}'.\n{details}"
            )

        self.active_model_name = (
            "XGBoost" if "XGBoost" in self.models else next(iter(self.models))
        )

    @property
    def available_models(self) -> list[str]:
        """Names of models whose artifacts loaded successfully."""
        return list(self.models.keys())

    def set_active_model(self, model_name: str) -> None:
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' is unavailable. Available models: "
                f"{self.available_models}"
            )
        self.active_model_name = model_name

    @staticmethod
    def _normalise_ground_truth(row: pd.Series) -> tuple[str, Any]:
        """Return a readable label and its original stored value.

        Elliptic source labels are: '1' = illicit, '2' = licit,
        'unknown' = unlabelled. The engineered target uses 1, 0 and -1.
        """
        raw_value: Any = None
        if "class" in row.index:
            raw_value = row["class"]
        elif "target" in row.index:
            raw_value = row["target"]

        if pd.isna(raw_value) or raw_value is None:
            return "Unknown", raw_value

        value = str(raw_value).strip().lower()
        if value in {"1", "1.0", "illicit"}:
            return "Illicit", raw_value
        if value in {"2", "2.0", "0", "0.0", "licit"}:
            return "Licit", raw_value
        if value in {"-1", "-1.0", "unknown", "nan", "none", ""}:
            return "Unknown", raw_value
        return "Unknown", raw_value

    @staticmethod
    def _positive_class_probability(model: Any, x_scaled: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(x_scaled)
        classes = list(getattr(model, "classes_", []))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError("Unexpected predict_proba output shape from the model.")
        if 1 not in classes:
            raise ValueError(
                f"The loaded model has no positive class '1'. Model classes: {classes}"
            )
        positive_index = classes.index(1)
        return probabilities[:, positive_index]

    def predict_batch(
        self,
        df_input: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Predict a batch of Elliptic transactions.

        Required model features are selected in the exact order used during
        training. Metadata columns such as txId, class and target are retained
        for reporting but are never passed to the model unless they are listed
        in the saved training feature list.
        """
        if not isinstance(df_input, pd.DataFrame):
            raise TypeError("df_input must be a pandas DataFrame.")
        if df_input.empty:
            return pd.DataFrame(
                columns=[
                    "txId", "Model", "Probability", "Illicit_Probability",
                    "Threshold", "Prediction", "Predicted_Target",
                    "Risk_Score", "Risk_Tier", "Actual_Class", "Actual_Label",
                ]
            )

        target_model = model_name or self.active_model_name
        if target_model not in self.models:
            raise ValueError(
                f"Model '{target_model}' is unavailable. Available models: "
                f"{self.available_models}"
            )

        missing_features = [
            feature for feature in self.feature_names if feature not in df_input.columns
        ]
        if missing_features:
            preview = ", ".join(missing_features[:10])
            raise ValueError(
                f"Input is missing {len(missing_features)} required model feature(s): "
                f"{preview}. Ensure the input has passed through the Elliptic "
                "feature-engineering pipeline."
            )

        # Coerce model features to numeric and reject missing/infinite values.
        x_frame = df_input.loc[:, self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        )
        x_values = x_frame.to_numpy(dtype=np.float64)
        invalid_mask = ~np.isfinite(x_values)
        if invalid_mask.any():
            bad_columns = x_frame.columns[invalid_mask.any(axis=0)].tolist()
            raise ValueError(
                "Model input contains missing or non-finite values in: "
                + ", ".join(bad_columns[:10])
                + ". Apply the same imputation/preprocessing used during training."
            )

        model = self.models[target_model]
        threshold = self.thresholds[target_model]
        x_scaled = self.scaler.transform(x_values)
        probabilities = self._positive_class_probability(model, x_scaled)
        predictions = (probabilities >= threshold).astype(int)

        results: list[dict[str, Any]] = []
        for position in range(len(df_input)):
            row = df_input.iloc[position]

            tx_id = row["txId"] if "txId" in row.index else position
            if pd.isna(tx_id):
                tx_id = position
            if isinstance(tx_id, np.generic):
                tx_id = tx_id.item()

            in_degree_value = pd.to_numeric(row.get("in_degree", 0), errors="coerce")
            out_degree_value = pd.to_numeric(row.get("out_degree", 0), errors="coerce")
            in_degree = 0.0 if pd.isna(in_degree_value) else float(in_degree_value)
            out_degree = 0.0 if pd.isna(out_degree_value) else float(out_degree_value)

            total_degree_value = pd.to_numeric(
                row.get("total_degree", np.nan), errors="coerce"
            )
            if pd.isna(total_degree_value):
                total_degree_value = in_degree + out_degree
            total_degree = max(0, int(total_degree_value))

            degree_ratio_value = pd.to_numeric(
                row.get("degree_ratio", np.nan), errors="coerce"
            )
            if pd.isna(degree_ratio_value):
                degree_ratio_value = in_degree / (total_degree + 1.0)
            degree_ratio = float(degree_ratio_value)

            actual_class, original_label = self._normalise_ground_truth(row)
            probability = float(probabilities[position])
            predicted_target = int(predictions[position])
            risk_score = calculate_crypto_risk_score(
                probability, degree_ratio, total_degree
            )

            results.append(
                {
                    "txId": tx_id,
                    "Model": target_model,
                    "Probability": round(probability, 6),
                    "Illicit_Probability": round(probability, 6),
                    "Threshold": round(float(threshold), 6),
                    "Prediction": "Illicit" if predicted_target == 1 else "Licit",
                    "Predicted_Target": predicted_target,
                    "Risk_Score": risk_score,
                    "Risk_Tier": assign_crypto_risk_tier(risk_score),
                    "Actual_Class": actual_class,
                    "Actual_Label": original_label,
                    "Ground_Truth_Known": actual_class != "Unknown",
                    "total_degree": total_degree,
                    "in_degree": int(in_degree) if pd.notna(in_degree) else 0,
                    "out_degree": int(out_degree) if pd.notna(out_degree) else 0,
                }
            )

        return pd.DataFrame(results)

    def predict_single(
        self,
        feature_dict: dict[str, Any],
        model_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Predict one transaction and return its result as a dictionary."""
        if not isinstance(feature_dict, dict):
            raise TypeError("feature_dict must be a dictionary.")
        result = self.predict_batch(pd.DataFrame([feature_dict]), model_name=model_name)
        if result.empty:
            raise ValueError("No transaction was supplied for prediction.")
        return result.iloc[0].to_dict()
