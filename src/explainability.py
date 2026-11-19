"""
Explainable AI (XAI) engine for AML detection using SHAP (TreeExplainer).
Generates:
1. Global Feature Importance (mean absolute SHAP) & Beeswarm Summary Plots per model.
2. Local Transaction Explanations with SHAP Waterfall plots corresponding to selected model.
3. Natural-language, human-readable compliance explanation generator.
Uniformly handles binary XGBoost (1D output) and Random Forest (2D [class 0, class 1] output).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.preprocessing import AMLPreprocessingPipeline
from src.risk_scoring import evaluate_transaction_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
RESULTS_GRAPHS_DIR = os.path.join("results", "graphs")


class AMLExplainer:
    """
    Explainability engine wrapping XGBoost / Random Forest with SHAP TreeExplainer.
    Dynamically generates model-specific global and local SHAP explanations.
    """

    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, "xgboost_model.pkl"),
        threshold_path: str = os.path.join(MODELS_DIR, "xgboost_threshold.json"),
        pipeline_prefix: str = "aml_pipeline",
        model_name: Optional[str] = None
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "Random Forest" if "random_forest" in model_path.lower() else "XGBoost"

        with open(threshold_path, "r") as f:
            threshold_info = json.load(f)
            self.threshold = float(threshold_info.get("optimal_threshold", 0.5))

        logger.info(f"Loaded threshold for {self.model_name}: {self.threshold:.4f}")
        self.pipeline = AMLPreprocessingPipeline.load(model_dir=MODELS_DIR, prefix=pipeline_prefix)
        self.feature_names = self.pipeline.feature_names

        logger.info(f"Initializing SHAP TreeExplainer for {self.model_name}...")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("AMLExplainer initialized successfully.")

    def _extract_positive_shap(self, shap_obj):
        """
        Extract SHAP values corresponding to the positive 'Laundering' class (class 1).
        If binary RF, shape is (N, features, 2) -> slice index 1.
        If XGB, shape is (N, features) -> use directly.
        """
        if len(shap_obj.values.shape) == 3 and shap_obj.values.shape[-1] == 2:
            return shap_obj[:, :, 1]
        return shap_obj

    def explain_global(
        self,
        X_sample: pd.DataFrame,
        output_dir: str = RESULTS_GRAPHS_DIR,
        max_display: int = 15,
        model_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        Generate and save global SHAP summary bar (mean absolute SHAP) and beeswarm plots.
        Names plots per model to prevent overwriting or displaying stale charts.
        
        Returns:
            Tuple[DataFrame of feature importances, bar_plot_path, beeswarm_plot_path]
        """
        os.makedirs(output_dir, exist_ok=True)
        active_name = model_name or self.model_name
        slug = active_name.lower().replace(" ", "_")

        shap_values = self.explainer(X_sample)
        shap_pos = self._extract_positive_shap(shap_values)

        # 1. Global Bar Plot (displays mean absolute SHAP values: mean(|SHAP value|))
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_pos, max_display=max_display, show=False)
        plt.title(f"Global SHAP Feature Importance — {active_name} (Laundering Class)", fontsize=13)
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"shap_global_bar_{slug}.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()

        # 2. Global Beeswarm Summary Plot (shows direction of feature impact)
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_pos, max_display=max_display, show=False)
        plt.title(f"SHAP Beeswarm Summary Plot — {active_name}", fontsize=13)
        plt.tight_layout()
        beeswarm_path = os.path.join(output_dir, f"shap_global_beeswarm_{slug}.png")
        plt.savefig(beeswarm_path, dpi=300)
        plt.close()

        # Tabular mean absolute SHAP ranking
        mean_abs_shap = np.abs(shap_pos.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        logger.info(f"Saved global SHAP plots for {active_name} to {output_dir}")
        return importance_df, bar_path, beeswarm_path

    def explain_transaction(
        self,
        transaction_features: pd.DataFrame,
        save_plot_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Produce local SHAP waterfall explanation and natural language risk reasoning.
        """
        if isinstance(transaction_features, pd.Series):
            transaction_features = transaction_features.to_frame().T

        active_name = model_name or self.model_name
        X_tx = transaction_features[self.feature_names]
        prob = float(self.model.predict_proba(X_tx)[0, 1])
        risk_meta = evaluate_transaction_risk(prob, threshold=self.threshold)

        shap_values = self.explainer(X_tx)
        shap_pos = self._extract_positive_shap(shap_values)
        explanation = shap_pos[0]

        feature_contributions = []
        for name, shap_val, feat_val in zip(self.feature_names, explanation.values, X_tx.iloc[0]):
            val_scalar = float(shap_val) if np.ndim(shap_val) == 0 else float(shap_val[0])
            feature_contributions.append({
                "feature": name,
                "value": float(feat_val),
                "shap_value": val_scalar,
                "direction": "Risk-Increasing" if val_scalar > 0 else "Risk-Decreasing",
                "abs_impact": abs(val_scalar)
            })

        feature_contributions = sorted(feature_contributions, key=lambda x: x["abs_impact"], reverse=True)
        reasoning_bullets = self._generate_nl_reasoning(feature_contributions, risk_meta["is_suspicious"], active_name)

        if save_plot_path:
            os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.title(f"SHAP Local Waterfall — {active_name} (Risk Score: {risk_meta['risk_score']}/100)", fontsize=11)
            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=300)
            plt.close()

        base_val = explanation.base_values
        base_scalar = float(base_val) if np.ndim(base_val) == 0 else float(base_val[0])

        return {
            "model_name": active_name,
            "prediction_meta": risk_meta,
            "top_features": feature_contributions[:10],
            "reasoning_bullets": reasoning_bullets,
            "base_value": base_scalar
        }

    def _generate_nl_reasoning(
        self,
        contributions: List[Dict[str, Any]],
        is_suspicious: bool,
        model_name: str
    ) -> List[str]:
        """
        Build non-hardcoded, data-driven audit explanations based on dominant SHAP drivers.
        """
        bullets = []
        top_positive = [c for c in contributions if c["shap_value"] > 0][:4]
        top_negative = [c for c in contributions if c["shap_value"] < 0][:2]

        feature_descriptions = {
            "Amount Received": "Transaction amount received",
            "Amount Paid": "Transaction amount paid",
            "Amount_Difference": "Discrepancy between amount paid and received",
            "Absolute_Amount_Difference": "Net magnitude of payment difference",
            "Amount_Ratio": "Ratio of amount paid to received",
            "Log_Amount_Received": "Log-scaled amount received volume",
            "Log_Amount_Paid": "Log-scaled amount paid volume",
            "Transaction_Hour": "Time-of-day execution hour",
            "Transaction_Day": "Day of the month execution timing",
            "Transaction_Month": "Monthly cycle timing",
            "Transaction_Weekday": "Day of week pattern",
            "Is_Night_Transaction": "Off-hours / nocturnal transaction timing",
            "Is_Weekend": "Weekend settlement activity",
            "Same_Bank": "Intra-bank transfer pattern",
            "Same_Currency": "Currency exchange relationship",
            "Source_Transaction_Frequency": "Source account historical velocity",
            "Destination_Transaction_Frequency": "Destination account historical velocity",
            "Previous_Source_Transactions": "Prior source transaction count",
            "Previous_Source_Amount": "Cumulative historical outflow from source account",
            "Previous_Destination_Transactions": "Prior destination transaction count",
            "Previous_Destination_Amount": "Cumulative historical inflow to destination account",
            "Payment Format_Frequency": "Payment channel rarity/frequency",
            "Receiving Currency_Frequency": "Receiving currency usage profile",
            "Payment Currency_Frequency": "Disbursing currency usage profile"
        }

        if is_suspicious:
            bullets.append(f"Alert triggered by {model_name}: Evaluated transaction profile as elevated money laundering risk.")
            for c in top_positive:
                desc = feature_descriptions.get(c["feature"], c["feature"])
                bullets.append(f"• Elevated Risk Factor: {desc} contributed strongly (+{c['shap_value']:.3f} log-odds) toward the suspicious classification.")
            if top_negative:
                desc_neg = feature_descriptions.get(top_negative[0]["feature"], top_negative[0]["feature"])
                bullets.append(f"• Mitigating Factor: {desc_neg} slightly reduced alert severity ({top_negative[0]['shap_value']:.3f} log-odds).")
        else:
            bullets.append(f"Normal transaction profile according to {model_name}: Features aligned with regular commercial activity.")
            for c in top_negative:
                desc = feature_descriptions.get(c["feature"], c["feature"])
                bullets.append(f"• Compliance indicator: {desc} aligns with regular, low-risk account patterns ({c['shap_value']:.3f} log-odds).")

        return bullets
