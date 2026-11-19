"""
Unified Elliptic Cryptocurrency Inference Engine.
Provides inference, risk scoring, and individual / batch prediction for the Elliptic Bitcoin dataset.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from src.elliptic.risk_scoring import calculate_crypto_risk_score, assign_crypto_risk_tier, format_crypto_risk_assessment

class EllipticPredictor:
    def __init__(self, models_dir="models/elliptic"):
        self.models_dir = models_dir
        
        # Load scaler and feature names
        scaler_path = os.path.join(models_dir, "elliptic_scaler.pkl")
        features_path = os.path.join(models_dir, "elliptic_feature_names.json")
        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing Elliptic preprocessing artifacts in {models_dir}")
            
        self.scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            self.feature_names = json.load(f)
            
        # Available models
        self.models = {}
        self.thresholds = {}
        
        model_configs = {
            "XGBoost": ("elliptic_xgboost.pkl", "elliptic_xgboost_threshold.json"),
            "Random Forest": ("elliptic_random_forest.pkl", "elliptic_random_forest_threshold.json"),
            "Logistic Regression": ("elliptic_logistic_regression.pkl", "elliptic_logistic_regression_threshold.json")
        }
        
        for name, (m_file, th_file) in model_configs.items():
            m_path = os.path.join(models_dir, m_file)
            th_path = os.path.join(models_dir, th_file)
            if os.path.exists(m_path) and os.path.exists(th_path):
                self.models[name] = joblib.load(m_path)
                with open(th_path) as f:
                    self.thresholds[name] = json.load(f)["threshold"]
                    
        self.active_model_name = "XGBoost" if "XGBoost" in self.models else list(self.models.keys())[0]

    def set_active_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        self.active_model_name = model_name

    def predict_batch(self, df_input: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """
        Runs batch prediction on dataframe containing Elliptic features.
        Preserves txId and ground truth if present.
        """
        target_model = model_name if model_name else self.active_model_name
        model = self.models[target_model]
        threshold = self.thresholds[target_model]
        
        # Check and align features
        missing_features = [f for f in self.feature_names if f not in df_input.columns]
        if missing_features:
            raise ValueError(f"Input DataFrame missing {len(missing_features)} required features: {missing_features[:5]}...")
            
        X_raw = df_input[self.feature_names].values
        X_scaled = self.scaler.transform(X_raw)
        
        probabilities = model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        results = []
        for i, row in enumerate(df_input.itertuples(index=False)):
            tx_id = getattr(row, 'txId', i)
            total_deg = getattr(row, 'total_degree', 0)
            deg_ratio = getattr(row, 'degree_ratio', 0.5)
            actual = getattr(row, 'class', 'unknown')
            
            prob = float(probabilities[i])
            score = calculate_crypto_risk_score(prob, deg_ratio, total_deg)
            tier = assign_crypto_risk_tier(score)
            
            results.append({
                "txId": tx_id,
                "Model": target_model,
                "Probability": round(prob, 4),
                "Threshold": round(threshold, 4),
                "Prediction": "Illicit" if predictions[i] == 1 else "Licit",
                "Predicted_Target": int(predictions[i]),
                "Risk_Score": score,
                "Risk_Tier": tier,
                "Actual_Class": actual
            })
            
        return pd.DataFrame(results)

    def predict_single(self, feature_dict: dict, model_name: str = None) -> dict:
        """Predicts risk for a single transaction dictionary."""
        df_single = pd.DataFrame([feature_dict])
        batch_res = self.predict_batch(df_single, model_name=model_name)
        return batch_res.iloc[0].to_dict()
