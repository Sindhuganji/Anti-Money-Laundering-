"""
Unit test suite for Phase 5 & 6 (Inference Engine, Risk Scoring, SHAP, Anomaly Scoring, and App Components).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.predict import AMLPredictor
from src.risk_scoring import calculate_risk_score, assign_risk_category, evaluate_transaction_risk


class TestUnifiedInferenceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = AMLPredictor(model_name="XGBoost")
        cls.df_test_sample = pd.read_csv("data/processed/aml_test.csv", nrows=10)

    def test_xgb_batch_prediction(self):
        self.predictor.set_model("XGBoost")
        res = self.predictor.predict_batch(self.df_test_sample, include_anomaly_scores=True)
        
        self.assertEqual(len(res), len(self.df_test_sample))
        self.assertIn("Predicted_Class", res.columns)
        self.assertIn("Risk_Score", res.columns)
        self.assertIn("Risk_Category", res.columns)
        self.assertIn("Alert_Triggered", res.columns)
        self.assertTrue(np.all((res["Risk_Score"] >= 0.0) & (res["Risk_Score"] <= 100.0)))

    def test_rf_batch_prediction(self):
        self.predictor.set_model("Random Forest")
        res = self.predictor.predict_batch(self.df_test_sample)
        
        self.assertEqual(len(res), len(self.df_test_sample))
        self.assertIn("Predicted_Class", res.columns)
        self.assertTrue(np.all((res["Risk_Score"] >= 0.0) & (res["Risk_Score"] <= 100.0)))

    def test_single_transaction_explanation(self):
        self.predictor.set_model("XGBoost")
        tx = self.df_test_sample.iloc[0].to_dict()
        res = self.predictor.predict_single(tx, explain=True)

        self.assertIn("prediction", res)
        self.assertIn("risk_score", res)
        self.assertIn("explanation", res)
        self.assertIn("top_features", res["explanation"])
        self.assertGreater(len(res["explanation"]["top_features"]), 0)
        self.assertIn("reasoning_bullets", res["explanation"])

    def test_risk_scoring_scale_and_categories(self):
        score_low = calculate_risk_score(0.1, calibrated_threshold=0.8)
        self.assertLess(score_low, 30.0)
        cat_low, _ = assign_risk_category(score_low)
        self.assertEqual(cat_low, "Low Risk")

        score_crit = calculate_risk_score(0.95, calibrated_threshold=0.8)
        self.assertGreaterEqual(score_crit, 80.0)
        cat_crit, _ = assign_risk_category(score_crit)
        self.assertEqual(cat_crit, "Critical Risk")


if __name__ == "__main__":
    unittest.main()
