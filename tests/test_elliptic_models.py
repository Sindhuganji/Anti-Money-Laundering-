"""
Unit tests for Phase 4, Phase 5, Phase 6, Phase 7, and Phase 8 of the Elliptic pipeline.
"""
import unittest
import os
import json
import numpy as np
import pandas as pd
from src.elliptic.predict import EllipticPredictor
from src.elliptic.risk_scoring import calculate_crypto_risk_score, assign_crypto_risk_tier
from src.elliptic.graph_analysis import generate_graph_analytics

class TestEllipticModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models/elliptic"
        cls.predictor = EllipticPredictor(models_dir=cls.models_dir)
        cls.labelled_path = "data/elliptic/processed/elliptic_labelled.csv"
        cls.sample_df = pd.read_csv(cls.labelled_path, nrows=50)

    def test_01_models_loaded(self):
        """Verify XGBoost, Random Forest, and Logistic Regression loaded with valid thresholds."""
        for name in ["XGBoost", "Random Forest", "Logistic Regression"]:
            self.assertIn(name, self.predictor.models)
            self.assertIn(name, self.predictor.thresholds)
            th = self.predictor.thresholds[name]
            self.assertTrue(0.0 < th < 1.0, f"Invalid threshold {th} for {name}")

    def test_02_batch_prediction(self):
        """Verify batch prediction returns valid predictions and probabilities."""
        res = self.predictor.predict_batch(self.sample_df, model_name="XGBoost")
        self.assertEqual(len(res), 50)
        self.assertIn("Probability", res.columns)
        self.assertIn("Risk_Score", res.columns)
        self.assertIn("Risk_Tier", res.columns)
        self.assertTrue((res["Probability"] >= 0.0).all() and (res["Probability"] <= 1.0).all())

    def test_03_model_switching(self):
        """Test switching active model between XGBoost and Random Forest."""
        self.predictor.set_active_model("Random Forest")
        res_rf = self.predictor.predict_batch(self.sample_df.head(5))
        self.assertEqual(res_rf.iloc[0]["Model"], "Random Forest")
        
        self.predictor.set_active_model("XGBoost")
        res_xgb = self.predictor.predict_batch(self.sample_df.head(5))
        self.assertEqual(res_xgb.iloc[0]["Model"], "XGBoost")

    def test_04_risk_scoring_tiers(self):
        """Verify risk scoring function produces correct ranges and tiers."""
        self.assertEqual(assign_crypto_risk_tier(95.0), "Critical")
        self.assertEqual(assign_crypto_risk_tier(75.0), "High")
        self.assertEqual(assign_crypto_risk_tier(45.0), "Medium")
        self.assertEqual(assign_crypto_risk_tier(15.0), "Low")

    def test_05_graph_analytics_output(self):
        """Verify graph visualization generation."""
        html_out = "results/elliptic/graphs/transaction_network.html"
        self.assertTrue(os.path.exists(html_out))
        self.assertGreater(os.path.getsize(html_out), 5000)

if __name__ == "__main__":
    unittest.main()
