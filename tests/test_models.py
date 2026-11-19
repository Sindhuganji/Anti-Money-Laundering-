"""
Unit test suite for Phase 3:
Tests:
1. Loading trained Random Forest model and threshold.
2. Loading trained XGBoost model and threshold.
3. Feature alignment and prediction consistency on unseen test sample.
4. Probability bounds and threshold binary decisions.
"""

import os
import json
import joblib
import unittest
import numpy as np
import pandas as pd

from src.preprocessing import AMLPreprocessingPipeline


class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models"
        cls.rf_model_path = os.path.join(cls.models_dir, "random_forest_model.pkl")
        cls.rf_thresh_path = os.path.join(cls.models_dir, "random_forest_threshold.json")
        cls.xgb_model_path = os.path.join(cls.models_dir, "xgboost_model.pkl")
        cls.xgb_thresh_path = os.path.join(cls.models_dir, "xgboost_threshold.json")
        cls.feature_names_path = os.path.join(cls.models_dir, "feature_names.json")

        cls.pipeline = AMLPreprocessingPipeline.load(model_dir=cls.models_dir, prefix="aml_pipeline")
        cls.rf_model = joblib.load(cls.rf_model_path)
        cls.xgb_model = joblib.load(cls.xgb_model_path)

        with open(cls.rf_thresh_path) as f:
            cls.rf_thresh = json.load(f)["optimal_threshold"]

        with open(cls.xgb_thresh_path) as f:
            cls.xgb_thresh = json.load(f)["optimal_threshold"]

        with open(cls.feature_names_path) as f:
            cls.feature_names = json.load(f)

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists(self.rf_model_path))
        self.assertTrue(os.path.exists(self.xgb_model_path))
        self.assertTrue(os.path.exists(self.rf_thresh_path))
        self.assertTrue(os.path.exists(self.xgb_thresh_path))

    def test_feature_count_and_ordering(self):
        self.assertEqual(len(self.feature_names), 24)
        self.assertEqual(self.pipeline.feature_names, self.feature_names)

    def test_unseen_test_sample_inference(self):
        # Read small slice of test dataset
        df_test = pd.read_csv("data/processed/aml_test.csv", nrows=50)
        X_test = self.pipeline.transform(df_test)

        self.assertEqual(X_test.shape, (50, 24))

        # RF inference
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]
        self.assertEqual(len(rf_prob), 50)
        self.assertTrue(np.all((rf_prob >= 0.0) & (rf_prob <= 1.0)))

        rf_pred = (rf_prob >= self.rf_thresh).astype(int)
        self.assertTrue(set(np.unique(rf_pred)).issubset({0, 1}))

        # XGB inference
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        self.assertEqual(len(xgb_prob), 50)
        self.assertTrue(np.all((xgb_prob >= 0.0) & (xgb_prob <= 1.0)))

        xgb_pred = (xgb_prob >= self.xgb_thresh).astype(int)
        self.assertTrue(set(np.unique(xgb_pred)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
