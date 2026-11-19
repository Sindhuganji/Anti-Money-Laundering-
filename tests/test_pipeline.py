"""
Unit tests for data_loader, feature_engineering, and preprocessing modules.
Implemented using standard library unittest to run seamlessly without extra dependencies.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.data_loader import validate_raw_schema
from src.feature_engineering import (
    build_features,
    fit_frequency_encoders,
    apply_frequency_encoders,
    extract_temporal_features,
    extract_amount_features,
    extract_relational_features
)
from src.preprocessing import split_data, AMLPreprocessingPipeline


def create_dummy_raw_data(n=200):
    np.random.seed(42)
    timestamps = pd.date_range("2022-09-01", periods=n, freq="h").strftime("%Y/%m/%d %H:%M")
    accounts = [f"acc_{i%15}" for i in range(n)]
    dest_accounts = [f"dest_{i%20}" for i in range(n)]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "From Bank": np.random.randint(10, 50, size=n),
        "Account": accounts,
        "To Bank": np.random.randint(10, 50, size=n),
        "Account.1": dest_accounts,
        "Amount Received": np.random.uniform(10, 10000, size=n),
        "Receiving Currency": np.random.choice(["US Dollar", "Euro", "Rupee"], size=n),
        "Amount Paid": np.random.uniform(10, 10000, size=n),
        "Payment Currency": np.random.choice(["US Dollar", "Euro", "Rupee"], size=n),
        "Payment Format": np.random.choice(["Cheque", "Credit Card", "Wire"], size=n),
        "Is Laundering": np.random.choice([0, 1], p=[0.95, 0.05], size=n)
    })
    return df


class TestAMLDataPipeline(unittest.TestCase):

    def test_schema_validation(self):
        df = create_dummy_raw_data(20)
        self.assertTrue(validate_raw_schema(df))

        df_invalid = df.drop(columns=["Timestamp"])
        with self.assertRaises(ValueError):
            validate_raw_schema(df_invalid)

    def test_temporal_and_amount_features(self):
        df = create_dummy_raw_data(50)
        df_temp = extract_temporal_features(df)
        self.assertIn("Transaction_Hour", df_temp.columns)
        self.assertIn("Is_Weekend", df_temp.columns)
        self.assertIn("Is_Night_Transaction", df_temp.columns)

        df_amt = extract_amount_features(df)
        self.assertIn("Amount_Difference", df_amt.columns)
        self.assertIn("Amount_Ratio", df_amt.columns)
        self.assertIn("Log_Amount_Received", df_amt.columns)

    def test_frequency_encoding_leakage_safety(self):
        train_data = pd.DataFrame({
            "Account": ["A", "A", "B"],
            "Receiving Currency": ["USD", "USD", "EUR"]
        })
        test_data = pd.DataFrame({
            "Account": ["A", "C"],  # "C" is an unseen account
            "Receiving Currency": ["USD", "JPY"]  # "JPY" is an unseen currency
        })

        encoders = fit_frequency_encoders(
            train_data,
            categorical_cols=["Receiving Currency"],
            identifier_cols=["Account"]
        )

        self.assertNotIn("C", encoders["Account"])
        self.assertNotIn("JPY", encoders["Receiving Currency"])

        transformed_test = apply_frequency_encoders(
            test_data,
            encoders,
            default_cat_val=0.0,
            default_id_val=1.0
        )

        # Unseen values correctly took default fallback without errors
        self.assertEqual(transformed_test.loc[1, "Source_Transaction_Frequency"], 1.0)
        self.assertEqual(transformed_test.loc[1, "Receiving Currency_Frequency"], 0.0)

    def test_stratified_split(self):
        df = create_dummy_raw_data(100)
        train_df, val_df, test_df = split_data(
            df,
            target_col="Is Laundering",
            test_size=0.20,
            val_size=0.20,
            random_state=42
        )

        self.assertEqual(len(test_df), 20)
        self.assertEqual(len(train_df), 64)
        self.assertEqual(len(val_df), 16)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 100)

    def test_pipeline_fit_transform_save_load(self):
        df = create_dummy_raw_data(100)
        fe_df, encoders = build_features(df)

        train_df, val_df, test_df = split_data(fe_df)

        pipe = AMLPreprocessingPipeline(scaler_type="robust", impute_strategy="median")
        pipe.fit(train_df, frequency_encoders=encoders)

        train_scaled = pipe.transform(train_df)
        val_scaled = pipe.transform(val_df)
        test_scaled = pipe.transform(test_df)

        self.assertEqual(train_scaled.shape[1], len(pipe.feature_names))
        self.assertEqual(val_scaled.shape[1], len(pipe.feature_names))
        self.assertEqual(test_scaled.shape[1], len(pipe.feature_names))
        self.assertFalse(train_scaled.isnull().any().any())

        # Test persistence
        temp_dir = tempfile.mkdtemp()
        try:
            pipe.save(model_dir=temp_dir, prefix="test_pipe")
            loaded_pipe = AMLPreprocessingPipeline.load(model_dir=temp_dir, prefix="test_pipe")
            re_transformed = loaded_pipe.transform(test_df)
            pd.testing.assert_frame_equal(test_scaled, re_transformed)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
