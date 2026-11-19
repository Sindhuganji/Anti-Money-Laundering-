"""
Unit tests for Phase 2 & Phase 3: Elliptic Preprocessing & Graph Feature Engineering.
"""
import unittest
import os
import json
import pandas as pd

class TestEllipticPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.processed_dir = "data/elliptic/processed"
        cls.metadata_path = os.path.join(cls.processed_dir, "elliptic_metadata.json")
        cls.labelled_path = os.path.join(cls.processed_dir, "elliptic_labelled.csv")
        cls.unknown_path = os.path.join(cls.processed_dir, "elliptic_unknown.csv")
        
        with open(cls.metadata_path) as f:
            cls.metadata = json.load(f)

    def test_01_metadata_structure(self):
        """Verify metadata accurately reflects total, labelled, and unknown transactions."""
        self.assertEqual(self.metadata["total_records"], 203769)
        self.assertEqual(self.metadata["labelled_records"], 46564)
        self.assertEqual(self.metadata["unknown_records"], 157205)
        self.assertEqual(self.metadata["illicit_count"], 4545)
        self.assertEqual(self.metadata["licit_count"], 42019)
        self.assertEqual(self.metadata["total_records"], self.metadata["labelled_records"] + self.metadata["unknown_records"])

    def test_02_feature_alignment_and_engineering(self):
        """Check that new engineered features exist in feature_names and have no nulls in labelled data."""
        feature_names = self.metadata["feature_names"]
        expected_engineered = [
            'in_degree', 'out_degree', 'total_degree', 'degree_ratio', 'degree_diff',
            'time_step_norm', 'time_step_sin', 'time_step_cos',
            'local_feature_mean', 'neighbor_feature_mean'
        ]
        for feat in expected_engineered:
            self.assertIn(feat, feature_names, f"Engineered feature {feat} missing from metadata feature list.")
            
        # Check sample of labelled records
        df_sample = pd.read_csv(self.labelled_path, nrows=100)
        for feat in expected_engineered:
            self.assertIn(feat, df_sample.columns)
            self.assertEqual(df_sample[feat].isnull().sum(), 0)

    def test_03_target_leakage_prevention(self):
        """Verify unknown transactions are not marked as licit or illicit in df_unknown."""
        df_unknown_sample = pd.read_csv(self.unknown_path, nrows=500)
        self.assertTrue((df_unknown_sample['target'] == -1).all())
        self.assertTrue((df_unknown_sample['class'] == 'unknown').all())

    def test_04_labelled_data_binary_targets(self):
        """Verify labelled dataset has only binary targets (0=licit, 1=illicit)."""
        df_labelled_sample = pd.read_csv(self.labelled_path, nrows=500)
        unique_targets = set(df_labelled_sample['target'].unique())
        self.assertTrue(unique_targets.issubset({0, 1}))

if __name__ == "__main__":
    unittest.main()
