"""
Unit tests for Phase 1: Elliptic Bitcoin Dataset Acquisition and Integrity.
"""
import unittest
import os
import pandas as pd
from src.elliptic.data_acquisition import validate_elliptic_dataset

class TestEllipticDataAcquisition(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = "data/elliptic/raw"
        cls.classes_path = os.path.join(cls.raw_dir, "elliptic_txs_classes.csv")
        cls.edgelist_path = os.path.join(cls.raw_dir, "elliptic_txs_edgelist.csv")
        cls.features_path = os.path.join(cls.raw_dir, "elliptic_txs_features.csv")
        cls.report_path = "ELLIPTIC_DATASET_REPORT.md"

    def test_01_files_exist_and_non_empty(self):
        """Verify all three required CSV files exist and have non-zero size."""
        for path in [self.classes_path, self.edgelist_path, self.features_path]:
            self.assertTrue(os.path.exists(path), f"File missing: {path}")
            self.assertGreater(os.path.getsize(path), 1000, f"File unexpectedly small: {path}")

    def test_02_classes_schema_and_values(self):
        """Verify classes file schema and discrete class categories."""
        df_classes = pd.read_csv(self.classes_path, nrows=1000)
        self.assertListEqual(list(df_classes.columns), ["txId", "class"])
        unique_classes = set(pd.read_csv(self.classes_path)["class"].unique())
        self.assertEqual(unique_classes, {"1", "2", "unknown"})

    def test_03_edgelist_schema(self):
        """Verify edgelist columns txId1 and txId2."""
        df_edges = pd.read_csv(self.edgelist_path, nrows=1000)
        self.assertListEqual(list(df_edges.columns), ["txId1", "txId2"])

    def test_04_features_columns_count(self):
        """Verify features CSV has exactly 167 columns."""
        df_features = pd.read_csv(self.features_path, header=None, nrows=10)
        self.assertEqual(df_features.shape[1], 167)

    def test_05_validation_report_generation(self):
        """Verify that validation function runs and generates ELLIPTIC_DATASET_REPORT.md."""
        res = validate_elliptic_dataset(self.raw_dir, self.report_path)
        self.assertTrue(os.path.exists(self.report_path))
        self.assertEqual(res["classes_rows"], 203769)
        self.assertEqual(res["features_rows"], 203769)
        self.assertEqual(len(res["time_steps"]), 49)

if __name__ == "__main__":
    unittest.main()
