"""
tests/test_data_pipeline.py - Unit tests for data preprocessing and cleaning
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_config import (
    RAW_DATASET_CSV,
    CLEANED_DATASET_CSV,
    DATA_DICTIONARY_CSV,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)
from data_preprocessing import load_and_preprocess_image


class TestDataPipeline(unittest.TestCase):

    def test_raw_dataset_exists_and_valid(self):
        """Validates that raw dataset.csv exists and contains exactly 500 records."""
        self.assertTrue(RAW_DATASET_CSV.exists(), "dataset.csv must exist in project directory")
        df = pd.read_csv(RAW_DATASET_CSV)
        self.assertEqual(len(df), 500, f"Expected 500 rows, found {len(df)}")
        expected_cols = {"image_id", "crop_type", "disease_label", "confidence_score", "severity", "leaf_area_cm2", "batch_id"}
        self.assertTrue(expected_cols.issubset(set(df.columns)), f"Missing expected columns: {expected_cols - set(df.columns)}")

    def test_data_dictionary_completeness(self):
        """Validates that data_dictionary.csv has descriptions for all 7 raw columns."""
        self.assertTrue(DATA_DICTIONARY_CSV.exists(), "data_dictionary.csv must exist")
        dict_df = pd.read_csv(DATA_DICTIONARY_CSV)
        self.assertEqual(len(dict_df), 7, f"Expected 7 dictionary rows, found {len(dict_df)}")
        self.assertFalse(dict_df["description_fill_in"].str.contains("Learner to fill in").any(), "Descriptions must be filled")

    def test_cleaned_dataset_logical_consistency(self):
        """Verifies that all Healthy leaves have severity sanitized to 'None'."""
        self.assertTrue(CLEANED_DATASET_CSV.exists(), "cleaned_dataset.csv must exist")
        df = pd.read_csv(CLEANED_DATASET_CSV, keep_default_na=False)
        
        # All healthy leaves must have severity 'None'
        healthy_df = df[df["disease_label"].str.lower() == "healthy"]
        self.assertTrue((healthy_df["severity"] == "None").all(), "All healthy leaves must have severity == 'None'")

        # Risk level for healthy leaves must be 'None'
        self.assertTrue((healthy_df["risk_level"] == "None").all(), "All healthy leaves must have risk_level == 'None'")

        # Confidence score bounds [0.0, 1.0]
        self.assertTrue((df["confidence_score"].astype(float) >= 0.0).all() and (df["confidence_score"].astype(float) <= 1.0).all())

        # Leaf area > 0
        self.assertTrue((df["leaf_area_cm2"].astype(float) > 0).all())

    def test_image_preprocessing_pipeline(self):
        """Verifies tensor shape and normalization bounds [-1.0, 1.0]."""
        test_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        img_path = Path("temp_test_leaf.png")
        test_img.save(img_path)

        try:
            tensor = load_and_preprocess_image(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
            self.assertEqual(tensor.shape, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), f"Unexpected tensor shape {tensor.shape}")
            self.assertTrue(tensor.min() >= -1.0 and tensor.max() <= 1.0, "Tensor values must be normalized in [-1.0, 1.0]")
        finally:
            if img_path.exists():
                img_path.unlink()


if __name__ == "__main__":
    unittest.main()
