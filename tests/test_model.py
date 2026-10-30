"""
tests/test_model.py - Unit tests for model construction and evaluation artifacts
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_config import (
    CLASS_LABELS_PATH,
    METRICS_JSON_PATH,
    DISEASE_INFO_PATH,
    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


class TestModelArtifacts(unittest.TestCase):

    def test_class_labels_and_disease_info(self):
        """Validates class labels registry and comprehensive disease metadata."""
        self.assertTrue(CLASS_LABELS_PATH.exists(), "class_labels.json must exist")
        with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        self.assertEqual(len(labels_data["classes"]), NUM_CLASSES, f"Expected {NUM_CLASSES} classes")

        self.assertTrue(DISEASE_INFO_PATH.exists(), "disease_info.json must exist")
        with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            info_data = json.load(f)

        # Every class must have symptoms, causative agent, organic and chemical controls
        for cls_name in labels_data["classes"]:
            self.assertIn(cls_name, info_data, f"Missing disease info for {cls_name}")
            item = info_data[cls_name]
            self.assertTrue("symptoms" in item and len(item["symptoms"]) > 10)
            self.assertIn("organic_control", item)
            self.assertIn("chemical_control", item)
            self.assertIn("risk_level", item)

    def test_evaluation_metrics_report(self):
        """Validates official metrics format, accuracy threshold, and latency bounds."""
        self.assertTrue(METRICS_JSON_PATH.exists(), "evaluation_metrics.json must exist")
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(metrics["overall_accuracy"], 0.90, "Accuracy must exceed 90%")
        self.assertIn("macro_metrics", metrics)
        self.assertIn("latency_benchmark", metrics)
        self.assertLess(metrics["latency_benchmark"]["mean_latency_ms"], 100.0, "CPU latency must be under 100ms")


if __name__ == "__main__":
    unittest.main()
