"""
tests/test_gradcam.py - Unit tests for Grad-CAM explainability module
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_config import IMAGE_HEIGHT, IMAGE_WIDTH
from gradcam import overlay_heatmap, generate_synthetic_saliency


class TestGradCAM(unittest.TestCase):

    def test_synthetic_saliency_shape_and_bounds(self):
        """Tests the fallback saliency algorithm for robust 2D heatmap generation."""
        dummy_tensor = np.random.uniform(-1.0, 1.0, size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
        heatmap = generate_synthetic_saliency(dummy_tensor)

        self.assertEqual(heatmap.shape, (IMAGE_HEIGHT, IMAGE_WIDTH), f"Expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}, got {heatmap.shape}")
        self.assertTrue(heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap values must be normalized in [0, 1]")

    def test_overlay_heatmap_shapes_and_types(self):
        """Tests OpenCV heatmap blending and overlay color map output."""
        orig_img = np.random.randint(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        heatmap = np.random.uniform(0.0, 1.0, size=(IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

        # Using matplotlib/numpy colormap fallback if cv2 not loaded
        colored_hm, superimposed = overlay_heatmap(orig_img, heatmap, alpha=0.55, colormap="JET")

        self.assertEqual(colored_hm.shape, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(superimposed.shape, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(colored_hm.dtype, np.uint8)
        self.assertEqual(superimposed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
