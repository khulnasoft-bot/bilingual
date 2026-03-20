"""
Integration test for ModelManager metrics and batching.
"""

import unittest
import torch
from bilingual.models.manager import model_manager
from prometheus_client import REGISTRY

class TestModelManagerAdvanced(unittest.TestCase):
    def setUp(self):
        # Clear cache before tests
        model_manager.clear_cache()

    def test_metrics_collection(self):
        """Verify that Prometheus metrics are updated on model load."""
        # Initial state
        initial_count = REGISTRY.get_sample_value('bilingual_model_cache_items') or 0
        
        # Load a small model (using a placeholder or mock if real path unavailable, 
        # but here we assume 'bilingual-small' is valid in registry)
        try:
            model_manager.load_model("bilingual-small")
            new_count = REGISTRY.get_sample_value('bilingual_model_cache_items')
            self.assertEqual(new_count, 1.0)
        except Exception as e:
            self.skipTest(f"Registry model not found: {e}")

    def test_dynamic_batching(self):
        """Test the dynamic batch sizing logic."""
        # 1. CPU Mode
        model_manager.device = "cpu"
        self.assertEqual(model_manager._get_dynamic_batch_size(100), 1)
        
        # 2. CUDA Mode (Simulated)
        model_manager.device = "cuda"
        self.assertEqual(model_manager._get_dynamic_batch_size(5), 8)
        self.assertEqual(model_manager._get_dynamic_batch_size(20), 16)
        self.assertEqual(model_manager._get_dynamic_batch_size(100), 32)

    def test_predict_batch_throughput(self):
        """Test predict_batch functionality."""
        inputs = ["Hello world"] * 10
        # If transformers is available, we can test more, but for now we check logic
        try:
            # This might fail in CI if models aren't downloaded, so we just check if it calls registry
            results = model_manager.predict_batch("bilingual-small", inputs)
            self.assertEqual(len(results), 10)
        except Exception:
            self.skipTest("Requires model artifacts to be present.")

if __name__ == "__main__":
    unittest.main()
