"""H86 deterministic verification of downloaded original Kaggle GPU receipt."""
import json
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
RECEIPT = HERE / "h86" / "evidence" / "kaggle-t4x2-real-gpu-probe.json"


class H86KaggleGpuEvidenceTests(unittest.TestCase):
    def test_proven_actual_t4_and_synthetic_only_scope(self):
        row = json.loads(RECEIPT.read_text(encoding="utf8"))
        self.assertEqual(row["schema"], "hala.h86.private-kaggle-gpu-research-probe.v1")
        self.assertEqual(row["status"], "PASS_GPU_TOY_STEP_ONLY")
        self.assertTrue(row["kaggle_session"])
        self.assertTrue(row["cuda_available"])
        self.assertEqual(row["gpu_count"], 2)
        self.assertEqual(row["gpu_names"], ["Tesla T4", "Tesla T4"])
        self.assertEqual(len(row["gpu_vram_bytes"]), 2)
        self.assertTrue(all(x >= 15_000_000_000 for x in row["gpu_vram_bytes"]))
        self.assertTrue(row["toy_step_updated"])
        self.assertGreater(row["toy_gradient_norm"], 0)
        self.assertTrue(math.isfinite(row["toy_gradient_norm"]))
        self.assertFalse(row["hala_model_training"])
        self.assertFalse(row["dataset_accessed"])
        self.assertFalse(row["paid_compute_requested"])
        self.assertFalse(row["commercial_training_qualified"])

    def test_probe_source_never_loads_hala_model_or_datasets(self):
        source = (HERE / "h86" / "gpu_research_probe.py").read_text(encoding="utf8")
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("kagglehub", source)
        self.assertNotIn("kaggle/input", source)
        self.assertNotIn("Qwen", source)
        self.assertIn("torch.optim.SGD", source)


if __name__ == "__main__":
    unittest.main()
