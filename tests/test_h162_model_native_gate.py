"""H162 third-app real native model OOD anti-oracle one-shot tests."""
import inspect
import unittest

from h162.replay_model_native import execute,preflight,run


class H162ExternalAppNativeModelTests(unittest.TestCase):
    def test_actual_native_drag_only_uses_model_predictions(self):
        source=inspect.getsource(execute)
        self.assertIn('action["points"]',source)
        self.assertIn("mouse_drag(page,start,end)",source)
        self.assertIn('case["image_sha256"]',source)
        self.assertNotIn("owner_reference_action",source)
        self.assertNotIn("bounding_box",source)

    def test_strict_gpu_source_training_provenance(self):
        source=inspect.getsource(preflight)
        for keyword in ("TRAINED_SHA","BASE_SHA","SOURCE_SHA",
                        "gpu_inference_calls","optimizer_steps",
                        "h140_blind_calls"):
            self.assertIn(keyword,source)

    def test_non_blind_paired_real_native_scoring(self):
        source=inspect.getsource(run)
        self.assertIn('consumed.open("x"',source)
        self.assertIn("historically_seen_app_NOT_sealed_BLIND",source)
        self.assertIn("real_original_app_task_success",source)
        self.assertIn("exact_paired_sign_p",source)


if __name__=="__main__":
    unittest.main()
