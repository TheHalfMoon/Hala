"""H161 independent live-original browser DEV scoring contract tests."""
import inspect
import unittest
from h161.replay_model_dev import preflight,run,todo_trial,board_trial


class H161ModelDevNativeGates(unittest.TestCase):
    def test_never_pass_owner_oracle_into_model_native_action(self):
        for fn in (todo_trial,board_trial):
            source=inspect.getsource(fn)
            self.assertIn("case",source)
            self.assertIn('case["image_sha256"]',source)
            self.assertNotIn("reference_point",source)
            self.assertNotIn("training_supervision",source)
            self.assertNotIn("bounding_box",source)

    def test_source_provenance_before_one_shot(self):
        text=inspect.getsource(preflight)
        self.assertIn("H161_72_OF_72_CROSS_CONTEXT_EXACT_NATIVE_SCREENSHOT_MATCH",text)
        self.assertIn("dev_labels_seen_by_GPU",text)
        self.assertIn("source_train_zip_sha256",text)
        self.assertIn("starting_H157_adapter_sha256",text)

    def test_consume_exact_frozen_model_after_source_preflight(self):
        source=inspect.getsource(run)
        self.assertIn('consumed.open("x"',source)
        self.assertIn("consumed.exists() or output.exists()",source)
        self.assertIn("model_dev_cases",source)
        self.assertIn("same_apps_as_training_not_OOD_BLIND",source)


if __name__=="__main__":
    unittest.main()
