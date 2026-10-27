"""H154 focused source-only cross-app admission tests (offline; no GPU)."""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h154 import miniwob_source_probe as h154


class H154OriginalTaskAdmission(unittest.TestCase):
    def test_three_distinct_native_action_classes(self):
        self.assertEqual(h154.TASKS, {
            "click-test": "click", "enter-text": "type_submit",
            "choose-list": "select_submit"})
        self.assertEqual(h154.REPEATS, 4)
        self.assertEqual(len(h154.PIN), 40)

    def test_original_minwob_source_sha_and_license_must_be_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                h154.admit_upstream(root)
            (root / "LICENSE").write_text("no license", encoding="utf8")
            with self.assertRaises(ValueError):
                h154.admit_upstream(root)

    def test_oracle_not_a_neural_model_result(self):
        source = inspect.getsource(h154.source_oracle)
        self.assertIn("page.mouse.click", source)
        self.assertIn("page.keyboard.type", source)
        self.assertIn("select_option", source)
        self.assertNotIn("model_predict(", source)
        outcome = inspect.getsource(h154.run)
        self.assertIn('"model_accuracy_measured": False', outcome)
        self.assertIn('"model_calls": 0', outcome)
        self.assertIn('"h140_blind_calls": 0', outcome)

    def test_create_only_owner_private_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "private-h154" / "source-only-miniwob-v2").mkdir(
                parents=True)
            with patch.object(h154, "admit_upstream", return_value=root):
                with self.assertRaises(FileExistsError):
                    h154.run(root, root)

    def test_unknown_action_requires_explicit_new_contract(self):
        with self.assertRaises(ValueError):
            h154.source_oracle(None, "submit_form_not_preregistered")


if __name__ == "__main__":
    unittest.main()
