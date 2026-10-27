"""H156 CPU-only DEV model replay safety gates (no GPU/browser)."""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h156.replay_model_dev import run
from h156.train_gpu import SOURCE_SHA


class H156DevNativeReplaySafetyTests(unittest.TestCase):
    def test_model_replay_does_not_inject_gold_into_actuator(self):
        source=inspect.getsource(run)
        self.assertIn("parse_v2(raw",source)
        self.assertIn("native_mouse_keyboard(page,actions)",source)
        self.assertIn('case["image_sha256"]',source)
        self.assertNotIn('training_supervision',source)

    def test_missing_model_receipt_rejected_without_browser(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner=Path(tmp)
            (owner/"private-h156"/"miniwob-action-train-dev-v2").mkdir(parents=True)
            with patch("h156.replay_model_dev.admit_upstream",return_value=owner),\
                 patch("h156.replay_model_dev.audit",
                       return_value={"source_sha256":SOURCE_SHA}):
                with self.assertRaises(FileNotFoundError):
                    run(owner,owner,owner)


if __name__=="__main__":
    unittest.main()
