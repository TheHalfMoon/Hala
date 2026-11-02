"""H346: native GUI trainer denies a self-certified, unadmitted manifest before GPU."""
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from training.preflight import require_admitted_gui_training_manifest
from training.scripts.train_adapter import load_json, train

DENIAL = "H346_OFFICIAL_GUI_TRAIN_MANIFEST_NOT_ADMITTED"


class OfficialAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'{"rights_cleared":true,"split":"train","official_admitted":true}'
        self.digest = sha256(self.raw).hexdigest()

    def test_self_declared_rights_and_opt_in_cannot_admit(self):
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset()):
            with self.assertRaisesRegex(ValueError, DENIAL):
                require_admitted_gui_training_manifest(self.raw)

    def test_exact_byte_binding_not_claimed_status_or_alias(self):
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset({self.digest})):
            self.assertEqual(require_admitted_gui_training_manifest(self.raw), self.digest)
            with self.assertRaisesRegex(ValueError, DENIAL):
                require_admitted_gui_training_manifest(self.raw + b" ")
            for invalid in (None, True, "", b"", b"x" * (4 * 1024 * 1024 + 1)):
                with self.subTest(value=type(invalid).__name__), self.assertRaisesRegex(ValueError, DENIAL):
                    require_admitted_gui_training_manifest(invalid)

    def test_real_train_entrypoint_refuses_before_baseline_gpu_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_bytes(self.raw)
            config = root / "config.json"
            config.write_text(json.dumps({
                "model_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": "a" * 40,
                "precision": "fp16", "max_steps": 1, "batch_size": 1,
                "gradient_accumulation": 1, "learning_rate": 0.0001,
                "lora_rank": 4, "platform": "kaggle", "download_remote_weights": True
            }), encoding="utf-8")
            output = root / "no-output"
            with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset()):
                with patch("h87.research_runtime.require_training_provider",
                           return_value="kaggle_noncommercial_research"):
                    with self.assertRaisesRegex(ValueError, DENIAL):
                        train(config, root / "nonexistent.jsonl", manifest,
                              root / "nonexistent-baseline.json", output)
            self.assertFalse(output.exists())

    def test_one_read_parsed_bytes_are_exactly_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_bytes(self.raw)
            with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset({self.digest})):
                self.assertTrue(load_json(path, require_gui_admission=True)["rights_cleared"])
            path.write_bytes(self.raw + b" ")
            with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset({self.digest})):
                with self.assertRaisesRegex(ValueError, DENIAL):
                    load_json(path, require_gui_admission=True)


if __name__ == "__main__":
    unittest.main()
