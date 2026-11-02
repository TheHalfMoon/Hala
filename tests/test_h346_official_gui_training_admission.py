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

    def test_checkpoint_identity_uses_admitted_bytes_not_reopened_path(self):
        # Entirely invented fixture: patch only this test's H199 allowlist.
        # A source loader can observe a replaced local path AFTER the original
        # bytes were admitted. Recovery must use the originally admitted SHA.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({
                "model_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": "a" * 40,
                "precision": "fp16", "max_steps": 1, "batch_size": 1,
                "gradient_accumulation": 1, "learning_rate": 0.0001,
                "lora_rank": 4, "platform": "kaggle", "download_remote_weights": True
            }), encoding="utf-8")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "model_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": "a" * 40,
                "evidence": "real_inference", "evaluated_tasks": 1,
                "verified_success_rate": 0.0, "deficit": "Invented test baseline"
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_bytes(json.dumps({
                "schema": "hala.training.manifest.v1", "split": "train",
                "rights_cleared": True, "source_kind": "synthetic",
                "dataset_sha256": "b" * 64,
                "families": ["invented-train"], "held_out_families": ["invented-dev"]
            }, sort_keys=True).encode("utf-8"))
            admitted = sha256(manifest.read_bytes()).hexdigest()

            def swap_manifest(*_args):
                manifest.write_bytes(b'{"unadmitted": true}')
                return []  # never used: resume recovery intercepts before GPU

            with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS",
                       frozenset({admitted})), \
                 patch("h87.research_runtime.require_training_provider",
                       return_value="kaggle_noncommercial_research"), \
                 patch("training.scripts.train_adapter.load_authoring_records",
                       side_effect=swap_manifest), \
                 patch("h82.orphan_recovery.recover_orphaned_checkpoint",
                       side_effect=RuntimeError("CAPTURED_ADMITTED_IDENTITY")) as recover, \
                 patch("training.scripts.train_adapter.probe") as gpu_probe:
                with self.assertRaisesRegex(RuntimeError, "CAPTURED_ADMITTED_IDENTITY"):
                    train(config, root / "invented.jsonl", manifest, baseline,
                          root / "output", resume=True)
                self.assertEqual(recover.call_args.args[1]["manifest_sha256"], admitted)
                self.assertNotEqual(sha256(manifest.read_bytes()).hexdigest(), admitted)
                gpu_probe.assert_not_called()

    def test_only_admitted_loader_can_expose_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_bytes(self.raw)
            with self.assertRaisesRegex(ValueError, "Only an H199-admitted"):
                load_json(path, with_admitted_digest=True)
            with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS",
                       frozenset({self.digest})):
                parsed, digest = load_json(
                    path, require_gui_admission=True, with_admitted_digest=True)
            self.assertEqual(digest, self.digest)
            self.assertTrue(parsed["rights_cleared"])


if __name__ == "__main__":
    unittest.main()
