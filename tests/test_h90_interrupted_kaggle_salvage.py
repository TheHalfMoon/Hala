"""H90 CPU-only synthetic interruption cases; not actual Kaggle model proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h73.kaggle_h73_cpu import TRAIN_SHA
from h90.recover_interrupted import (SCHEMA, admit_interrupted_checkpoint,
                                     expected_research_configuration)
from training.artifacts import sha256_file
from training.checkpoints import completed_checkpoint_result, publish_checkpoint


class InterruptedKaggleResearchCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / "hala-h88-research-checkpoint"
        self.config = expected_research_configuration()
        self.runtime = {
            "status": "tested", "cuda_available": True,
            "devices": [{"name": "Tesla T4", "tensor_gradient_changed": True}],
            "qualification_steps": [{
                "step": 1, "gradient_norm_before_clip": 1.25,
                "changed_parameter_tensors_this_step": 2
            }],
        }
        self.state = {
            "config_sha256": hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            "dataset_sha256": TRAIN_SHA,
            "manifest_sha256": "d" * 64,
            "step": 1, "max_steps": 1,
            "metrics": [{"step": 1, "loss": 0.7, "seconds": 2.0}],
            "changed_parameter_tensors": 2, "peak_vram_bytes": 350000,
            "runtime": self.runtime,
        }

        def writer(stage):
            (stage / "adapter").mkdir()
            (stage / "adapter/adapter_model.safetensors").write_bytes(
                b"synthetic LOra bytes, NOT trained Qwen")
            (stage / "resume.pt").write_bytes(
                b"synthetic optimizer bytes, NOT a Torch optimizer")

        self.version = publish_checkpoint(self.output, self.state, writer)
        self.final = completed_checkpoint_result(self.version, self.state, self.config)

    def write_final(self):
        (self.output / "training-result.json").write_text(
            json.dumps(self.final, sort_keys=True), encoding="utf8")

    def test_recovers_verified_complete_version_without_any_final_receipts(self):
        report = admit_interrupted_checkpoint(self.root)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"],
                         "RECOVERABLE_LOCAL_BYTES_ONLY_NEEDS_PROVIDER_PROOF")
        self.assertEqual(report["optimizer_step"], 1)
        self.assertFalse(report["train_result_present"])
        self.assertFalse(report["provider_authenticated_readback_verified"])
        self.assertFalse(report["model_training_independently_verified"])
        self.assertFalse(report["model_quality_evaluated"])
        self.assertFalse(report["commercial_use"])

    def test_accepts_matching_finalization_without_h87_h88_receipt(self):
        self.write_final()
        report = admit_interrupted_checkpoint(self.root)
        self.assertTrue(report["train_result_present"])
        self.assertEqual(report["checkpoint_sha256"],
                         sha256_file(self.version / "checkpoint.json"))

    def test_failed_attempt_receipt_cannot_block_salvage(self):
        (self.root / "hala-h88-research-attempt.json").write_text(
            json.dumps({"status": "FAILED_NO_QUALIFIED_MODEL_CHECKPOINT"}))
        self.assertEqual(admit_interrupted_checkpoint(self.root)["optimizer_step"], 1)

    def test_tampered_optimizer_fails_original_checksum_gate(self):
        (self.version / "resume.pt").write_bytes(b"different bytes")
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_missing_adapter_fails(self):
        (self.version / "adapter/adapter_model.safetensors").unlink()
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_missing_latest_pointer_not_salvaged(self):
        (self.output / "latest.json").unlink()
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_wrong_pinned_config_identity_not_salvaged(self):
        from unittest.mock import patch
        with patch("h90.recover_interrupted.expected_research_configuration",
                   return_value={**self.config, "learning_rate": 0.0001}):
            with self.assertRaises(ValueError):
                admit_interrupted_checkpoint(self.root)

    def test_wrong_training_dataset_state_not_salvaged(self):
        from unittest.mock import patch
        with patch("h90.recover_interrupted.TRAIN_SHA", "e" * 64):
            with self.assertRaises(ValueError):
                admit_interrupted_checkpoint(self.root)

    def test_nonfinite_or_zero_gradient_not_qualified(self):
        from unittest.mock import patch
        for value in (0, float("nan"), -1):
            with self.subTest(value=value):
                fake_state = json.loads(json.dumps(self.state))
                fake_state["runtime"]["qualification_steps"][0]["gradient_norm_before_clip"] = value
                with patch("h90.recover_interrupted.load_checkpoint",
                           return_value=(self.version, fake_state)):
                    with self.assertRaises(ValueError):
                        admit_interrupted_checkpoint(self.root)

    def test_final_result_must_match_checkpoint(self):
        self.write_final()
        self.final["metrics"][0]["loss"] = 0.1
        self.write_final()
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_final_result_cannot_claim_qualified_adapter(self):
        self.final["adapter_qualified"] = True
        self.write_final()
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_final_result_must_bind_exact_artifact_manifest(self):
        self.final["artifacts"]["resume.pt"] = "f" * 64
        self.write_final()
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root)

    def test_no_directory_is_refused(self):
        with self.assertRaises(ValueError):
            admit_interrupted_checkpoint(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
