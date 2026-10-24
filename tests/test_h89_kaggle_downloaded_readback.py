"""H89 CPU fixtures verify checkpoint bytes without running actual Qwen/GPU."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from h73.kaggle_h73_cpu import TRAIN_SHA
from h89.readback import SOURCE_SHA, SCHEMA, admit_downloaded_output
from training.artifacts import sha256_file
from training.checkpoints import completed_checkpoint_result, publish_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION


class KaggleDownloadedCheckpointAdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / "hala-h88-research-checkpoint"
        self.config = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "platform": "kaggle",
            "precision": "fp16", "max_steps": 1, "batch_size": 1,
            "gradient_accumulation": 1, "lora_rank": 8,
            "audit_records": 32, "learning_rate": 0.00005,
        }
        self.runtime = {
            "status": "tested", "cuda_available": True,
            "devices": [{"name": "Tesla T4", "tensor_gradient_changed": True}],
            "qualification_steps": [{
                "step": 1, "gradient_norm_before_clip": 1.25,
                "changed_parameter_tensors_this_step": 2,
            }],
        }
        self.state = {
            "config_sha256": hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            "dataset_sha256": TRAIN_SHA, "manifest_sha256": "d" * 64,
            "step": 1, "max_steps": 1, "metrics": [
                {"step": 1, "loss": 0.7, "seconds": 3.0}],
            "changed_parameter_tensors": 2, "peak_vram_bytes": 350000,
            "runtime": self.runtime,
        }

        def writer(stage):
            (stage / "adapter").mkdir()
            (stage / "adapter/adapter_model.safetensors").write_bytes(
                b"synthetic adapter fixture, NOT Qwen weights")
            (stage / "resume.pt").write_bytes(
                b"synthetic optimizer fixture, NOT real torch weights")

        self.version = publish_checkpoint(self.output, self.state, writer)
        self.metadata_sha = sha256_file(self.version / "checkpoint.json")
        self.weights_sha = sha256_file(
            self.version / "adapter/adapter_model.safetensors")
        final = completed_checkpoint_result(self.version, self.state, self.config)
        (self.output / "training-result.json").write_text(
            json.dumps(final), encoding="utf8")
        self.attempt = {
            "schema": "hala.h88.noncommercial-private-kaggle-research-attempt.v1",
            "status": "REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED",
            "stage": "CHECKPOINT_RELOADED", "source_sha": SOURCE_SHA,
            "model_trained_verified": True,
            "provider_private_readback_verified": False,
            "commercial_use": False, "optimizer_step": 1,
            "checkpoint_sha256": self.metadata_sha,
        }
        self.training = {
            "schema": "hala.h87.noncommercial-kaggle-real-qwen-step.v1",
            "status": "REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "dataset_sha256": TRAIN_SHA, "commercial_use": False,
            "model_quality_evaluated": False,
            "provider_readback_verified": False, "step": 1,
            "checkpoint_version": self.version.name,
            "checkpoint_sha256": self.metadata_sha,
            "adapter_safetensors_sha256": self.weights_sha,
            "changed_parameter_tensors": 2, "gradient_norm_before_clip": 1.25,
        }
        self.write_receipts()

    def write_receipts(self):
        (self.root / "hala-h88-research-attempt.json").write_text(
            json.dumps(self.attempt), encoding="utf8")
        (self.root / "hala-h87-training-receipt.json").write_text(
            json.dumps(self.training), encoding="utf8")

    def test_complete_synthetic_pair_admitted_as_bytes_only(self):
        report = admit_downloaded_output(self.root)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["checkpoint_sha256"], self.metadata_sha)
        self.assertEqual(report["status"],
                         "DOWNLOADED_CHECKPOINT_BYTES_VERIFIED_ONLY")
        self.assertFalse(report["provider_private_readback_verified"])
        self.assertFalse(report["remote_provider_provenance_verified"])
        self.assertFalse(report["heldout_gui_evaluation_completed"])

    def test_training_not_completed_cannot_be_admitted(self):
        self.attempt["model_trained_verified"] = False
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_wrong_original_h71_dataset_sha_rejected(self):
        self.training["dataset_sha256"] = "c" * 64
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_same_step_different_checkpoint_sha_rejected(self):
        self.attempt["checkpoint_sha256"] = "e" * 64
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_tampered_adapter_bytes_rejected(self):
        (self.version / "adapter/adapter_model.safetensors").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_missing_optimizer_resume_rejected(self):
        (self.version / "resume.pt").unlink()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_forged_gradient_rejected(self):
        self.training["gradient_norm_before_clip"] = 0
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_changed_configuration_not_qualified(self):
        path = self.output / "training-result.json"
        final = json.loads(path.read_text())
        final["configuration"]["lora_rank"] = 16
        path.write_text(json.dumps(final))
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_missing_training_receipt_is_fail_closed(self):
        (self.root / "hala-h87-training-receipt.json").unlink()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_attempt_cannot_pretend_remote_authenticated_readback(self):
        self.attempt["provider_private_readback_verified"] = True
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)

    def test_training_receipt_cannot_pretend_remote_authenticated_readback(self):
        self.training["provider_readback_verified"] = True
        self.write_receipts()
        with self.assertRaises(ValueError):
            admit_downloaded_output(self.root)


if __name__ == "__main__":
    unittest.main()
