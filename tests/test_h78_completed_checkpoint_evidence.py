"""H78: no provider/GPU; real SHA-checked synthetic checkpoint artifacts."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h78.checkpoint_evidence import checkpoint_sha_if_present, qualify_completed_host_checkpoint
from h65.safe_diagnostic import supervise
from tests.test_h65_safe_diagnostic import FakeChild
from training.checkpoints import publish_checkpoint, load_checkpoint


def fixture(root: Path, *, step=2, max_steps=2, changed=1, gradient=1.5,
            steps_in_receipt=None):
    step_receipts = [{"step": i, "gradient_norm_before_clip": gradient,
                      "changed_parameter_tensors_this_step": 2}
                     for i in range(1, 1 + (steps_in_receipt if steps_in_receipt is not None else step))]
    state = {
        "config_sha256": "1"*64,
        "dataset_sha256": "2"*64,
        "manifest_sha256": "3"*64,
        "step": step, "max_steps": max_steps,
        "metrics": [{"step": i, "loss": 0.5, "seconds": 1.0} for i in range(1, step+1)],
        "changed_parameter_tensors": changed, "peak_vram_bytes": 10000,
        "runtime": {"qualification_steps": step_receipts},
    }
    def write(stage):
        (stage / "adapter").mkdir()
        (stage / "adapter" / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
        (stage / "resume.pt").write_bytes(b"synthetic-resume")
    publish_checkpoint(root, state, write)
    return load_checkpoint(root, None)


class CompletedEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkpoint-staging"

    def test_fresh_full_requested_step_returns_bounded_truthful_proof(self):
        fixture(self.root)
        evidence = qualify_completed_host_checkpoint(self.root, 2, None)
        self.assertEqual(evidence["status"], "VERIFIED_COMPLETED_HOST_CHECKPOINT")
        self.assertEqual(evidence["step"], 2)
        self.assertEqual(evidence["max_steps"], 2)
        self.assertEqual(evidence["dataset_sha256"], "2"*64)
        self.assertFalse(evidence["adapter_qualified"])
        self.assertFalse(evidence["private_provider_readback_verified"])
        self.assertTrue(evidence["evaluation_required"])

    def test_prior_valid_snapshot_sha_is_detected_before_launch(self):
        fixture(self.root)
        fingerprint = checkpoint_sha_if_present(self.root)
        self.assertEqual(len(fingerprint), 64)
        with self.assertRaisesRegex(ValueError, "stale preexisting"):
            qualify_completed_host_checkpoint(self.root, 2, fingerprint)

    def test_missing_checkpoint_never_qualifies(self):
        self.assertIsNone(checkpoint_sha_if_present(self.root))
        with self.assertRaises((ValueError, OSError)):
            qualify_completed_host_checkpoint(self.root, 2, None)

    def test_incomplete_earlier_step_is_rejected_even_if_egress_successful(self):
        fixture(self.root, step=1, max_steps=3)
        with self.assertRaisesRegex(ValueError, "complete gradient-bearing"):
            qualify_completed_host_checkpoint(self.root, 3, None)

    def test_requested_budget_disagrees_with_checkpoint_identity(self):
        fixture(self.root, step=1, max_steps=1)
        with self.assertRaisesRegex(ValueError, "complete gradient-bearing"):
            qualify_completed_host_checkpoint(self.root, 2, None)

    def test_unchanged_adapter_is_not_a_gradient_bearing_result(self):
        fixture(self.root, changed=0)
        with self.assertRaisesRegex(ValueError, "gradient-bearing"):
            qualify_completed_host_checkpoint(self.root, 2, None)

    def test_missing_per_step_gradient_receipt_fails(self):
        fixture(self.root, steps_in_receipt=1)
        with self.assertRaisesRegex(ValueError, "full real optimizer-step"):
            qualify_completed_host_checkpoint(self.root, 2, None)

    def test_zero_or_nonfinite_gradient_receipt_fails(self):
        for value in (0, -1):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "stage"
                fixture(root, gradient=value)
                with self.assertRaisesRegex(ValueError, "updated-gradient receipt"):
                    qualify_completed_host_checkpoint(root, 2, None)
        # The original immutable serializer rejects NaN/Inf even earlier,
        # before an invalid receipt can become a published checkpoint.
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "Out of range"):
                    fixture(Path(tmp) / "stage", gradient=value)

    def test_corrupted_optimizer_artifact_is_not_qualified(self):
        version, _ = fixture(self.root)
        (version / "resume.pt").write_bytes(b"corrupted optimizer")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            qualify_completed_host_checkpoint(self.root, 2, None)

    def test_invalid_budget_never_loads_checkpoint(self):
        for budget in [0, -1, 601, True, 1.5]:
            with self.subTest(budget=budget), mock.patch("h78.checkpoint_evidence.load_checkpoint") as load:
                with self.assertRaises(ValueError):
                    qualify_completed_host_checkpoint(self.root, budget, None)
                load.assert_not_called()


class SupervisorAdmission(unittest.TestCase):
    def test_successful_child_without_finished_checkpoint_is_a_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "checkpoint-staging", step=1, max_steps=3)
            child = FakeChild([None, 0])
            report = supervise("existing-t4", "a"*40, root,
                               max_steps=3, steps="train",
                               runner=lambda *a, **kw: child,
                               snapshot=lambda *a: {"status": "ALREADY_VERIFIED", "step": 1},
                               hardware=lambda _: "T4",
                               sleep=lambda _: None, clock=lambda: 0.0)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["reason"], "unqualified_completed_checkpoint")
            self.assertEqual(report["successful_egress_polls"], 2)
            self.assertIsNone(report["completed_checkpoint_evidence"])
            self.assertEqual(load_checkpoint(root / "checkpoint-staging", None)[1]["step"], 1)

    def test_finished_sha_verified_checkpoint_is_accepted_at_exact_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = FakeChild([None, 0])
            def snapshot(session, destination):
                if not (destination / "latest.json").exists():
                    fixture(destination, step=2, max_steps=2)
                return {"status": "VERIFIED_HOST_STAGING", "step": 2}
            report = supervise("existing-t4", "a"*40, root, steps="train", max_steps=2,
                               runner=lambda *a, **kw: child, snapshot=snapshot,
                               hardware=lambda _: "T4",
                               sleep=lambda _: None, clock=lambda: 0.0)
            self.assertEqual(report["status"], "COMPLETED_WITH_HOST_EGRESS")
            self.assertEqual(report["reason"], "verified_completed_host_checkpoint")
            self.assertEqual(report["completed_checkpoint_evidence"]["step"], 2)
            self.assertFalse(report["private_provider_readback_verified"])

    def test_same_old_checkpoint_present_before_launch_is_not_new_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "checkpoint-staging", step=1, max_steps=1)
            child = FakeChild([None, 0])
            report = supervise("existing-t4", "a"*40, root, steps="train", max_steps=1,
                               runner=lambda *a, **kw: child,
                               snapshot=lambda *a: {"status": "ALREADY_VERIFIED", "step": 1},
                               hardware=lambda _: "T4",
                               sleep=lambda _: None, clock=lambda: 0.0)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["reason"], "unqualified_completed_checkpoint")
            self.assertIsNone(report["completed_checkpoint_evidence"])

    def test_child_exit_137_remains_failure_but_partial_host_copy_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "checkpoint-staging", step=1, max_steps=3)
            child = FakeChild([None, 137])
            report = supervise("existing-t4", "a"*40, root, steps="train", max_steps=3,
                               runner=lambda *a, **kw: child,
                               snapshot=lambda *a: {"status": "ALREADY_VERIFIED", "step": 1},
                               hardware=lambda _: "T4",
                               sleep=lambda _: None, clock=lambda: 0.0)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["reason"], "child_nonzero")
            self.assertEqual(load_checkpoint(root / "checkpoint-staging", None)[1]["step"], 1)


if __name__ == "__main__":
    unittest.main()
