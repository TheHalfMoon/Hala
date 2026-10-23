"""H65 host supervisor behavior under deterministic fake Colab sessions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h65.safe_diagnostic import supervise


class FakeChild:
    pid = 1357
    def __init__(self, results):
        self.results = list(results)
        self.returncode = None
        self.terminated = False
        self.calls = 0

    def poll(self):
        self.calls += 1
        if self.returncode is not None:
            return self.returncode
        value = self.results.pop(0) if self.results else 0
        if value is not None:
            self.returncode = value
        return value

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -15 if self.terminated else 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class SupervisorTests(unittest.TestCase):
    def invoke(self, *, states=(None, 0), snapshots=None, hardware=None, steps="train"):
        child = FakeChild(states)
        process = mock.Mock(return_value=child)
        values = iter(snapshots or [
            {"status": "VERIFIED_HOST_STAGING", "step": 1},
            {"status": "ALREADY_VERIFIED", "step": 1},
        ])
        snapshot = mock.Mock(side_effect=lambda *a: next(values, {"status": "ALREADY_VERIFIED"}))
        h = mock.Mock(side_effect=hardware) if isinstance(hardware, list) else mock.Mock(
            return_value="T4" if hardware is None else hardware
        )
        with tempfile.TemporaryDirectory() as root:
            report = supervise("hala-h65", "a"*40, Path(root),
                               interval=30, max_seconds=90, steps=steps,
                               runner=process, snapshot=snapshot,
                               hardware=h, sleep=lambda _: None, clock=lambda: 0.0,
                               checkpoint_verifier=lambda *args: {"status": "SYNTHETIC_VALIDATION_ONLY"})
            receipts = (Path(root) / "monitor-receipts.jsonl").read_text().splitlines()
            return report, child, process, snapshot, [json.loads(x) for x in receipts]

    def test_success_needs_actual_checkpoint_egress(self):
        result, child, process, snapshot, receipts = self.invoke()
        self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")
        self.assertFalse(result["private_provider_readback_verified"])
        self.assertGreaterEqual(result["successful_egress_polls"], 1)
        self.assertEqual(process.call_count, 1)
        self.assertTrue(any(r["event"] == "egress" for r in receipts))

    def test_missing_checkpoint_never_masquerades_as_trained_model(self):
        result, *_ = self.invoke(
            snapshots=[{"status": "NO_PUBLISHED_CHECKPOINT"},
                       {"status": "NO_PUBLISHED_CHECKPOINT"}])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "missing_verified_checkpoint_egress")

    def test_child_exit_nonzero_is_not_success_despite_valid_checkpoint(self):
        result, *_ = self.invoke(states=(None, 137))
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["child_exit"], 137)

    def test_three_missing_session_checks_terminate_abandoned_host_wait(self):
        result, child, _, _, records = self.invoke(
            states=(None, None, None, None),
            hardware=["T4", "", "", ""])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "provider_session_lost")
        self.assertTrue(child.terminated)
        self.assertTrue(any(r.get("reason") == "provider_session_lost" for r in records))

    def test_cpu_session_is_rejected_before_child_spawn(self):
        with tempfile.TemporaryDirectory() as root:
            fake = mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "dedicated T4"):
                supervise("hala-h65", "a"*40, Path(root), runner=fake,
                          hardware=lambda _: "CPU")
            fake.assert_not_called()

    def test_snapshot_validation_exception_does_not_count_as_a_success(self):
        result, *_ = self.invoke(
            snapshots=[{"status": "NO_PUBLISHED_CHECKPOINT"},
                       {"status": "NO_PUBLISHED_CHECKPOINT"}])
        self.assertEqual(result["successful_egress_polls"], 0)

    def test_no_train_qualifies_on_child_exit_without_checkpoint(self):
        result, *_ = self.invoke(
            steps="export,curriculum",
            snapshots=[{"status": "NO_PUBLISHED_CHECKPOINT"},
                       {"status": "NO_PUBLISHED_CHECKPOINT"}])
        self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")

    def test_invalid_pinned_sha_blocks_any_remote_operation(self):
        with tempfile.TemporaryDirectory() as root:
            runner = mock.Mock()
            with self.assertRaisesRegex(ValueError, "40-character"):
                supervise("hala-h65", "../invalid", Path(root), runner=runner)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
