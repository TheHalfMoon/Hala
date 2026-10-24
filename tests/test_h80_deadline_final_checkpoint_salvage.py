"""H80 deterministic closeout tests; no Colab, GPU or source media used."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h65.safe_diagnostic import supervise
from tests.test_h65_safe_diagnostic import FakeChild
from training.checkpoints import load_checkpoint
from tests.test_h65_provider_publish import fixture


SHA = "c3d837e126172e963480823ef85cd8a72b3d66c2"


class FinalEgressOnDeadline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "host"
        self.snapshots = mock.Mock()
        self.child = FakeChild([None, None, None, None])

    def run_deadline(self, snapshots, **kwargs):
        self.snapshots.side_effect = snapshots
        ticks = iter([0.0, 31.0])
        result = supervise("preexisting-t4", SHA, self.root,
                           max_seconds=30, interval=30,
                           steps="train", max_steps=1,
                           runner=lambda *a, **kw: self.child,
                           snapshot=self.snapshots,
                           hardware=lambda _: "T4",
                           sleep=lambda seconds: self.fail("unexpected monitor sleep"),
                           clock=lambda: next(ticks),
                           **kwargs)
        records = [json.loads(line) for line in
                   (self.root / "monitor-receipts.jsonl").read_text("utf8").splitlines()]
        return result, records

    def test_last_readonly_egress_after_deadline_salvages_step_but_does_not_claim_training_success(self):
        def recover(session, stage):
            if self.snapshots.call_count == 2:
                fixture(Path(stage))
                return {"status": "VERIFIED_HOST_STAGING", "step": 2}
            return {"status": "NO_PUBLISHED_CHECKPOINT"}
        report, receipts = self.run_deadline(recover)
        self.assertTrue(self.child.terminated)
        self.assertEqual(self.snapshots.call_count, 2)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["reason"], "monitor_deadline")
        self.assertEqual(report["successful_egress_polls"], 1)
        self.assertEqual(report["deadline_final_salvage"], "ATTEMPTED_ONCE")
        self.assertEqual(report["last_egress"], "VERIFIED_HOST_STAGING")
        self.assertFalse(report["private_provider_readback_verified"])
        self.assertIsNone(report["completed_checkpoint_evidence"])
        self.assertEqual(load_checkpoint(self.root / "checkpoint-staging", None)[1]["step"], 2)
        self.assertEqual(receipts[-1]["event"], "final")
        self.assertEqual(receipts[-1]["status"], "FAILED")
        self.assertEqual([x["event"] for x in receipts].count("deadline_final_egress"), 1)
        self.assertEqual([x["event"] for x in receipts].count("stop"), 1)

    def test_late_salvage_status_unverified_does_not_increment_success(self):
        report, records = self.run_deadline(lambda *a: {"status": "METADATA_UNAVAILABLE"})
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["successful_egress_polls"], 0)
        self.assertEqual(report["last_egress"], "METADATA_UNAVAILABLE")
        self.assertEqual(self.snapshots.call_count, 2)
        self.assertTrue(any(x["event"] == "deadline_final_egress" for x in records))

    def test_closeout_snapshot_exception_is_recorded_redacted_and_not_success(self):
        def fail_second(*args):
            if self.snapshots.call_count == 1:
                return {"status": "NO_PUBLISHED_CHECKPOINT"}
            raise IOError("synthetic private provider detail must not leak")
        report, records = self.run_deadline(fail_second)
        self.assertEqual(report["last_egress"], "EGRESS_FAILED")
        self.assertEqual(report["status"], "FAILED")
        errors = [x for x in records if x["event"] == "deadline_final_egress_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_class"], "OSError")
        self.assertNotIn("synthetic private provider detail", json.dumps(records))

    def test_missing_provider_does_not_make_one_more_remote_call(self):
        hardware = iter(["T4", "", "", ""])
        child = FakeChild([None] * 10)
        snapshot = mock.Mock(return_value={"status": "NO_PUBLISHED_CHECKPOINT"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = supervise("preexisting-t4", SHA, root,
                               steps="train", interval=30, max_seconds=90,
                               runner=lambda *a, **kw: child,
                               snapshot=snapshot, hardware=lambda _: next(hardware),
                               clock=lambda: 0.0, sleep=lambda _: None)
            records = [json.loads(line) for line in
                       (root / "monitor-receipts.jsonl").read_text("utf8").splitlines()]
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["reason"], "provider_session_lost")
        self.assertEqual(report["deadline_final_salvage"], "SKIPPED_PROVIDER_MISSING")
        self.assertEqual(snapshot.call_count, 3)
        self.assertTrue(child.terminated)
        self.assertEqual(records[-1]["event"], "final")
        self.assertFalse(any(r["event"] == "deadline_final_egress" for r in records))

    def test_h77_can_privately_back_up_final_partial_salvage_without_rewriting_failure(self):
        from h77.private_checkpoint_continuation import supervise_then_private_readback
        from tests.test_h77_private_checkpoint_continuation import verified_response
        snapshots = mock.Mock()
        def egress(session, stage):
            if snapshots.call_count == 2:
                fixture(Path(stage))
                return {"status": "VERIFIED_HOST_STAGING", "step": 2}
            return {"status": "NO_PUBLISHED_CHECKPOINT"}
        snapshots.side_effect = egress
        calls = []
        def simulated_publish(stage, token, work, **kwargs):
            calls.append((str(stage), token, kwargs.get("remove_host_copy")))
            return verified_response(stage)
        child = FakeChild([None, None])
        ticks = iter([0.0, 31.0])
        def monitor(session, sha, dest, **kw):
            return supervise(session, sha, dest,
                             max_seconds=30, interval=30,
                             steps="train", max_steps=3,
                             runner=lambda *a, **kwargs: child,
                             snapshot=snapshots, hardware=lambda _: "T4",
                             clock=lambda: next(ticks),
                             sleep=lambda seconds: self.fail("unexpected sleep"))
        result = supervise_then_private_readback(
            "preexisting-t4", SHA, self.root,
            environ={"KAGGLE_API_TOKEN": "synthetic-credential-only"},
            monitor=monitor, publisher=simulated_publish)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "monitor_deadline")
        self.assertTrue(result["private_provider_readback_verified"])
        self.assertEqual(result["private_publication_status"], "PRIVATE_READBACK_VERIFIED")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][2], False)
        self.assertEqual(load_checkpoint(self.root / "checkpoint-staging", None)[1]["step"], 2)
        self.assertNotIn("synthetic-credential-only",
                         (self.root / "h77-private-continuation.json").read_text("utf8"))

    def test_positive_snapshot_prior_to_deadline_remains_rescuable_but_failed(self):
        report, records = self.run_deadline(
            lambda *a: {"status": "ALREADY_VERIFIED", "step": 1})
        self.assertEqual(report["successful_egress_polls"], 2)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["reason"], "monitor_deadline")
        self.assertEqual(len([x for x in records if x["event"] == "final"]), 1)


if __name__ == "__main__":
    unittest.main()
