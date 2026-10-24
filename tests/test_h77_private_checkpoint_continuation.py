"""H77 isolated CPU tests; all private provider calls are synthetic and mocked."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h77.private_checkpoint_continuation import supervise_then_private_readback
from tests.test_h65_provider_publish import fixture
from training.checkpoints import load_checkpoint

TOKEN = "only-synthetic-private-credential"
SHA = "191f593534dbfb68238e329f82617d466779706b"
SCHEMA = "hala.h65.supervised-diagnostic.v1"


def verified_response(root):
    _, state = load_checkpoint(Path(root), None)
    pointer = json.loads((Path(root) / "latest.json").read_text("utf8"))
    return {
        "schema": "hala.h65.private-checkpoint-readback.v1",
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "provider_published": True,
        "provider_is_private": True,
        "provider_readback_verified": True,
        "host_copy_removed": False,
        "private_provider_ref": "abdulazizshehri/hala-h77-private-verified-only",
        "step": state["step"],
    }


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dest = Path(self.temp.name) / "receipts"
        self.env = {"KAGGLE_API_TOKEN": TOKEN}

    def monitor(self, status="COMPLETED_WITH_HOST_EGRESS", polls=1,
                prepare_checkpoint=True):
        def _monitor(session, sha, dest, **kw):
            self.assertEqual(session, "existing-t4")
            self.assertEqual(sha, SHA)
            self.assertEqual(Path(dest), self.dest.resolve())
            if prepare_checkpoint:
                fixture(Path(dest) / "checkpoint-staging")
            return {"schema": SCHEMA, "status": status, "reason": "host_evidence_only",
                    "successful_egress_polls": polls,
                    "private_provider_readback_verified": False, "child_exit": 0 if status != "FAILED" else 137}
        return _monitor

    def call(self, **kwargs):
        return supervise_then_private_readback(
            "existing-t4", SHA, self.dest,
            environ=self.env, monitor=kwargs.pop("monitor", self.monitor()),
            publisher=kwargs.pop("publisher", lambda stage, token, work, **kw: verified_response(stage)),
            **kwargs)

    def test_valid_private_published_readback_receipt_and_staging_kept(self):
        calls = []
        def publish(stage, token, work, **kwargs):
            calls.append((stage, token, work, kwargs))
            return verified_response(stage)
        result = self.call(publisher=publish, max_steps=1)
        self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")
        self.assertTrue(result["private_provider_readback_verified"])
        self.assertEqual(result["private_publication_status"], "PRIVATE_READBACK_VERIFIED")
        self.assertEqual(result["private_checkpoint_step"], 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], TOKEN)
        self.assertIs(calls[0][3]["remove_host_copy"], False)
        self.assertEqual(load_checkpoint(self.dest / "checkpoint-staging", None)[1]["step"], 2)
        assert (self.dest / "h77-private-continuation.json").exists()
        receipt = (self.dest / "h77-private-continuation.json").read_text("utf8")
        event = (self.dest / "monitor-receipts.jsonl").read_text("utf8")
        self.assertNotIn(TOKEN, receipt + event)
        self.assertFalse(json.loads(receipt).get("host_copy_removed", False))
        self.assertEqual(json.loads(event.splitlines()[-1])["event"], "h77_private_continuation")

    def test_gpu_run_failed_but_salvaged_checkpoint_is_still_published_without_false_success(self):
        result = self.call(monitor=self.monitor("FAILED", 1, True))
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["private_provider_readback_verified"])
        self.assertEqual(result["child_exit"], 137)
        self.assertFalse(result.get("adapter_qualified", False))

    def test_missing_host_checkpoint_never_uploads_even_with_fake_positive_poll(self):
        uploader = mock.Mock(return_value={"provider_readback_verified": True})
        result = self.call(monitor=self.monitor(prepare_checkpoint=False), publisher=uploader)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "private_readback_failed")
        self.assertFalse(result["private_provider_readback_verified"])
        uploader.assert_not_called()

    def test_no_egress_does_not_upload_or_claim_success(self):
        uploader = mock.Mock()
        result = self.call(monitor=self.monitor(polls=0, prepare_checkpoint=False),
                           publisher=uploader)
        uploader.assert_not_called()
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["private_publication_status"], "SKIPPED_NO_VERIFIED_HOST_CHECKPOINT")
        self.assertFalse(result["private_provider_readback_verified"])

    def test_provider_readback_missing_false_public_or_wrong_step_never_qualifies(self):
        cases = [
            {"provider_readback_verified": False},
            {"provider_published": False},
            {"provider_is_private": False},
            {"host_copy_removed": True},
            {"step": 1},
            {"private_provider_ref": ""},
        ]
        for override in cases:
            with self.subTest(override=override):
                with tempfile.TemporaryDirectory() as tmp:
                    self.dest = Path(tmp) / "run"
                    def publish(stage, token, work, **kw):
                        return {**verified_response(stage), **override}
                    result = self.call(publisher=publish)
                    self.assertEqual(result["status"], "FAILED")
                    self.assertFalse(result["private_provider_readback_verified"])
                    self.assertEqual(result["private_publication_status"], "PRIVATE_READBACK_FAILED")
                    load_checkpoint(self.dest / "checkpoint-staging", None)

    def test_failed_upload_never_removes_only_checkpoint(self):
        publisher = mock.Mock(side_effect=RuntimeError("invalid-secret-should-not-print"))
        result = self.call(publisher=publisher)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["private_publication_error_class"], "RuntimeError")
        self.assertNotIn("invalid-secret", json.dumps(result))
        self.assertFalse(result["private_provider_readback_verified"])
        load_checkpoint(self.dest / "checkpoint-staging", None)

    def test_missing_or_unsafe_environment_refuses_before_gpu_operation(self):
        monitor = mock.Mock()
        publisher = mock.Mock()
        for token_env, environs in [
            ("KAGGLE_API_TOKEN", {}),
            ("KAGGLE_API_TOKEN", {"KAGGLE_API_TOKEN": "no"}),
            ("../../KEY", {"../../KEY": TOKEN}),
            ("H77 HACK", {"H77 HACK": TOKEN}),
        ]:
            with self.subTest(token_env=token_env):
                with self.assertRaises(ValueError):
                    supervise_then_private_readback("existing-t4", SHA, self.dest,
                        token_env=token_env, environ=environs,
                        monitor=monitor, publisher=publisher)
        monitor.assert_not_called()
        publisher.assert_not_called()

    def test_no_private_opt_in_default_h65_cli_invokes_legacy_monitor_only(self):
        from h65 import safe_diagnostic
        argv = ["--session", "existing-t4", "--pinned-sha", SHA,
                "--destination", str(self.dest)]
        with mock.patch.object(safe_diagnostic, "supervise",
                               return_value={"status": "COMPLETED_WITH_HOST_EGRESS"}) as monitor:
            self.assertEqual(safe_diagnostic.main(argv), 0)
        monitor.assert_called_once()

    def test_opt_in_cli_wires_private_readback_and_fails_on_unverified_result(self):
        from h65 import safe_diagnostic
        import h77.private_checkpoint_continuation as continuation
        argv = ["--session", "existing-t4", "--pinned-sha", SHA,
                "--destination", str(self.dest), "--private-readback-after-egress"]
        with mock.patch.object(continuation, "supervise_then_private_readback",
                               return_value={"status": "FAILED"}) as controller:
            self.assertEqual(safe_diagnostic.main(argv), 1)
        controller.assert_called_once()


if __name__ == "__main__":
    unittest.main()
