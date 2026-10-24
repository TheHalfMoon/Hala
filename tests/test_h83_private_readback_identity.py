"""H83: provider readback is bound to the exact immutable checkpoint SHA.

Tests are synthetic, provider-free and do not execute a GPU or model.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h65.provider_publish import publish_latest
from h77.private_checkpoint_continuation import supervise_then_private_readback
from h61.publish_checkpoint import collect, sha256_file
from tests.test_h65_provider_publish import fixture
from tests.test_h77_private_checkpoint_continuation import verified_response
from training.checkpoints import load_checkpoint


class PrivateReadbackIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "host"
        self.work = Path(self.temp.name) / "private"
        fixture(self.root)
        self.original_pointer = json.loads((self.root / "latest.json").read_text("utf8"))

    def uploader(self, stage, ref, title, subtitle, token, work, **overrides):
        members = collect(stage)
        reply = {
            "provider_ref": ref,
            "member_sha256": {n: sha256_file(p) for n, p in members.items()},
            "published_members": len(members),
            "published_bytes": sum(p.stat().st_size for p in members.values()),
            "provider_published": True,
            "provider_is_private": True,
            "anonymous_provider_http_status": 403,
            "provider_readback_verified": True,
        }
        reply.update(overrides)
        return reply

    def test_exact_provider_items_and_sha_admit_without_deleting_host_copy(self):
        result = publish_latest(self.root, "synthetic-token-abcd", self.work,
                                uploader=self.uploader)
        self.assertEqual(result["checkpoint_sha256"],
                         self.original_pointer["checkpoint_sha256"])
        self.assertTrue(result["provider_readback_verified"])
        self.assertFalse(result["host_copy_removed"])
        self.assertEqual(load_checkpoint(self.root, None)[1]["step"], 2)

    def test_a_different_provider_dataset_reference_is_not_verified(self):
        with self.assertRaisesRegex(ValueError, "exact bundle"):
            publish_latest(self.root, "synthetic-token-abcd", self.work,
                           uploader=lambda *a: self.uploader(*a, provider_ref="other/bundle"),
                           remove_host_copy=True)
        load_checkpoint(self.root, None)
        self.assertFalse((self.work / "private-readback-receipt.json").exists())

    def test_mismatched_published_item_bytes_refuses_private_success(self):
        for bad in ("hash", "missing-file", "unexpected-file",
                    "wrong-published-count", "wrong-byte-budget"):
            with self.subTest(bad=bad):
                def forged(*args):
                    receipt = self.uploader(*args)
                    if bad == "hash":
                        member = next(iter(receipt["member_sha256"]))
                        receipt["member_sha256"][member] = "f" * 64
                    elif bad == "missing-file":
                        receipt["member_sha256"].pop(next(iter(receipt["member_sha256"])))
                    elif bad == "unexpected-file":
                        receipt["member_sha256"]["extra"] = "0" * 64
                    elif bad == "wrong-published-count":
                        receipt["published_members"] += 1
                    elif bad == "wrong-byte-budget":
                        receipt["published_bytes"] -= 1
                    return receipt
                with self.assertRaisesRegex(ValueError, "exact bundle"):
                    publish_latest(self.root, "synthetic-token-abcd", self.work,
                                   uploader=forged, remove_host_copy=True)
                load_checkpoint(self.root, None)
                self.assertFalse((self.work / "private-readback-receipt.json").exists())

    def test_host_checkpoint_mutated_while_uploading_is_not_reported_verified(self):
        def mutate_then_approve(stage, ref, title, subtitle, token, work):
            result = self.uploader(stage, ref, title, subtitle, token, work)
            version, _ = load_checkpoint(self.root, None)
            (version / "resume.pt").write_bytes(b"tampered-after-upload")
            return result
        with self.assertRaises(ValueError):
            publish_latest(self.root, "synthetic-token-abcd", self.work,
                           uploader=mutate_then_approve, remove_host_copy=True)
        self.assertTrue((self.root / "latest.json").exists())
        self.assertFalse((self.work / "private-readback-receipt.json").exists())


class ContinuationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dest = Path(self.temp.name) / "operator"

    def monitor(self, session, sha, target, **kw):
        fixture(Path(target) / "checkpoint-staging")
        return {"schema": "hala.h65.supervised-diagnostic.v1",
                "status": "COMPLETED_WITH_HOST_EGRESS",
                "successful_egress_polls": 1, "child_exit": 0}

    def run_private(self, mutate):
        def publisher(stage, token, work, **kw):
            return mutate(verified_response(stage))
        return supervise_then_private_readback(
            "existing-t4", "a" * 40, self.dest,
            monitor=self.monitor, publisher=publisher,
            environ={"KAGGLE_API_TOKEN": "synthetic-token-abcd"})

    def test_same_step_different_checkpoint_digest_never_qualifies(self):
        report = self.run_private(lambda row: {**row, "checkpoint_sha256": "f" * 64})
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["private_provider_readback_verified"])
        self.assertEqual(report["private_publication_status"], "PRIVATE_READBACK_FAILED")
        load_checkpoint(self.dest / "checkpoint-staging", None)
        self.assertNotIn("private_checkpoint_sha256", report)

    def test_missing_digest_or_invalid_schema_never_qualifies(self):
        for update in ({"checkpoint_sha256": None}, {"schema": "other"}):
            with self.subTest(update=update):
                with tempfile.TemporaryDirectory() as tmp:
                    self.dest = Path(tmp) / "receipts"
                    report = self.run_private(lambda row: {**row, **update})
                    self.assertEqual(report["status"], "FAILED")
                    self.assertFalse(report["private_provider_readback_verified"])
                    load_checkpoint(self.dest / "checkpoint-staging", None)

    def test_matching_digest_is_recorded_in_private_continuation_receipt(self):
        report = self.run_private(lambda row: row)
        ptr = json.loads((self.dest / "checkpoint-staging/latest.json").read_text("utf8"))
        self.assertEqual(report["status"], "COMPLETED_WITH_HOST_EGRESS")
        self.assertTrue(report["private_provider_readback_verified"])
        self.assertEqual(report["private_checkpoint_sha256"], ptr["checkpoint_sha256"])
        event = (self.dest / "monitor-receipts.jsonl").read_text("utf8")
        self.assertIn(ptr["checkpoint_sha256"], event)
        self.assertNotIn("synthetic-token-abcd", event)


if __name__ == "__main__":
    unittest.main()
