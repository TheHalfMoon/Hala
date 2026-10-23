"""H76 synthetic host-crash repair and immutable checkpoint identity tests."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h65.checkpoint_egress import snapshot_once
from tests.test_h65_checkpoint_egress import FakeColab, make_remote
from training.checkpoints import load_checkpoint, publish_checkpoint


class CrashRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.remote = make_remote(self.root / "remote")
        self.local = self.root / "local"
        self.fake = FakeColab(self.remote)

    def simulate_crash_after_version_rename_before_latest(self):
        remote_pointer = json.loads((self.remote / "latest.json").read_text("utf8"))
        source = self.remote / remote_pointer["version"]
        target = self.local / remote_pointer["version"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        self.assertFalse((self.local / "latest.json").exists())
        return target

    def snapshot(self):
        return snapshot_once("hala-h65", self.local, runner=self.fake)

    def test_restart_repairs_atomic_pointer_from_complete_cryptographically_identical_version(self):
        target = self.simulate_crash_after_version_rename_before_latest()
        result = self.snapshot()
        self.assertEqual(result["status"], "RECOVERED_HOST_STAGING")
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["downloaded_bytes"], 0)
        self.assertEqual(result["storage_kind"], "host_local_only_not_provider_readback")
        self.assertEqual(len(self.fake.calls), 2)  # latest.json and checkpoint.json
        version, progress = load_checkpoint(self.local, None)
        self.assertEqual(version, target.resolve())
        self.assertEqual(progress["step"], 1)
        self.assertEqual(self.snapshot()["status"], "ALREADY_VERIFIED")

    def test_partial_resume_artifact_never_gets_published(self):
        target = self.simulate_crash_after_version_rename_before_latest()
        (target / "resume.pt").write_bytes(b"truncated and invalid")
        with self.assertRaisesRegex(ValueError, "digest|file-set"):
            self.snapshot()
        self.assertFalse((self.local / "latest.json").exists())

    def test_extra_file_in_preexisting_version_is_rejected(self):
        target = self.simulate_crash_after_version_rename_before_latest()
        (target / "hidden.bin").write_bytes(b"untrusted extra payload")
        with self.assertRaisesRegex(ValueError, "file-set"):
            self.snapshot()
        self.assertFalse((self.local / "latest.json").exists())

    def test_locally_rewritten_metadata_never_recovers(self):
        target = self.simulate_crash_after_version_rename_before_latest()
        metadata = target / "checkpoint.json"
        content = json.loads(metadata.read_text("utf8"))
        content["state"]["dataset_sha256"] = "a"*64
        metadata.write_text(json.dumps(content), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.snapshot()
        self.assertFalse((self.local / "latest.json").exists())

    def test_no_pointer_and_no_version_downloads_normally(self):
        result = self.snapshot()
        self.assertEqual(result["status"], "VERIFIED_HOST_STAGING")
        self.assertGreater(result["downloaded_bytes"], 0)
        self.assertEqual(load_checkpoint(self.local, None)[1]["step"], 1)

    def test_identity_fork_rejected_before_remote_new_version_download(self):
        self.snapshot()
        foreign = self.root / "foreign"
        state = {
            "config_sha256": "1"*64,
            "dataset_sha256": "f"*64,  # changed training source
            "manifest_sha256": "3"*64,
            "step": 2, "max_steps": 10,
            "metrics": [{"step": 1, "loss": 1.0, "seconds": 0.1},
                        {"step": 2, "loss": 0.9, "seconds": 0.1}],
            "changed_parameter_tensors": 1,
            "peak_vram_bytes": 1024, "runtime": {},
        }
        def writer(where):
            (where / "adapter").mkdir()
            (where / "adapter" / "adapter_model.safetensors").write_bytes(b"foreign")
            (where / "resume.pt").write_bytes(b"foreign optimizer")
        publish_checkpoint(foreign, state, writer)
        self.fake = FakeColab(foreign)
        with self.assertRaisesRegex(ValueError, "training identity"):
            self.snapshot()
        # Only latest.json + checkpoint.json were read: never transfer foreign weights.
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(load_checkpoint(self.local, None)[1]["step"], 1)

    def test_identity_preserving_next_step_remains_supported(self):
        self.assertEqual(self.snapshot()["step"], 1)
        make_remote(self.remote, step=2)
        result = self.snapshot()
        self.assertEqual(result["status"], "VERIFIED_HOST_STAGING")
        self.assertEqual(load_checkpoint(self.local, None)[1]["step"], 2)

    def test_recover_step_two_after_crash_while_local_pointer_remains_on_step_one(self):
        self.assertEqual(self.snapshot()["step"], 1)
        make_remote(self.remote, step=2)
        latest = json.loads((self.remote / "latest.json").read_text("utf8"))
        source = self.remote / latest["version"]
        target = self.local / latest["version"]
        shutil.copytree(source, target)
        self.assertEqual(load_checkpoint(self.local, None)[1]["step"], 1)
        result = self.snapshot()
        self.assertEqual(result["status"], "RECOVERED_HOST_STAGING")
        self.assertEqual(result["step"], 2)
        self.assertEqual(load_checkpoint(self.local, None)[1]["step"], 2)

    def test_supervisor_recognizes_recovered_valid_checkpoint_not_plain_exit_code(self):
        from h65.safe_diagnostic import supervise
        from tests.test_h65_safe_diagnostic import FakeChild
        snapshots = iter([{"status": "RECOVERED_HOST_STAGING", "step": 1},
                          {"status": "ALREADY_VERIFIED", "step": 1}])
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild([None, 0])
            result = supervise("hala-h65", "a"*40, Path(root),
                               interval=30, max_seconds=90, steps="train",
                               hardware=lambda _: "T4",
                               runner=lambda *a, **kw: child,
                               snapshot=lambda *a: next(snapshots),
                               sleep=lambda _: None, clock=lambda: 0.0)
            self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")
            self.assertEqual(result["successful_egress_polls"], 2)
            self.assertFalse(result["private_provider_readback_verified"])


if __name__ == "__main__":
    unittest.main()
