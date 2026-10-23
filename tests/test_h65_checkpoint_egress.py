"""H65 checkpoint recovery: synthetic local artifacts, zero GPU/provider access."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h65.checkpoint_egress import REMOTE_TRAINING_ROOT, snapshot_once
from training.checkpoints import publish_checkpoint, load_checkpoint


class FakeColab:
    def __init__(self, source: Path):
        self.source = source
        self.calls = []
        self.omit = set()
        self.mutate = set()

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        assert command[:4] == ["colab", "download", "-s", "hala-h65"]
        remote = command[4]
        assert remote.startswith(REMOTE_TRAINING_ROOT + "/")
        relative = remote[len(REMOTE_TRAINING_ROOT) + 1:]
        target = Path(command[5])
        source_file = self.source / relative
        if relative in self.omit or not source_file.is_file():
            return subprocess.CompletedProcess(command, 1, "", "not found")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        if relative in self.mutate:
            target.write_bytes(target.read_bytes() + b"tampered")
        return subprocess.CompletedProcess(command, 0, "downloaded", "")


def make_remote(path: Path, step=1):
    state = {
        "config_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "step": step,
        "max_steps": 10,
        "metrics": [{"step": i, "loss": 1.0, "seconds": 0.2}
                    for i in range(1, step + 1)],
        "changed_parameter_tensors": 1,
        "peak_vram_bytes": 1024,
        "runtime": {},
    }
    def write(root):
        (root / "adapter").mkdir()
        (root / "adapter" / "adapter_model.safetensors").write_bytes(b"mock adapter")
        (root / "resume.pt").write_bytes(b"mock optimizer")
    publish_checkpoint(path, state, write)
    return path


class VerifiedEgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.remote = make_remote(self.root / "remote")
        self.local = self.root / "local"
        self.fake = FakeColab(self.remote)

    def tearDown(self):
        self.temp.cleanup()

    def run_once(self):
        return snapshot_once("hala-h65", self.local, runner=self.fake)

    def test_exact_verified_bundle_is_saved_with_valid_atomic_pointer(self):
        result = self.run_once()
        self.assertEqual(result["status"], "VERIFIED_HOST_STAGING")
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["storage_kind"], "host_local_only_not_provider_readback")
        version, state = load_checkpoint(self.local, None)
        self.assertEqual(state["step"], 1)
        self.assertTrue((version / "resume.pt").is_file())

    def test_restart_reuses_verified_checkpoint_without_redundant_artifacts(self):
        self.run_once()
        self.fake.calls.clear()
        result = self.run_once()
        self.assertEqual(result["status"], "ALREADY_VERIFIED")
        self.assertEqual(len(self.fake.calls), 1)

    def test_incorrect_member_digest_never_publishes_pointer(self):
        pointer = json.loads((self.remote / "latest.json").read_text())
        self.fake.mutate.add(pointer["version"] + "/resume.pt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_once()
        self.assertFalse((self.local / "latest.json").exists())

    def test_missing_member_leaves_no_success_pointer(self):
        pointer = json.loads((self.remote / "latest.json").read_text())
        self.fake.omit.add(pointer["version"] + "/resume.pt")
        with self.assertRaisesRegex(IOError, "incomplete"):
            self.run_once()
        self.assertFalse((self.local / "latest.json").exists())

    def test_missing_remote_latest_cannot_fabricate_progress(self):
        self.fake.omit.add("latest.json")
        self.assertEqual(self.run_once()["status"], "NO_PUBLISHED_CHECKPOINT")
        self.assertFalse((self.local / "latest.json").exists())

    def test_modified_checkpoint_metadata_cannot_be_accepted(self):
        pointer = json.loads((self.remote / "latest.json").read_text())
        self.fake.mutate.add(pointer["version"] + "/checkpoint.json")
        with self.assertRaisesRegex(ValueError, "digest mismatches"):
            self.run_once()

    def test_invalid_remote_pointer_is_refused(self):
        remote_pointer = self.remote / "latest.json"
        raw = json.loads(remote_pointer.read_text())
        raw["version"] = "../../etc/passwd"
        remote_pointer.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "pointer is invalid"):
            self.run_once()

    def test_no_remote_rollback_after_newer_checkpoint_is_retained(self):
        self.run_once()
        make_remote(self.remote, step=2)
        fresh = self.run_once()
        self.assertEqual(fresh["step"], 2)
        # Restore the older remote pointer via a still-retained version.
        older_version = next((self.remote / "checkpoints").glob("step-00000001-*"))
        pointer = json.loads((self.remote / "latest.json").read_text())
        pointer["version"] = "checkpoints/" + older_version.name
        import hashlib
        pointer["checkpoint_sha256"] = hashlib.sha256(
            (older_version / "checkpoint.json").read_bytes()).hexdigest()
        (self.remote / "latest.json").write_text(json.dumps(pointer))
        result = self.run_once()
        self.assertEqual(result["status"], "OLDER_REMOTE_SNAPSHOT")
        _, state = load_checkpoint(self.local, None)
        self.assertEqual(state["step"], 2)

    def test_unknown_session_name_cannot_be_used_in_provider_path(self):
        with self.assertRaisesRegex(ValueError, "unsafe Colab session"):
            snapshot_once("../../foreign", self.local, runner=self.fake)

    def test_success_cannot_be_called_private_provider_publication(self):
        result = self.run_once()
        self.assertNotIn("provider_readback_verified", result)
        self.assertIn("host_local_only", result["storage_kind"])


if __name__ == "__main__":
    unittest.main()
