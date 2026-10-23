"""H75 remote launcher and H61 parser integration: synthetic, provider-free."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h62 import remote_h62_launch as launch
from h61 import colab_run

CODE = "6eed2e800645a83871ae742d2ac2c2a899dc706a"
SOURCE = "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe"
ZIP = "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a"


class RemoteAuditContract(unittest.TestCase):
    def test_non_staged_audit_refused(self):
        with self.assertRaisesRegex(RuntimeError, "staged"):
            launch.qualify_h75_audit("", "train", "1", "32")

    def test_unsafe_diagnostic_cases_refused_before_packages(self):
        for sha, steps, count, audit in [
                (ZIP, "train", "4", "32"),
                (ZIP, "evaluate", "1", "32"),
                (ZIP, "train", "1", "100"),
                (ZIP, "train", "1", "bad"),
                (ZIP, "train", "1", "-1"),
                (ZIP, "train", "1", "1.1")]:
            with self.subTest(sha=sha, steps=steps, count=count, audit=audit):
                with self.assertRaises(RuntimeError):
                    launch.qualify_h75_audit(sha, steps, count, audit)

    def test_absent_opt_in_keeps_full_legacy(self):
        self.assertEqual(launch.qualify_h75_audit("", "train", "600", "0"), 0)

    def test_remote_validated_pinned_source_and_flag_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            environment = {
                "COLAB_RELEASE_TAG": "test-free-colab",
                "HALA_FREE_COLAB_RESEARCH": "1",
                "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
                "HALA_PINNED_SOURCE_SHA": CODE,
                "H67_STAGE_SHA256": ZIP,
                "H67_STAGE_SOURCE_SHA": SOURCE,
                "H61_STEPS": "train",
                "H61_MAX_STEPS": "1",
                "H61_PROCESSOR_AUDIT_RECORDS": "32",
                "H62_WORK": str(work),
            }
            done = type("Completed", (), {"returncode": 0})()
            with (mock.patch.dict(os.environ, environment, clear=True),
                  mock.patch.object(Path, "exists", return_value=True),
                  mock.patch.object(launch, "fetch_pinned_source", return_value=CODE),
                  mock.patch.object(launch, "package_versions", return_value={}),
                  mock.patch.object(launch, "ensure_packages", return_value={}),
                  mock.patch("h67.stage.restore_stage", return_value={"status": "RESTORED_TRAIN_STAGE", "sha256": ZIP, "git_sha": SOURCE}) as restore,
                  mock.patch.object(launch.subprocess, "run", return_value=done) as run):
                self.assertEqual(launch.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--processor-audit-records") + 1], "32")
            self.assertEqual(command[command.index("--max-steps") + 1], "1")
            self.assertEqual(restore.call_args.args[3], SOURCE)
            receipt = json.loads((work / "h62-remote-launch-receipt.json").read_text("utf8"))
            self.assertEqual(receipt["repo_head"], CODE)
            self.assertEqual(receipt["processor_audit_records"], 32)
            self.assertEqual(receipt["h67_source_git_sha"], SOURCE)

    def test_rejected_remote_diagnostic_does_not_install_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (mock.patch.dict(os.environ, {
                      "COLAB_RELEASE_TAG": "test",
                      "HALA_FREE_COLAB_RESEARCH": "1",
                      "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
                      "HALA_PINNED_SOURCE_SHA": CODE,
                      "H67_STAGE_SHA256": ZIP,
                      "H67_STAGE_SOURCE_SHA": SOURCE,
                      "H61_STEPS": "train",
                      "H61_MAX_STEPS": "50",
                      "H61_PROCESSOR_AUDIT_RECORDS": "32",
                      "H62_WORK": str(Path(tmp) / "work")}, clear=True),
                  mock.patch.object(Path, "exists", return_value=True),
                  mock.patch.object(launch, "fetch_pinned_source", return_value=CODE),
                  mock.patch.object(launch, "ensure_packages") as install):
                with self.assertRaisesRegex(RuntimeError, "bounded"):
                    launch.main()
            install.assert_not_called()

    def test_direct_h61_command_without_h67_admission_refused(self):
        argv = ["colab_run", "--repo", ".", "--work", "unused-work",
                "--steps", "train", "--max-steps", "1",
                "--processor-audit-records", "32"]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.dict(os.environ, {"H67_STAGE_SHA256": "",
                                               "H67_STAGE_SOURCE_SHA": ""}, clear=False):
                with self.assertRaisesRegex(ValueError, "H67 staged"):
                    colab_run.main()


if __name__ == "__main__":
    unittest.main()
