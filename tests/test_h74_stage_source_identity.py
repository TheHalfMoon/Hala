"""H74 provenance regression: original TRAIN source identity is NOT executable code identity.

No GPU, provider, network, dataset bytes or model weights are used.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h62.remote_h62_launch import qualify_stage_source
from h62.run_h61_diagnostic import execute
from h65.safe_diagnostic import supervise

ORIGINAL_H71 = "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe"
NEW_CODE = "5e52d2843e822a807fe3c9a33227949686a8e739"
ZIP_SHA = "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a"


class StageSourceIdentity(unittest.TestCase):
    def test_independent_original_and_new_code_shas(self):
        self.assertNotEqual(ORIGINAL_H71, NEW_CODE)
        self.assertEqual(qualify_stage_source(ZIP_SHA, ORIGINAL_H71), ORIGINAL_H71)

    def test_no_staged_source_is_acceptable(self):
        self.assertEqual(qualify_stage_source("", ""), "")

    def test_missing_original_source_refused(self):
        with self.assertRaisesRegex(RuntimeError, "BOTH"):
            qualify_stage_source(ZIP_SHA, "")

    def test_standalone_source_refused(self):
        with self.assertRaisesRegex(RuntimeError, "BOTH"):
            qualify_stage_source("", ORIGINAL_H71)

    def test_invalid_original_refused(self):
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            qualify_stage_source(ZIP_SHA, "../a")

    def test_invalid_archive_refused(self):
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            qualify_stage_source("z" * 64, ORIGINAL_H71)

    def test_execute_passes_separate_env_without_allocating(self):
        with mock.patch("h62.run_h61_diagnostic.run", return_value=(0, "", "")) as run:
            execute("preexisting-t4", NEW_CODE, "train", "slider=1", "1", "1",
                    "{}", 90, ZIP_SHA, ORIGINAL_H71)
        command = run.call_args.args[0]
        self.assertEqual(command.count("H67_STAGE_SHA256=" + ZIP_SHA), 1)
        self.assertEqual(command.count("H67_STAGE_SOURCE_SHA=" + ORIGINAL_H71), 1)
        self.assertEqual(command.count("HALA_PINNED_SOURCE_SHA=" + NEW_CODE), 1)

    def test_execute_blocks_half_pair_without_provider_call(self):
        with mock.patch("h62.run_h61_diagnostic.run") as run:
            with self.assertRaisesRegex(ValueError, "together"):
                execute("session", NEW_CODE, "train", "", "1", "1",
                        "{}", 90, ZIP_SHA, "")
        run.assert_not_called()

    def test_supervisor_blocks_half_pair_before_hardware_call(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch("h65.safe_diagnostic.session_hardware") as hardware:
                with self.assertRaisesRegex(ValueError, "both required"):
                    supervise("session", NEW_CODE, Path(temp),
                              stage_sha256=ZIP_SHA, stage_source_sha="")
            hardware.assert_not_called()

    def test_supervisor_blocks_stale_code_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "40-character"):
                supervise("session", "bad-code", Path(temp),
                          stage_sha256=ZIP_SHA, stage_source_sha=ORIGINAL_H71)


    def test_remote_restores_archived_original_not_new_code_identity(self):
        import json
        import os
        from h62 import remote_h62_launch as launch
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "work"
            environment = {
                "COLAB_RELEASE_TAG": "test-release",
                "HALA_FREE_COLAB_RESEARCH": "1",
                "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
                "HALA_PINNED_SOURCE_SHA": NEW_CODE,
                "H67_STAGE_SHA256": ZIP_SHA,
                "H67_STAGE_SOURCE_SHA": ORIGINAL_H71,
                "H61_STEPS": "train",
                "H62_WORK": str(work),
            }
            fake_completed = type("Done", (), {"returncode": 0})()
            with (mock.patch.dict(os.environ, environment, clear=True),
                  mock.patch.object(Path, "exists", return_value=True),
                  mock.patch.object(launch, "fetch_pinned_source", return_value=NEW_CODE),
                  mock.patch.object(launch, "package_versions", return_value={}),
                  mock.patch.object(launch, "ensure_packages", return_value={}),
                  mock.patch("h67.stage.restore_stage", return_value={"status": "RESTORED_TRAIN_STAGE"}) as restore,
                  mock.patch.object(launch.subprocess, "run", return_value=fake_completed)):
                self.assertEqual(launch.main(), 0)
            args = restore.call_args.args
            self.assertEqual(args[1], ZIP_SHA)
            self.assertEqual(args[3], ORIGINAL_H71)
            receipt = json.loads((work / "h62-remote-launch-receipt.json").read_text("utf8"))
            self.assertEqual(receipt["repo_head"], NEW_CODE)
            self.assertEqual(receipt["h67_source_git_sha"], ORIGINAL_H71)
            self.assertEqual(receipt["h67_zip_sha256"], ZIP_SHA)

    def test_remote_missing_archive_identity_blocks_before_package_installs(self):
        import os
        from h62 import remote_h62_launch as launch
        with tempfile.TemporaryDirectory() as temp:
            with (mock.patch.dict(os.environ, {
                        "COLAB_RELEASE_TAG": "test-release",
                        "HALA_FREE_COLAB_RESEARCH": "1",
                        "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
                        "HALA_PINNED_SOURCE_SHA": NEW_CODE,
                        "H67_STAGE_SHA256": ZIP_SHA,
                        "H62_WORK": str(Path(temp)/"work"),
                    }, clear=True),
                  mock.patch.object(Path, "exists", return_value=True),
                  mock.patch.object(launch, "fetch_pinned_source", return_value=NEW_CODE),
                  mock.patch.object(launch, "ensure_packages") as install):
                with self.assertRaisesRegex(RuntimeError, "BOTH"):
                    launch.main()
            install.assert_not_called()

if __name__ == "__main__":
    unittest.main()
