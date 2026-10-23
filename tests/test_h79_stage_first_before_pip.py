"""H79: TRAIN archive admission before pinning packages (provider/GPU-free tests)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h62 import remote_h62_launch as launch

CODE = "623c698c8cfcfc6ef2890bb0061cc513ff510a3a"
ORIGINAL = "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe"
ZIP = "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a"


class StageFirst(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.work = Path(t.name) / "work"
        self.sequence = []

    def environment(self, **overrides):
        base = {
            "COLAB_RELEASE_TAG": "test-synthetic-runtime",
            "HALA_FREE_COLAB_RESEARCH": "1",
            "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
            "HALA_PINNED_SOURCE_SHA": CODE,
            "H67_STAGE_SHA256": ZIP,
            "H67_STAGE_SOURCE_SHA": ORIGINAL,
            "H61_STEPS": "train",
            "H61_MAX_STEPS": "1",
            "H62_WORK": str(self.work),
        }
        base.update(overrides)
        return base

    def _invoke(self, restore, *, env=None):
        done = type("Completed", (), {"returncode": 0})()
        def fake_packages():
            self.sequence.append("pip")
            return {"torch": "2.11.0", "transformers": "5.16.1"}
        def fake_restore(*args, **kwargs):
            self.sequence.append("restore")
            return restore(*args, **kwargs)
        with (mock.patch.dict(os.environ, env or self.environment(), clear=True),
              mock.patch.object(Path, "exists", return_value=True),
              mock.patch.object(launch, "fetch_pinned_source", return_value=CODE),
              mock.patch.object(launch, "package_versions", return_value={}),
              mock.patch.object(launch, "ensure_packages", side_effect=fake_packages),
              mock.patch("h67.stage.restore_stage", side_effect=fake_restore) as stage,
              mock.patch.object(launch.subprocess, "run", return_value=done)):
            status = launch.main()
        receipt = json.loads((self.work / "h62-remote-launch-receipt.json").read_text("utf8"))
        return status, stage, receipt

    def test_stage_authenticates_before_packages_and_remote_training(self):
        def admit(path, sha, target, original, **kwargs):
            self.assertEqual(path, Path("/content/hala-h67-stage.zip"))
            self.assertEqual(sha, ZIP)
            self.assertEqual(original, ORIGINAL)
            self.assertTrue(kwargs["rederive"])
            return {"status": "RESTORED_TRAIN_STAGE", "sha256": ZIP,
                    "git_sha": ORIGINAL, "dataset_sha256": "f"*64}
        code, stage, receipt = self._invoke(admit)
        self.assertEqual(code, 0)
        self.assertEqual(self.sequence, ["restore", "pip"])
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(receipt["h67_stage"]["sha256"], ZIP)
        self.assertEqual(receipt["status"], "completed")
        self.assertIn("packages_after_install", receipt)

    def test_missing_uploaded_stage_refused_before_pin(self):
        def absent(*a, **kw):
            raise ValueError("stage archive differs from operator SHA-256")
        with self.assertRaisesRegex(ValueError, "stage archive"):
            self._invoke(absent)
        self.assertEqual(self.sequence, ["restore"])
        receipt = json.loads((self.work / "h62-remote-launch-receipt.json").read_text("utf8"))
        self.assertNotIn("packages_after_install", receipt)
        self.assertNotEqual(receipt.get("status"), "completed")

    def test_corrupt_zip_or_train_rederive_failure_never_reaches_pip(self):
        for err in ("stage file SHA-256 mismatch", "derived TRAIN SHA differs"):
            with self.subTest(err=err):
                self.sequence.clear()
                def rejected(*a, **kw):
                    raise ValueError(err)
                with self.assertRaises(ValueError):
                    self._invoke(rejected)
                self.assertEqual(self.sequence, ["restore"])

    def test_wrong_stage_result_digest_fails_closed_before_pinning(self):
        for patch in ({"status": "INVALID"}, {"sha256": "e"*64},
                      {"git_sha": "b"*40}, {"status": None}):
            with self.subTest(patch=patch):
                self.sequence.clear()
                def wrong(*a, **kw):
                    return {"status": "RESTORED_TRAIN_STAGE",
                            "sha256": ZIP, "git_sha": ORIGINAL, **patch}
                with self.assertRaisesRegex(RuntimeError, "H79 authenticated"):
                    self._invoke(wrong)
                self.assertEqual(self.sequence, ["restore"])

    def test_staged_cpu_export_is_forbidden_before_pin_or_model(self):
        with self.assertRaisesRegex(RuntimeError, "CPU-only steps forbidden"):
            self._invoke(lambda *a, **kw: self.fail("unexpected restore"),
                         env=self.environment(H61_STEPS="export,train"))
        self.assertEqual(self.sequence, [])

    def test_non_staged_legacy_path_still_skips_restore(self):
        code, stage, receipt = self._invoke(
            lambda *a, **kw: self.fail("unstaged code must not restore"),
            env=self.environment(H67_STAGE_SHA256="", H67_STAGE_SOURCE_SHA="",
                                 H61_STEPS="export"))
        self.assertEqual(code, 0)
        self.assertEqual(self.sequence, ["pip"])
        self.assertEqual(stage.call_count, 0)
        self.assertIn("packages_after_install", receipt)


if __name__ == "__main__":
    unittest.main()
