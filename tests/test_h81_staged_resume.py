"""H81 strict replay of an H67 TRAIN archive after Colab interruption.

Real SHA256 and ZIP/member checks on synthetic CPU-only data; zero provider.
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h67 import stage
from h81.staged_resume import admit_stage
from tests.test_h67_cpu_stage import fixture

SOURCE = "a" * 40


class ReusableStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cpu = self.path / "cpu"
        fixture(self.cpu)
        self.zip = self.path / "source.zip"
        self.remote = self.path / "gpu_run"
        with mock.patch("h61.curriculum.load_export", return_value=[{}]):
            self.source = stage.make_stage(self.cpu, self.zip, SOURCE)
        self.addCleanup(mock.patch.stopall)

    def admit(self, **kw):
        with mock.patch("h61.curriculum.load_export", return_value=[{}]) as rederive:
            result = admit_stage(
                self.zip, kw.get("digest", self.source["sha256"]),
                self.remote, kw.get("git_sha", SOURCE), rederive=True)
            rederive.assert_called_once()
            return result

    def first(self):
        initial = self.admit()
        self.assertEqual(initial["status"], "RESTORED_TRAIN_STAGE")
        self.assertNotIn("reused_verified_existing", initial)

    def test_first_then_reuse_without_reextracting_original_source(self):
        self.first()
        (self.remote / "training").mkdir()
        (self.remote / "training" / "latest.json").write_text('{"step":1}', "utf8")
        (self.remote / "training-config.json").write_text('{"max_steps":2}', "utf8")
        (self.remote / "run-receipt.json").write_text('{"status":"failed"}', "utf8")
        with mock.patch.object(stage, "restore_stage", side_effect=AssertionError("re-extracted original")):
            again = self.admit()
        self.assertTrue(again["reused_verified_existing"])
        self.assertEqual(again["sha256"], self.source["sha256"])
        self.assertEqual(again["git_sha"], SOURCE)
        self.assertEqual((self.remote / "training" / "latest.json").read_text("utf8"), '{"step":1}')
        self.assertFalse(again["private_provider_readback_verified"])

    def test_modified_original_screenshot_is_rejected_even_with_existing_receipt(self):
        self.first()
        (self.remote / "export/images/slider/scene.jpg").write_bytes(b"modified image")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.admit()

    def test_changed_train_labels_block_reuse(self):
        self.first()
        (self.remote / stage.CURRICULUM).write_bytes(b'{"id":"tampered"}\n')
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.admit()

    def test_extra_unreviewed_source_member_blocks_reuse(self):
        self.first()
        (self.remote / "export/foreign.bin").write_bytes(b"extra source")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.admit()

    def test_missing_training_media_blocks_reuse(self):
        self.first()
        (self.remote / "export/images/slider/scene.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.admit()

    def test_wrong_original_source_git_identity_blocks_reuse(self):
        self.first()
        with self.assertRaisesRegex(ValueError, "pinned source"):
            self.admit(git_sha="b" * 40)

    def test_wrong_zip_hash_blocks_reuse(self):
        self.first()
        with self.assertRaisesRegex(ValueError, "digest"):
            self.admit(digest="0" * 64)

    def test_original_archive_modified_after_initial_restoration_blocks(self):
        self.first()
        with self.zip.open("ab") as f:
            f.write(b"extra")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.admit()

    def test_foreign_top_level_file_blocks_reuse(self):
        self.first()
        (self.remote / "unexpected-source.txt").write_text("danger", "utf8")
        with self.assertRaisesRegex(ValueError, "untrusted"):
            self.admit()

    def test_regular_file_instead_of_run_dir_does_not_overwrite(self):
        self.remote.write_text("occupied", "utf8")
        with self.assertRaises(FileExistsError):
            self.admit()
        self.assertEqual(self.remote.read_text("utf8"), "occupied")

    def test_incomplete_existing_restore_refused_without_overwriting(self):
        self.remote.mkdir()
        with self.assertRaisesRegex(ValueError, "original TRAIN export directory missing"):
            self.admit()
        self.assertFalse((self.remote / "export").exists())

    def test_existing_symlink_export_is_never_followed(self):
        self.first()
        other = self.path / "outside"
        other.mkdir()
        target = self.remote / "export"
        import shutil
        shutil.rmtree(target)
        try:
            target.symlink_to(other, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("host disallows symlink creation")
        with self.assertRaisesRegex(ValueError, "linked"):
            self.admit()


if __name__ == "__main__":
    unittest.main()
