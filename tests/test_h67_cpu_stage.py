"""H67 CPU-to-T4 stage: deterministic synthetic fixtures, zero GPU/network."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from h67 import stage
from h62 import run_h61_diagnostic as operator
from h65 import safe_diagnostic as supervisor

SHA = "a" * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(work):
    root = work / "export"
    (root / "images" / "slider").mkdir(parents=True)
    media = b"synthetic screenshot bytes; not real training data"
    (root / "images" / "slider" / "scene.jpg").write_bytes(media)
    rows = b'{"id":"synthetic-scene"}\n'
    (root / "h61_drag_curriculum.jsonl").write_bytes(rows)
    manifest = {
        "schema": "hala.training.manifest.v1",
        "split": "train",
        "no_benchmark_eval_source": True,
        "training_executed": False,
        "dataset_sha256": sha(rows),
        "train_rows": 1,
        "media": [{"path": "images/slider/scene.jpg", "sha256": sha(media)}],
    }
    (root / "h61_drag_curriculum_manifest.json").write_text(json.dumps(manifest))
    (root / "slider").mkdir()
    (root / "slider" / "receipt.json").write_text('{"scope":"TRAIN"}')
    (work / "public-dev-baseline-receipt.json").write_text('{"baseline":"measured"}')
    return root


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.work = self.path / "cpu_work"
        fixture(self.work)
        self.bundle = self.path / "handoff.zip"
        self.restore = self.path / "gpu_run"

    def tearDown(self):
        self.tmp.cleanup()

    def create(self):
        # The transport fixture is synthetic. Mock only expensive H61
        # rederivation; the original curriculum validator stays enabled
        # on actual CPU/GPU operation.
        with mock.patch("h61.curriculum.load_export", return_value=[{}]) as validate:
            report = stage.make_stage(self.work, self.bundle, SHA)
            validate.assert_called_once()
        return report

    def unpack(self, digest=None):
        with mock.patch("h61.curriculum.load_export", return_value=[{}]) as validate:
            report = stage.restore_stage(
                self.bundle, digest or stage.digest(self.bundle), self.restore, SHA)
            validate.assert_called_once()
        return report

    def test_round_trip_strict_hashes_and_original_curriculum_validation(self):
        report = self.create()
        restored = self.unpack(report["sha256"])
        self.assertEqual(restored["status"], "RESTORED_TRAIN_STAGE")
        self.assertEqual(restored["dataset_sha256"], report["dataset_sha256"])
        self.assertEqual((self.restore / stage.CURRICULUM).read_bytes(),
                         (self.work / stage.CURRICULUM).read_bytes())
        self.assertFalse(restored["private_provider_readback_verified"])

    def test_same_training_content_produces_reproducible_sha(self):
        report = self.create()
        other = stage.make_stage(self.work, self.path / "second.zip", SHA, rederive=False)
        self.assertEqual(report["sha256"], other["sha256"])

    def test_wrong_sha_rejected_before_creating_target(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "operator SHA"):
            self.unpack("f" * 64)
        self.assertFalse(self.restore.exists())

    def test_wrong_pinned_source_commit_refused(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "pinned source"):
            stage.restore_stage(self.bundle, stage.digest(self.bundle), self.restore, "b"*40,
                                rederive=False)
        self.assertFalse(self.restore.exists())

    def test_repeat_import_refuses_overwrite_even_if_bytes_identical(self):
        self.create()
        self.unpack()
        with self.assertRaises(FileExistsError):
            self.unpack()

    def test_corrupt_image_fails_cpu_stage_validation(self):
        (self.work / "export/images/slider/scene.jpg").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "media mismatch"):
            stage.make_stage(self.work, self.bundle, SHA, rederive=False)
        self.assertFalse(self.bundle.exists())

    def test_missing_source_baseline_fails_closed(self):
        (self.work / stage.BASELINE).unlink()
        with self.assertRaisesRegex(ValueError, "metadata"):
            stage.make_stage(self.work, self.bundle, SHA, rederive=False)

    def test_source_symlink_refused(self):
        (self.work / "export" / "danger").symlink_to(self.work / stage.BASELINE)
        with self.assertRaisesRegex(ValueError, "symlink forbidden"):
            stage.make_stage(self.work, self.bundle, SHA, rederive=False)

    def test_zip_archive_duplicate_member_refused(self):
        self.create()
        with zipfile.ZipFile(self.bundle, "a") as z:
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", UserWarning)
                z.writestr(stage.CURRICULUM, b"evil")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stage.restore_stage(self.bundle, stage.digest(self.bundle), self.restore, SHA,
                                rederive=False)
        self.assertFalse(self.restore.exists())

    def test_zip_traversal_not_extracted(self):
        self.create()
        with zipfile.ZipFile(self.bundle, "a") as z:
            z.writestr("../escape", "evil")
        with self.assertRaisesRegex(ValueError, "member-set mismatch"):
            stage.restore_stage(self.bundle, stage.digest(self.bundle), self.restore, SHA,
                                rederive=False)
        self.assertFalse((self.path / "escape").exists())
        self.assertFalse(self.restore.exists())

    def test_zip_wrong_file_digest_fails_even_with_rehashed_zip(self):
        self.create()
        replaced = self.path / "different.zip"
        with zipfile.ZipFile(self.bundle) as src, zipfile.ZipFile(replaced, "w") as dst:
            for info in src.infolist():
                content = src.read(info)
                if info.filename == stage.CURRICULUM:
                    # Preserve length so inventory size checks pass; the
                    # independent SHA check must reject the changed bytes.
                    content = bytes([content[0] ^ 1]) + content[1:]
                dst.writestr(info, content)
        with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
            stage.restore_stage(replaced, stage.digest(replaced), self.restore, SHA,
                                rederive=False)
        self.assertFalse(self.restore.exists())

    def test_stage_content_is_not_a_training_or_public_release_receipt(self):
        report = self.create()
        self.assertEqual(report["status"], "CPU_STAGE_CREATED")
        self.assertFalse(report["private_provider_readback_verified"])
        self.assertNotIn("gpu_trained", report)

    def test_cpu_step_names_forbidden_in_staged_operator(self):
        with self.assertRaisesRegex(ValueError, "CPU preparation forbidden"):
            operator.main(["--session", "hala-h67", "--pinned-sha", SHA,
                           "--local-dir", str(self.path / "ops"),
                           "--stage-sha256", "1" * 64,
                           "--steps", "export,curriculum,train"])

    def test_stage_hash_is_transmitted_to_colab_exec(self):
        called = []
        def fake_run(command, timeout=None):
            called.append(command)
            return 0, "", ""
        with mock.patch.object(operator, "run", side_effect=fake_run):
            operator.execute("hala-h67", SHA, "train", "slider=1", "3", "1",
                             "{}", 30, "e" * 64)
        args = called[0]
        self.assertIn("H67_STAGE_SHA256=" + "e"*64, args)

    def test_supervisor_passes_only_digest_and_not_cpu_data(self):
        class Child:
            pid = 99
            returncode = 0
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def terminate(self): self.returncode = -15
            def kill(self): self.returncode = -9
        calls = []
        def proc(cmd, **kwargs):
            calls.append(cmd)
            return Child()
        with tempfile.TemporaryDirectory() as tmp:
            result = supervisor.supervise("hala-h67", SHA, Path(tmp),
                steps="train", stage_sha256="f"*64, runner=proc,
                hardware=lambda _: "T4", snapshot=lambda *a: {"status": "ALREADY_VERIFIED"},
                sleep=lambda _: None, clock=lambda: 0.0)
        self.assertIn("--stage-sha256", calls[0])
        self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")

    def test_supervisor_refuses_cpu_source_export_on_gpu(self):
        with self.assertRaisesRegex(ValueError, "CPU preparation forbidden"):
            supervisor.supervise("hala-h67", SHA, self.path / "out",
                                 stage_sha256="f"*64, steps="export,train")


if __name__ == "__main__":
    unittest.main()
