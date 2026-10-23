"""H68: do not promote a small successful browser pilot to full TRAIN."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h68.readiness import audit, REQUIRED
from h61.colab_run import step_firstparty
from h61.firstparty_drag import build_scenes


class ReadinessTests(unittest.TestCase):
    def test_requirements_match_full_h61_original_source_counts(self):
        self.assertEqual(sum(REQUIRED.values()), 11000)
        self.assertEqual(set(REQUIRED), {"slider", "sheet", "slide_resize", "text_highlight"})

    def test_empty_workdir_is_never_qualified(self):
        with tempfile.TemporaryDirectory() as temp:
            report = audit(Path(temp), rederive=False)
            self.assertEqual(report["status"], "NOT_READY")
            self.assertEqual(len(report["errors"]), 6)
            self.assertFalse(report["training_executed"])
            self.assertFalse(report["provider_byte_readback_verified"])

    def test_actual_bilingual_authoring_is_balanced_at_2000(self):
        scenes = build_scenes(2000)
        self.assertEqual(len(scenes), 2000)
        self.assertEqual(sum(s["language"] == "ar" for s in scenes), 1000)
        self.assertEqual(sum(s["language"] == "en" for s in scenes), 1000)
        self.assertEqual(set(s["kind"] for s in scenes), set(REQUIRED))

    def test_scene_count_must_be_multiple_of_eight(self):
        with tempfile.TemporaryDirectory() as temp:
            for scenes in (1, 7, 9, 2001):
                with self.subTest(scenes=scenes), self.assertRaises(ValueError):
                    step_firstparty(Path(temp), Path(temp), scenes)

    def test_full_receipt_prevents_accidental_shortened_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "export" / "firstparty"
            target.mkdir(parents=True)
            (target / "receipt.json").write_text(json.dumps({
                "scenes_authored": 8, "scenes_accepted": 8}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                step_firstparty(Path(temp), Path(temp), 2000)

    def test_partial_render_refused_before_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch("h61.firstparty_drag.build",
                            return_value={"scenes_authored": 8,
                                          "scenes_accepted": 7}):
                with self.assertRaisesRegex(RuntimeError, "lost required scenes"):
                    step_firstparty(Path(temp), Path(temp), 8)

    def test_cpu_preparation_receipt_must_not_claim_colab_or_metered_units(self):
        import os
        import sys
        from h61.colab_run import main
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "cpu"
            args = ["h61.colab_run", "--repo", temp, "--work", str(work),
                    "--steps", "firstparty", "--firstparty-scenes", "8"]
            with (mock.patch.object(sys, "argv", args),
                  mock.patch.dict(os.environ, {"COLAB_RELEASE_TAG": ""}),
                  mock.patch("h61.colab_run.step_firstparty",
                             return_value={"scenes_authored": 8,
                                           "scenes_accepted": 8})):
                self.assertEqual(main(), 0)
            receipt = json.loads((work / "run-receipt.json").read_text())
            self.assertEqual(receipt["provider"], "host_cpu_preparation")
            self.assertIsNone(receipt["paid_compute_units_consumed"])
            self.assertEqual(receipt["status"], "completed")

    def test_pilot_cannot_claim_full_gpu_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            firstparty = root / "export" / "firstparty"
            firstparty.mkdir(parents=True)
            scenes = {"schema": "hala.h61.firstparty-bilingual-drag.v1",
                      "scenes_authored": 8, "scenes_accepted": 8,
                      "scenes_rejected": 0, "by_language": {"ar": 4, "en": 4},
                      "labels_sha256": hashlib.sha256(b"").hexdigest()}
            (firstparty / "labels.jsonl").write_bytes(b"")
            (firstparty / "receipt.json").write_text(json.dumps(scenes))
            result = audit(root, rederive=False)
            self.assertEqual(result["status"], "NOT_READY")
            self.assertTrue(any("firstparty" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
