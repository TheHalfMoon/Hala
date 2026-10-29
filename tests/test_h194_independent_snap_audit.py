"""H194 independently saved OWNER NEW source raster checks, synthetic only."""
import json
from pathlib import Path
import tempfile
import unittest

from h194.audit_owner_private_snap import audit_owner_new_slider
from test_h194_owner_private_snap_capture import fixture, synthetic_run, fake_png


def audit(root: Path):
    return audit_owner_new_slider(
        private_dir=root / "Hala-H194-Private",
        older_private_root=root, history_root=root,
    )


class IndependentH194SourceAuditTests(unittest.TestCase):
    def test_two_new_persisted_source_png_rgba_sha_and_separate_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            value = audit(root)
            self.assertEqual(value["new_private_preaction_images_rehashed"], 2)
            self.assertEqual(value["private_native_keyboard_reference_records"], 2)
            self.assertEqual(value["previous_new_source_images_compared"], 16)
            self.assertEqual(value["old_png_sha_candidates_compared"], 1)
            self.assertFalse(value["train_dev_admitted"])
            self.assertFalse(value["old_ood_accessed"])
            self.assertFalse(value["full_historical_source_and_visual_provenance"])

    def test_corrupted_saved_source_image_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            path = root / "Hala-H194-Private/images/h194-original-interact-snap-00.png"
            path.write_bytes(fake_png("green"))
            with self.assertRaisesRegex(ValueError, "decoded RGBA changed"):
                audit(root)

    def test_private_gold_cannot_become_model_visible_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            path = root / "Hala-H194-Private/source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][0]["private_original_action"] = {
                "key": "ArrowRight"
            }
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model input role changed"):
                audit(root)

    def test_deceptive_dev_eligible_flag_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            path = root / "Hala-H194-Private/source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][1]["dev_eligible"] = True
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model input role changed"):
                audit(root)

    def test_known_old_metadata_reuse_blocks_independent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            doc = json.loads(
                (root / "Hala-H194-Private/source-prospects-not-admitted.json")
                .read_text(encoding="utf-8")
            )
            (root / "private-h140/manifest.json").write_text(
                json.dumps({"image_sha256": doc["cases"][0]["image_sha256"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                audit(root)

    def test_cross_cohort_private_visual_collision_blocks_saved_new_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            doc = json.loads(
                (root / "Hala-H194-Private/source-prospects-not-admitted.json")
                .read_text(encoding="utf-8")
            )
            path = root / "Hala-H190-Private/source-prospects-not-admitted.json"
            old = json.loads(path.read_text(encoding="utf-8"))
            old["cases"][0]["image_sha256"] = doc["cases"][1]["image_sha256"]
            path.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap prior train/dev"):
                audit(root)

    def test_private_reference_native_goal_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            synthetic_run(root)
            path = root / "Hala-H194-Private/OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
            refs = json.loads(path.read_text(encoding="utf-8"))
            refs["rows"][0]["private_original_action"]["key"] = "Enter"
            path.write_text(json.dumps(refs), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keyboard reference altered"):
                audit(root)


if __name__ == "__main__":
    unittest.main()
