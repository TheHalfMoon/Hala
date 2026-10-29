"""H190 new-screenshot synthetic post-save auditor mutation suite.

No actual private source media and NO prior consumed/heldout image opened.
"""
from hashlib import sha256
from pathlib import Path
from io import BytesIO
import json
import tempfile
import unittest

from PIL import Image

from h174.source_candidates import CANDIDATES
from h189.interact_preaction_native import SPECS
from h190.audit_owner_private_interact_dev import independent_private_dev_audit
from test_h190_owner_private_interact_dev_capture import previous_train


def make_fixture(root: Path):
    history = root / "private-h141"
    history.mkdir()
    (history / "manifest.json").write_text(
        json.dumps({"image_sha256": "f"*64}), encoding="utf-8",
    )
    train = root / "Hala-H188-Private"
    previous_train(train)
    dev = root / "Hala-H190-Private"
    (dev / "images").mkdir(parents=True)
    inputs, gold = [], []
    for index, spec in enumerate(SPECS):
        ident = f"h190-original-interact-{index:02d}"
        image = Image.new("RGB", (960, 720), ("red", "blue")[index])
        buff = BytesIO()
        image.save(buff, format="PNG")
        png = buff.getvalue()
        rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        rel = f"images/{ident}.png"
        (dev / rel).write_bytes(png)
        inputs.append({
            "case_id": ident,
            "app": "interact-dropzones",
            "source_repo": CANDIDATES["interact-dropzones"]["repo"],
            "source_git_sha": CANDIDATES["interact-dropzones"]["git_sha"],
            "language": spec["language"],
            "instruction": spec["instruction"],
            "action_type": "drag", "image": rel,
            "image_sha256": sha256(png).hexdigest(),
            "visual_rgba_sha256": rgba,
            "purpose": "QUARANTINED_PROSPECTIVE_DEV_RESEARCH_ONLY",
            "train_eligible": False, "dev_eligible": False,
            "ood_eligible": False,
        })
        gold.append({
            "case_id": ident,
            "private_original_drag_start": [120.0 + index, 80.0],
            "private_original_drop_target": [550.0 + index, 440.0],
            "not_for_model_input": True, "dev_eligible": False,
        })
    doc = {
        "schema": "hala.h190.owner-private-interact-dev-preaction-quarantine.v1",
        "status": "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
        "source_original_native_replayed": True,
        "known_old_sha_candidates_excluded": True,
        "different_original_source_from_train": True,
        "complete_prior_source_history_proven": False,
        "complete_prior_rgba_history_proven": False,
        "official_train_dev_dataset_admitted": False,
        "case_count": len(inputs), "cases": inputs,
    }
    (dev / "source-prospects-not-admitted.json").write_text(
        json.dumps(doc), encoding="utf-8",
    )
    (dev / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": gold}), encoding="utf-8",
    )
    return dev, train, history


def check(dev, train, root):
    return independent_private_dev_audit(
        private_dir=dev, historical_root=root, previous_train_private=train,
    )


class H190IndependentNewImageAuditTests(unittest.TestCase):
    def test_only_two_genuine_shaped_private_pngs_with_separated_native_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, _ = make_fixture(root)
            result = check(dev, train, root)
            self.assertEqual(result["private_preaction_scenes"], 2)
            self.assertEqual(result["private_reference_drag_goals"], 2)
            self.assertTrue(result["cross_train_visual_sha_disjoint"])
            self.assertEqual(result["compared_old_png_sha_metadata"], 1)
            self.assertFalse(result["train_dev_admitted"])
            self.assertFalse(result["complete_historical_visual_provenance"])
            self.assertFalse(result["old_ood_accessed"])

    def test_changed_saved_original_png_pixels_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, _ = make_fixture(root)
            image = Image.new("RGB", (960, 720), "green")
            image.save(dev / "images/h190-original-interact-00.png", format="PNG")
            with self.assertRaisesRegex(ValueError, "source raster altered"):
                check(dev, train, root)

    def test_owner_gold_in_model_input_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, _ = make_fixture(root)
            path = dev / "source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][0]["private_original_drop_target"] = [1.0, 2.0]
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role/task identity altered"):
                check(dev, train, root)

    def test_admission_flag_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, _ = make_fixture(root)
            path = dev / "source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][0]["dev_eligible"] = True
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role/task identity altered"):
                check(dev, train, root)

    def test_cross_train_private_sha_overlap_blocks_new_dev(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, _ = make_fixture(root)
            new_doc = json.loads(
                (dev / "source-prospects-not-admitted.json").read_text(encoding="utf-8")
            )
            old_path = train / "source-prospects-not-admitted.json"
            old = json.loads(old_path.read_text(encoding="utf-8"))
            old["cases"][0]["image_sha256"] = new_doc["cases"][0]["image_sha256"]
            old_path.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps private Pickr"):
                check(dev, train, root)

    def test_old_historical_sha_metadata_match_blocks_new_dev(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev, train, hist = make_fixture(root)
            new_doc = json.loads(
                (dev / "source-prospects-not-admitted.json").read_text(encoding="utf-8")
            )
            (hist / "manifest.json").write_text(
                json.dumps({"image_sha256": new_doc["cases"][1]["image_sha256"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                check(dev, train, root)


if __name__ == "__main__":
    unittest.main()
