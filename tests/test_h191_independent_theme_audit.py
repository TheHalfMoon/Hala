"""H191 independent saved private image QA using only synthetic fresh images."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from h174.source_candidates import CANDIDATES
from h191.pickr_three_theme_preaction import SPECS
from h191.owner_private_theme_capture import SCHEMA
from h191.audit_owner_theme_capture import audit_saved_owner_theme_preactions
from test_h191_owner_private_theme_capture import fake_prior, fake_png


def prepared(root: Path):
    old = root / "private-h140"
    old.mkdir()
    (old / "manifest.json").write_text(
        json.dumps({"image_sha256": "f"*64}), encoding="utf-8",
    )
    train = fake_prior(
        root, "Hala-H188-Private", "pickr",
        "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    dev = fake_prior(
        root, "Hala-H190-Private", "interact-dropzones",
        "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    dest = root / "Hala-H191-Private"
    (dest / "images").mkdir(parents=True)
    cases, goals = [], []
    for i, spec in enumerate(SPECS):
        name = f"h191-original-pickr-{i:02d}"
        png = fake_png(i)
        with Image.open(BytesIO(png)) as image:
            rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        rel = f"images/{name}.png"
        (dest / rel).write_bytes(png)
        cases.append({
            "case_id": name, "app": "pickr",
            "source_repo": CANDIDATES["pickr"]["repo"],
            "source_git_sha": CANDIDATES["pickr"]["git_sha"],
            "theme": spec["theme"], "seed_index": spec["seed_index"],
            "target_index": spec["target_index"],
            "language": spec["language"], "instruction": spec["instruction"],
            "image": rel, "image_sha256": sha256(png).hexdigest(),
            "visual_rgba_sha256": rgba,
            "purpose": "QUARANTINED_ORIGINAL_THREE_THEME_TRAIN_RESEARCH_ONLY",
            "train_eligible": False, "dev_eligible": False, "ood_eligible": False,
        })
        goals.append({
            "case_id": name,
            "original_native_target_center": {
                "x": float(110+i), "y": float(220+i),
            },
            "not_for_model_input": True, "train_eligible": False,
        })
    doc = {
        "schema": SCHEMA,
        "status": "H191_TWELVE_PRIVATE_PICKR_PREACTION_QUARANTINE_ONLY",
        "source_original_native_replayed": True,
        "known_old_sha_candidates_excluded": True,
        "previous_train_dev_visual_sha_disjoint": True,
        "complete_prior_source_history_proven": False,
        "complete_prior_rgba_history_proven": False,
        "official_train_dev_dataset_admitted": False,
        "case_count": 12, "cases": cases,
    }
    (dest / "source-prospects-not-admitted.json").write_text(
        json.dumps(doc), encoding="utf-8",
    )
    (dest / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": goals}), encoding="utf-8",
    )
    return dest, train, dev, old


def audit(dest, train, dev):
    return audit_saved_owner_theme_preactions(
        new_private_dir=dest, previous_pickr_private=train,
        previous_dev_private=dev, history_root=dest.parent,
    )


class H191SavedThemeAuditTests(unittest.TestCase):
    def test_independently_rehash_twelve_source_screens_and_private_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, _ = prepared(Path(folder))
            got = audit(dest, train, dev)
            self.assertEqual(got["actual_new_images_rehashed"], 12)
            self.assertEqual(got["actual_native_owner_gold_separately_checked"], 12)
            self.assertEqual(got["old_png_sha_exclusions"], 1)
            self.assertTrue(got["source_train_dev_visual_disjoint"])
            self.assertFalse(got["train_dev_admitted"])
            self.assertFalse(got["historical_visual_history_complete"])
            self.assertFalse(got["ood_accessed"])

    def test_source_pixel_content_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, _ = prepared(Path(folder))
            (dest / "images/h191-original-pickr-00.png").write_bytes(fake_png(31))
            with self.assertRaisesRegex(ValueError, "corruption"):
                audit(dest, train, dev)

    def test_old_png_hash_collision_cannot_be_mislabeled_new(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, history = prepared(Path(folder))
            doc = json.loads(
                (dest / "source-prospects-not-admitted.json").read_text(encoding="utf-8")
            )
            (history / "manifest.json").write_text(json.dumps({
                "image_sha256": doc["cases"][5]["image_sha256"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                audit(dest, train, dev)

    def test_model_input_cannot_gain_owner_private_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, _ = prepared(Path(folder))
            m = dest / "source-prospects-not-admitted.json"
            doc = json.loads(m.read_text(encoding="utf-8"))
            doc["cases"][0]["original_native_target_center"] = {"x": 10, "y": 20}
            m.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity modified"):
                audit(dest, train, dev)

    def test_tampered_dataset_admission_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, _ = prepared(Path(folder))
            m = dest / "source-prospects-not-admitted.json"
            doc = json.loads(m.read_text(encoding="utf-8"))
            doc["official_train_dev_dataset_admitted"] = True
            m.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim changed"):
                audit(dest, train, dev)

    def test_owner_target_outside_source_viewport_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest, train, dev, _ = prepared(Path(folder))
            m = dest / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
            doc = json.loads(m.read_text(encoding="utf-8"))
            doc["rows"][3]["original_native_target_center"]["x"] = 1200.0
            m.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coordinate corrupted"):
                audit(dest, train, dev)


if __name__ == "__main__":
    unittest.main()
