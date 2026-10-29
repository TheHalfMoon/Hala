"""H191 new owner-private PRE-action Pickr quarantine safety, synthetic only.

No real owner images/gold, consumed OOD, old model, or weights are opened.
Actual browser source/run must be qualified separately.
"""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from h174.source_candidates import CANDIDATES
from h191.pickr_three_theme_preaction import SPECS
from h191.owner_private_theme_capture import (
    _prior_manifest_source_metadata, _validate_new_private_folder,
    _verified_new_preaction_memory,
)


def fake_png(i: int) -> bytes:
    im = Image.new("RGB", (960, 720), ((i*13)%256, (i*19)%256, (i*29)%256))
    blob = BytesIO()
    im.save(blob, format="PNG")
    return blob.getvalue()


def fake_captures():
    return tuple(
        (
            dict(spec), fake_png(idx),
            {"x": float(150 + idx), "y": float(200 + idx)},
        )
        for idx, spec in enumerate(SPECS) for _ in range(2)
    )


def fake_prior(root: Path, name: str, source: str, status: str) -> Path:
    folder = root / name
    folder.mkdir()
    pin = CANDIDATES[source]
    (folder / "source-prospects-not-admitted.json").write_text(json.dumps({
        "status": status,
        "official_train_dev_dataset_admitted": False,
        "cases": [
            {
                "source_repo": pin["repo"], "source_git_sha": pin["git_sha"],
                "image_sha256": str(i+1)*64,
                "visual_rgba_sha256": ("a" if i == 0 else "b")*64,
                "train_eligible": False, "dev_eligible": False,
                "ood_eligible": False,
            } for i in range(2)
        ],
    }), encoding="utf-8")
    return folder


class OwnerThemeSourceQuarantineTests(unittest.TestCase):
    def test_twelve_unique_owner_private_original_pngs_and_separate_goals(self):
        got = _verified_new_preaction_memory(list(fake_captures()))
        self.assertEqual(len(got), 12)
        self.assertEqual(len({x["png_sha256"] for x in got}), 12)
        self.assertEqual(len({x["rgba_sha256"] for x in got}), 12)
        self.assertEqual({x["spec"]["theme"] for x in got},
                         {"classic", "monolith", "nano"})
        self.assertEqual({x["spec"]["language"] for x in got}, {"ar", "en"})
        self.assertTrue(all(set(row["goal"]) == {"x", "y"} for row in got))

    def test_source_browser_missing_second_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _verified_new_preaction_memory(list(fake_captures())[:-1])

    def test_source_png_or_native_target_replay_mismatch_rejected(self):
        for kind in ("png", "goal"):
            rows = list(fake_captures())
            spec, png, goal = rows[1]
            rows[1] = (
                spec,
                fake_png(100) if kind == "png" else png,
                {"x": 35.0, "y": 200.0} if kind == "goal" else goal,
            )
            with self.subTest(kind=kind), self.assertRaisesRegex(
                ValueError, "pixel or target center changed"
            ):
                _verified_new_preaction_memory(rows)

    def test_offscreen_or_nan_private_reference_gold_rejected(self):
        for invalid in (-1.0, 960.0, float("nan"), "fake"):
            rows = list(fake_captures())
            for i in (0, 1):
                spec, png, _ = rows[i]
                rows[i] = (spec, png, {"x": invalid, "y": 200.0})
            with self.subTest(invalid=str(invalid)), self.assertRaisesRegex(
                ValueError, "target center invalid"
            ):
                _verified_new_preaction_memory(rows)

    def test_identical_preaction_image_across_different_colors_rejected(self):
        rows = list(fake_captures())
        rows[2] = (rows[2][0], rows[0][1], rows[2][2])
        rows[3] = (rows[3][0], rows[0][1], rows[3][2])
        with self.assertRaisesRegex(ValueError, "pixel duplicates"):
            _verified_new_preaction_memory(rows)

    def test_private_folder_must_be_new_non_git_with_exact_name(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            _validate_new_private_folder(folder / "Hala-H191-Private")
            with self.assertRaisesRegex(ValueError, "exact new absolute"):
                _validate_new_private_folder(folder / "public-images")
            with self.assertRaisesRegex(ValueError, "exact new absolute"):
                _validate_new_private_folder(Path("Hala-H191-Private"))
            (folder / "Hala-H191-Private").mkdir()
            with self.assertRaisesRegex(ValueError, "refuses overwrite"):
                _validate_new_private_folder(folder / "Hala-H191-Private")

    def test_source_private_folder_with_git_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "outside Git"):
                _validate_new_private_folder(folder / "Hala-H191-Private")

    def test_prior_train_and_dev_original_source_implementations_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            pickr = fake_prior(
                base, "Hala-H188-Private", "pickr",
                "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            interact = fake_prior(
                base, "Hala-H190-Private", "interact-dropzones",
                "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            p = _prior_manifest_source_metadata(
                pickr, "Hala-H188-Private", "pickr",
                "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            d = _prior_manifest_source_metadata(
                interact, "Hala-H190-Private", "interact-dropzones",
                "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            self.assertTrue(p["pins"].isdisjoint(d["pins"]))
            self.assertEqual(len(p["png"]), 2)
            self.assertEqual(len(d["rgba"]), 2)

    def test_prior_case_eligibility_tamper_blocks_new_source(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            pickr = fake_prior(
                base, "Hala-H188-Private", "pickr",
                "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            file = pickr / "source-prospects-not-admitted.json"
            doc = json.loads(file.read_text(encoding="utf-8"))
            doc["cases"][0]["train_eligible"] = True
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                _prior_manifest_source_metadata(
                    pickr, "Hala-H188-Private", "pickr",
                    "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
                )

    def test_prior_source_wrong_app_git_pin_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            pickr = fake_prior(
                base, "Hala-H188-Private", "pickr",
                "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
            )
            file = pickr / "source-prospects-not-admitted.json"
            doc = json.loads(file.read_text(encoding="utf-8"))
            doc["cases"][1]["source_git_sha"] = "f"*40
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                _prior_manifest_source_metadata(
                    pickr, "Hala-H188-Private", "pickr",
                    "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
                )


if __name__ == "__main__":
    unittest.main()
