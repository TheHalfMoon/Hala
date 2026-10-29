"""H194 synthetic owner-only source quarantine and old/cross-split SHA firewall.

Mocked original browser here is NOT native proof. Real Microsoft Edge
must separately replay the immutable original app with owner SHA gate.
"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from h174.source_candidates import CANDIDATES
from h194.owner_private_snap_capture import (
    PREVIOUS, _private_dest, _prior_visual_sha_only, _replay_captures,
    quarantine_original_slider,
)
from h193.interact_snap_preaction import SPECS


def fake_png(color: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (960, 720), color).save(buffer, format="PNG")
    return buffer.getvalue()


def seed_captures():
    return [
        (dict(spec), fake_png(("red", "blue")[idx]), {
            "action_type": "key_press", "key": "ArrowRight",
            "slider": spec["target"],
            "target_center": [780.0 + idx, 210.0 + idx],
        })
        for idx, spec in enumerate(SPECS) for _ in range(2)
    ]


def fixture(root: Path):
    # Mocked source builder needs a temporary empty loopback served root;
    # this is NOT a real source build or real native GUI proof.
    (root / ".h193-derived-original-snap").mkdir()
    phase = root / "private-h140"
    phase.mkdir()
    (phase / "manifest.json").write_text(
        json.dumps({"image_sha256": "f"*64}), encoding="utf-8"
    )
    for n, (name, status, count, family) in enumerate(PREVIOUS):
        prior = root / name
        prior.mkdir()
        pin = CANDIDATES[family]
        doc = {
            "status": status,
            "official_train_dev_dataset_admitted": False,
            "cases": [
                {
                    "source_repo": pin["repo"],
                    "source_git_sha": pin["git_sha"],
                    "train_eligible": False,
                    "dev_eligible": False,
                    "ood_eligible": False,
                    "image_sha256": f"{(n+1)*100+i:064x}",
                    "visual_rgba_sha256": f"{(n+1)*1000+i:064x}",
                } for i in range(count)
            ],
        }
        (prior / "source-prospects-not-admitted.json").write_text(
            json.dumps(doc), encoding="utf-8",
        )


def native_fake(*_args, capture_preaction, **_kwargs):
    for spec, png, goal in seed_captures():
        capture_preaction(spec, png, goal)
    return {
        "original_visual_source_tasks": 2,
        "real_native_seed_and_target_keyboard_actions": 12,
        "owner_old_sha_checked": True,
        "train_or_dev_cases_admitted": 0,
        "model_calls": 0,
    }


def synthetic_run(root: Path):
    with patch("h194.owner_private_snap_capture._verify_original_built_snap_source"), \
         patch("h194.owner_private_snap_capture.run_original_snap",
               side_effect=native_fake):
        return quarantine_original_slider(
            original_donor=root, original_scratch=root,
            historical_root=root, earlier_private_root=root,
            private_output_dir=root / "Hala-H194-Private",
            browser_channel="chromium",
        )


class H194QuarantineTests(unittest.TestCase):
    def test_real_shaped_two_source_preaction_pngs_and_reference_actions(self):
        r = _replay_captures(seed_captures())
        self.assertEqual(len(r), 2)
        self.assertNotEqual(r[0]["image_sha256"], r[1]["image_sha256"])
        self.assertNotEqual(r[0]["visual_rgba_sha256"], r[1]["visual_rgba_sha256"])
        self.assertEqual({x["spec"]["language"] for x in r}, {"en", "ar"})

    def test_no_already_existing_private_folder_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "Hala-H194-Private").mkdir()
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                synthetic_run(root)

    def test_reject_missing_original_second_context(self):
        with self.assertRaisesRegex(ValueError, "screenshots incomplete"):
            _replay_captures(seed_captures()[:-1])

    def test_real_source_frame_or_native_action_replay_drift_fails(self):
        for k in ("pixels", "goal"):
            values = seed_captures()
            spec, png, goal = values[1]
            values[1] = (
                spec,
                fake_png("green") if k == "pixels" else png,
                {"action_type": "key_press", "key": "ArrowRight",
                 "slider": "grid-x", "target_center": [700.0, 210.0]}
                if k == "goal" else goal,
            )
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "replay changed"):
                _replay_captures(values)

    def test_original_slider_target_and_model_gold_shape_rejected(self):
        values = seed_captures()
        for bad in ({"action_type": "key_press", "key": "Enter",
                     "slider": "grid-x", "target_center": [780.0, 210.0]},
                    {"action_type": "key_press", "key": "ArrowRight",
                     "slider": "grid-x", "target_center": [-1.0, 210.0]},
                    {"action_type": "key_press", "key": "ArrowRight",
                     "slider": "grid-x", "target_center": [780.0, float("nan")]}):
            original = values[0][2]
            values[0] = (values[0][0], values[0][1], bad)
            values[1] = (values[1][0], values[1][1], bad)
            with self.subTest(bad=str(bad)), self.assertRaises(ValueError):
                _replay_captures(values)
            values[0] = (values[0][0], values[0][1], original)
            values[1] = (values[1][0], values[1][1], original)

    def test_previous_source_cohorts_only_sha_metadata_and_exact_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            for name, status, count, family in PREVIOUS:
                previous = _prior_visual_sha_only(root / name, status, count, family)
                self.assertEqual(len(previous["png"]), count)
                self.assertEqual(len(previous["rgba"]), count)

    def test_prior_source_role_tamper_blocks_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / PREVIOUS[1][0] / "source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][0]["dev_eligible"] = True
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "eligibility drift"):
                synthetic_run(root)
            self.assertFalse((root / "Hala-H194-Private").exists())

    def test_old_original_sha_reuse_blocks_private_images_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "private-h140" / "manifest.json").write_text(
                json.dumps({"image_sha256": sha256(fake_png("red")).hexdigest()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                synthetic_run(root)
            self.assertFalse((root / "Hala-H194-Private").exists())

    def test_cross_train_dev_prospect_sha_overlap_blocks_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / PREVIOUS[2][0] / "source-prospects-not-admitted.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["cases"][0]["image_sha256"] = sha256(fake_png("blue")).hexdigest()
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats prior prospective"):
                synthetic_run(root)
            self.assertFalse((root / "Hala-H194-Private").exists())

    def test_synthetic_writer_keeps_real_shaped_images_and_gold_only_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            result = synthetic_run(root)
            self.assertEqual(result["saved_private_pngs"], 2)
            self.assertEqual(result["separate_private_native_actions"], 2)
            self.assertEqual(result["existing_private_source_images_screened"], 16)
            self.assertFalse(result["official_train_dev_admitted"])
            self.assertEqual(result["model_calls"], 0)
            self.assertEqual(result["public_images_uploaded"], 0)
            dest = root / "Hala-H194-Private"
            self.assertEqual(len(list((dest/"images").glob("*.png"))), 2)
            inputs = json.loads(
                (dest/"source-prospects-not-admitted.json").read_text(encoding="utf-8")
            )
            refs = json.loads(
                (dest/"OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").read_text(encoding="utf-8")
            )
            self.assertFalse(inputs["official_train_dev_dataset_admitted"])
            self.assertEqual(len(refs["rows"]), 2)
            self.assertNotIn("private_original_action", str(inputs))
            self.assertTrue(all(row["not_for_model_input"] for row in refs["rows"]))


if __name__ == "__main__":
    unittest.main()
