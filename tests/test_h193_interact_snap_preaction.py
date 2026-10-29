"""H193 synthetic strict metadata mutations; real browser is separate CI."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h193.interact_snap_preaction import (
    OriginalSnapPreaction, SPECS, validate_specs, validate_replay,
)


def receipts() -> tuple[OriginalSnapPreaction, ...]:
    return tuple(
        OriginalSnapPreaction(
            id=spec["id"], language=spec["language"],
            seeded_slider=spec["seed"], target_slider=spec["target"],
            before_target_value=300, after_target_value=301,
            seeded_slider_value=499,
            native_slider_center=(770.0+index, 235.0+index),
            preaction_png_sha256=("1" if index == 0 else "2")*64,
            preaction_rgba_sha256=("a" if index == 0 else "b")*64,
            postaction_rgba_sha256=("c" if index == 0 else "d")*64,
        )
        for index, spec in enumerate(SPECS) for _ in range(2)
    )


class OriginalInteractSnapSliderTests(unittest.TestCase):
    def test_original_two_bilingual_distinct_controls_and_source_role(self):
        validate_specs()
        self.assertEqual({s["target"] for s in SPECS}, {"grid-x","grid-y"})
        self.assertTrue(all(s["seed"] != s["target"] for s in SPECS))
        proof = validate_replay(receipts())
        self.assertEqual(proof["original_visual_source_tasks"], 2)
        self.assertEqual(proof["real_native_seed_and_target_keyboard_actions"], 12)
        self.assertEqual(proof["distinct_preaction_rgba"], 2)
        self.assertEqual(proof["original_source_family_role"],
                         "PROSPECTIVE_DEV_NOT_ADMITTED")
        self.assertEqual(proof["saved_images"], 0)
        self.assertEqual(proof["train_or_dev_cases_admitted"], 0)
        self.assertFalse(proof["old_ood_read"])
        self.assertEqual(proof["model_calls"], 0)

    def test_missing_fresh_context_fails(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_replay(receipts()[:-1])

    def test_wrong_native_control_or_language_cannot_be_forged(self):
        for key, value in (
            ("id", "invented"), ("language", "fr"),
            ("seeded_slider", "offset-x"), ("target_slider", "snap-range")
        ):
            with self.subTest(key=key):
                rows = list(receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "drift"):
                    validate_replay(tuple(rows))

    def test_missing_original_native_slider_value_reward_rejected(self):
        for key, value in (
            ("before_target_value", 299),
            ("after_target_value", 302),
            ("seeded_slider_value", 498),
        ):
            with self.subTest(key=key):
                rows = list(receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "reward invalid"):
                    validate_replay(tuple(rows))

    def test_source_pixel_not_changed_after_action_cannot_pass(self):
        rows = list(receipts())
        rows[0] = replace(rows[0],
                          postaction_rgba_sha256=rows[0].preaction_rgba_sha256)
        with self.assertRaisesRegex(ValueError, "visibly redraw"):
            validate_replay(tuple(rows))

    def test_independent_replay_geometry_png_or_rgba_drift_detected(self):
        for key, value in (
            ("preaction_png_sha256", "e"*64),
            ("preaction_rgba_sha256", "f"*64),
            ("postaction_rgba_sha256", "9"*64),
            ("native_slider_center", (900.0, 250.0)),
        ):
            with self.subTest(key=key):
                rows = list(receipts())
                rows[1] = replace(rows[1], **{key: value})
                with self.assertRaisesRegex(ValueError, "replay"):
                    validate_replay(tuple(rows))

    def test_source_scenes_with_identical_preaction_pixels_rejected(self):
        for key, value in (
            ("preaction_png_sha256", "1"*64),
            ("preaction_rgba_sha256", "a"*64),
        ):
            with self.subTest(key=key):
                rows = list(receipts())
                for i in (2, 3):
                    rows[i] = replace(rows[i], **{key: value})
                with self.assertRaisesRegex(ValueError, "duplicate pixels"):
                    validate_replay(tuple(rows))

    def test_bad_hash_or_invisible_slider_target_rejected(self):
        for key, value, message in (
            ("preaction_png_sha256", "G"*64, "visibly redraw"),
            ("preaction_rgba_sha256", "not a sha", "visibly redraw"),
            ("native_slider_center", (-1.0, 200.0), "viewport"),
        ):
            with self.subTest(key=key):
                rows = list(receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, message):
                    validate_replay(tuple(rows))

    def test_known_old_png_or_rgba_blocks_research_prospect(self):
        for key, value in (
            ("image_sha256", "1"*64),
            ("visual_rgba_sha256", "b"*64),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                phase = Path(folder) / "private-h140"
                phase.mkdir()
                (phase / "manifest.json").write_text(
                    json.dumps({key: value}), encoding="utf-8",
                )
                old = scan_legacy_manifest_sha_fields([folder])
                with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                    validate_replay(receipts(), old)

    def test_no_old_hash_collision_does_not_admit_train_dev_or_ood(self):
        with tempfile.TemporaryDirectory() as folder:
            phase = Path(folder) / "private-h140"
            phase.mkdir()
            (phase / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8",
            )
            old = scan_legacy_manifest_sha_fields([folder])
            p = validate_replay(receipts(), old)
            self.assertTrue(p["owner_old_sha_checked"])
            self.assertEqual(p["train_or_dev_cases_admitted"], 0)
            self.assertEqual(p["saved_images"], 0)
            self.assertFalse(p["complete_prior_raster_history_proven"])


if __name__ == "__main__":
    unittest.main()
