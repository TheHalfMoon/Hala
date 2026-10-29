"""H191 synthetic strict source replay and history firewall mutation tests."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h191.pickr_three_theme_preaction import (
    OriginalThemePreaction, SPECS, THEMES, SEEDS, validate_specs,
    validate_replay,
)


def synthetic():
    return tuple(
        OriginalThemePreaction(
            id=spec["id"], theme=spec["theme"],
            seed_index=spec["seed_index"],
            target_index=spec["target_index"],
            language=spec["language"],
            before_native_color="rgb-before",
            after_native_color="rgb-after",
            original_target_center=(float(100 + i), float(150 + i)),
            preaction_png_sha256=f"{i+1:064x}",
            preaction_rgba_sha256=f"{i+100:064x}",
        )
        for i, spec in enumerate(SPECS) for _ in range(2)
    )


class H191OriginalThreeThemeTests(unittest.TestCase):
    def test_all_three_original_pinned_theme_names_and_four_native_seeds(self):
        validate_specs()
        self.assertEqual(THEMES, ("classic", "monolith", "nano"))
        self.assertEqual(SEEDS, (2, 3, 4, 5))
        self.assertEqual(len(SPECS), 12)
        self.assertEqual({row["language"] for row in SPECS}, {"ar", "en"})
        self.assertTrue(all(x["seed_index"] != x["target_index"] for x in SPECS))

    def test_synthetic_receipts_never_admit_any_data_or_train_model(self):
        got = validate_replay(synthetic())
        self.assertEqual(got["unique_original_preaction_scenes"], 12)
        self.assertEqual(got["original_native_seed_and_target_actions"], 48)
        self.assertEqual(got["distinct_png_preaction"], 12)
        self.assertEqual(got["distinct_rgba_preaction"], 12)
        self.assertEqual(got["official_train_cases_admitted"], 0)
        self.assertFalse(got["old_ood_accessed"])
        self.assertFalse(got["historical_source_complete"])
        self.assertEqual(got["model_calls"], 0)

    def test_missing_second_source_context_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing fresh"):
            validate_replay(synthetic()[:-1])

    def test_theme_seed_language_or_target_change_rejected(self):
        for key, value in (
            ("theme", "not-original"),
            ("seed_index", 6),
            ("target_index", 1),
            ("language", "fr"),
            ("id", "some-copied-gui"),
        ):
            with self.subTest(key=key):
                rows = list(synthetic())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    validate_replay(tuple(rows))

    def test_native_color_not_changed_rejected(self):
        rows = list(synthetic())
        rows[0] = replace(rows[0], after_native_color="rgb-before")
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_replay(tuple(rows))

    def test_png_or_rgba_replay_drift_rejected(self):
        for key, value in (
            ("preaction_png_sha256", "f"*64),
            ("preaction_rgba_sha256", "e"*64),
            ("original_target_center", (200.0, 200.0)),
        ):
            with self.subTest(key=key):
                rows = list(synthetic())
                rows[1] = replace(rows[1], **{key: value})
                with self.assertRaisesRegex(ValueError, "drift"):
                    validate_replay(tuple(rows))

    def test_preaction_duplicate_pixels_even_across_original_themes_rejected(self):
        for key, value in (
            ("preaction_png_sha256", synthetic()[0].preaction_png_sha256),
            ("preaction_rgba_sha256", synthetic()[0].preaction_rgba_sha256),
        ):
            with self.subTest(key=key):
                rows = list(synthetic())
                for i in (2, 3):
                    rows[i] = replace(rows[i], **{key: value})
                with self.assertRaisesRegex(ValueError, "duplicate pixels"):
                    validate_replay(tuple(rows))

    def test_sha_not_hex_and_source_goal_outside_frame_rejected(self):
        for key, value, error in (
            ("preaction_png_sha256", "INVALID", "malformed"),
            ("preaction_rgba_sha256", "F"*64, "malformed"),
            ("original_target_center", (-1.0, 30.0), "geometry invalid"),
        ):
            with self.subTest(key=key):
                rows = list(synthetic())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, error):
                    validate_replay(tuple(rows))

    def test_conservative_old_png_or_rgba_visual_reuse_fails_closed(self):
        for field, sample in (
            ("image_sha256", synthetic()[0].preaction_png_sha256),
            ("visual_rgba_sha256", synthetic()[4].preaction_rgba_sha256),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                phase = Path(root) / "private-h140"
                phase.mkdir()
                (phase / "manifest.json").write_text(
                    json.dumps({field: sample}), encoding="utf-8",
                )
                hist = scan_legacy_manifest_sha_fields([root])
                with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                    validate_replay(synthetic(), hist)

    def test_unique_new_hash_does_not_claim_history_complete_or_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h140"
            phase.mkdir()
            (phase / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8",
            )
            old = scan_legacy_manifest_sha_fields([root])
            got = validate_replay(synthetic(), old)
            self.assertTrue(got["owner_old_image_sha_screened"])
            self.assertFalse(got["old_decoded_rgba_history_complete"])
            self.assertEqual(got["official_train_cases_admitted"], 0)


if __name__ == "__main__":
    unittest.main()
