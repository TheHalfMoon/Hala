"""H187 synthetic-only receipt mutations; actual original browser via hosted CI."""
from dataclasses import replace
import tempfile
from pathlib import Path
import json
import unittest

from h187.pickr_preaction_native import (
    PreactionNativeReceipt, SPECS, validate_specs,
    validate_replayed_preaction_scenes,
)
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields


def rows():
    return (
        PreactionNativeReceipt(SPECS[0]["id"], "en", 1, 0,
                               "rgba(0,0,0,1)", "rgba(255,0,0,1)",
                               "1"*64, "a"*64),
        PreactionNativeReceipt(SPECS[0]["id"], "en", 1, 0,
                               "rgba(0,0,0,1)", "rgba(255,0,0,1)",
                               "1"*64, "a"*64),
        PreactionNativeReceipt(SPECS[1]["id"], "ar", 0, 1,
                               "rgba(255,0,0,1)", "rgba(0,0,0,1)",
                               "2"*64, "b"*64),
        PreactionNativeReceipt(SPECS[1]["id"], "ar", 0, 1,
                               "rgba(255,0,0,1)", "rgba(0,0,0,1)",
                               "2"*64, "b"*64),
    )


class OriginalPreactionPickrSourceTests(unittest.TestCase):
    def test_exact_two_bilingual_seeded_original_scenes(self):
        validate_specs()
        self.assertEqual({r["language"] for r in SPECS}, {"en", "ar"})
        self.assertEqual({r["seed_index"] for r in SPECS}, {0, 1})
        for r in SPECS:
            self.assertNotEqual(r["seed_index"], r["target_index"])

    def test_pure_synthetic_receipt_never_claims_train_or_ood_admission(self):
        receipt = validate_replayed_preaction_scenes(rows())
        self.assertEqual(receipt["original_preaction_scenes"], 2)
        self.assertEqual(receipt["original_native_action_successes"], 4)
        self.assertEqual(receipt["distinct_preaction_decoded_rgba"], 2)
        self.assertEqual(receipt["train_dev_examples_admitted"], 0)
        self.assertEqual(receipt["train_dev_images_persisted"], 0)
        self.assertEqual(receipt["new_ood_images_persisted"], 0)
        self.assertEqual(receipt["model_calls"], 0)

    def test_wrong_app_or_language_or_seed_rejected(self):
        for key, value in (("id", "invented-source"),
                           ("language", "fr"),
                           ("seed_index", 0),
                           ("target_index", 1)):
            payload = list(rows())
            payload[0] = replace(payload[0], **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_replayed_preaction_scenes(tuple(payload))

    def test_missing_second_browser_context_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_replayed_preaction_scenes(rows()[:-1])

    def test_native_action_not_successful_rejected(self):
        payload = list(rows())
        payload[0] = replace(
            payload[0],
            after_native_color=payload[0].before_native_color,
        )
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_repeat_png_change_rejected(self):
        payload = list(rows())
        payload[1] = replace(payload[1], before_png_sha256="f"*64)
        with self.assertRaisesRegex(ValueError, "contexts disagree"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_repeat_rgba_change_rejected(self):
        payload = list(rows())
        payload[1] = replace(payload[1], before_rgba_sha256="e"*64)
        with self.assertRaisesRegex(ValueError, "contexts disagree"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_same_preaction_png_across_source_scenes_rejected(self):
        payload = list(rows())
        for i in (2, 3):
            payload[i] = replace(payload[i], before_png_sha256="1"*64)
        with self.assertRaisesRegex(ValueError, "distinct preaction"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_same_preaction_rgba_across_source_scenes_rejected(self):
        payload = list(rows())
        for i in (2, 3):
            payload[i] = replace(payload[i], before_rgba_sha256="a"*64)
        with self.assertRaisesRegex(ValueError, "distinct preaction"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_invalid_hash_or_receipt_extra_gold_rejected(self):
        payload = list(rows())
        payload[0] = replace(payload[0], before_rgba_sha256="NOTSHA")
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_replayed_preaction_scenes(tuple(payload))
        payload = list(rows())
        payload[0] = {**payload[0].__dict__, "gold_xy": [1, 2]}
        with self.assertRaisesRegex(ValueError, "strict source-native"):
            validate_replayed_preaction_scenes(tuple(payload))

    def test_old_realistic_png_sha_collision_blocks_no_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            phase = root / "private-h141"
            phase.mkdir()
            (phase / "manifest.json").write_text(json.dumps({
                "cases": [{"at_rest_sha256": "1"*64}]
            }), encoding="utf8")
            old = scan_legacy_manifest_sha_fields([root])
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                validate_replayed_preaction_scenes(rows(), old)

    def test_old_realistic_rgba_sha_collision_blocks_no_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            phase = root / "private-h141"
            phase.mkdir()
            (phase / "manifest.json").write_text(json.dumps({
                "cases": [{"visual_rgba_sha256": "b"*64}]
            }), encoding="utf8")
            old = scan_legacy_manifest_sha_fields([root])
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                validate_replayed_preaction_scenes(rows(), old)

    def test_metadata_only_old_registry_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as folder:
            phase = Path(folder) / "private-h140"
            phase.mkdir()
            (phase / "manifest.json").write_text(json.dumps({
                "cases": [{"image_sha256": "f"*64}]
            }), encoding="utf8")
            old = scan_legacy_manifest_sha_fields([folder])
            got = validate_replayed_preaction_scenes(rows(), old)
            self.assertTrue(got["known_owner_history_shas_screened"])
            self.assertFalse(got["historical_source_and_visual_completeness_proven"])
            self.assertFalse(got["model_training_approved"])


if __name__ == "__main__":
    unittest.main()
