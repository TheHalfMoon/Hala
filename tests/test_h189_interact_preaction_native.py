"""H189 synthetic metadata-mutation tests; real native original app via CI."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h189.interact_preaction_native import (
    PreactionDropProof, SPECS, validate_specification, verify_preaction_replay,
)


def synthetic_receipts() -> tuple[PreactionDropProof, ...]:
    result = []
    for index, spec in enumerate(SPECS):
        for _ in range(2):
            result.append(PreactionDropProof(
                id=spec["id"],
                language=spec["language"],
                drag=spec["drag"],
                drop=spec["drop"],
                seed_drop=spec["seed_drop"],
                source_seed_native_reward="Dropped",
                source_target_before="Drag me…",
                source_target_after="Dropped",
                unrelated_drag_untouched=True,
                pre_png_sha256=("1" if index == 0 else "2")*64,
                pre_rgba_sha256=("a" if index == 0 else "b")*64,
                native_target_start=(100.0 + index, 200.0),
                native_target_end=(400.0 + index, 500.0),
            ))
    return tuple(result)


class OriginalInteractPreactionTests(unittest.TestCase):
    def test_seed_rules_are_original_unmodified_source_pins(self):
        validate_specification()
        self.assertEqual({s["seed_drag"] for s in SPECS}, {"drag3"})
        self.assertEqual({s["seed_drop"] for s in SPECS}, {"drop3", "drop1"})
        self.assertEqual({s["language"] for s in SPECS}, {"en", "ar"})
        self.assertEqual({s["drag"] for s in SPECS}, {"drag1", "drag2"})
        for spec in SPECS:
            self.assertNotEqual(spec["seed_drop"], spec["drop"])

    def test_receipt_never_promotes_dev_or_old_ood_or_model(self):
        result = verify_preaction_replay(synthetic_receipts())
        self.assertEqual(result["original_preaction_scenes"], 2)
        self.assertEqual(result["native_seed_and_target_successes"], 8)
        self.assertEqual(result["distinct_preaction_decoded_rgba"], 2)
        self.assertEqual(result["dev_dataset_images_admitted"], 0)
        self.assertEqual(result["train_dataset_images_admitted"], 0)
        self.assertEqual(result["ood_images_created"], 0)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["full_historical_visual_provenance"])

    def test_missing_independent_browser_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, "count wrong"):
            verify_preaction_replay(synthetic_receipts()[:-1])

    def test_wrong_drag_drop_seed_language_id_rejected(self):
        for key, value in (
            ("id", "wrong-source"), ("language", "fr"),
            ("drag", "drag4"), ("drop", "drop3"),
            ("seed_drop", "drop2"),
        ):
            with self.subTest(key=key):
                rows = list(synthetic_receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "identity drift"):
                    verify_preaction_replay(tuple(rows))

    def test_missing_real_seed_or_target_reward_rejected(self):
        for field, value in (
            ("source_seed_native_reward", "Drag me"),
            ("source_target_before", "Dropped"),
            ("source_target_after", "I'm in"),
            ("unrelated_drag_untouched", False),
        ):
            with self.subTest(field=field):
                rows = list(synthetic_receipts())
                rows[0] = replace(rows[0], **{field: value})
                with self.assertRaisesRegex(ValueError, "success missing"):
                    verify_preaction_replay(tuple(rows))

    def test_png_or_rgba_independent_replay_drift_rejected(self):
        for field, value in (
            ("pre_png_sha256", "f"*64),
            ("pre_rgba_sha256", "e"*64),
            ("native_target_start", (112.0, 200.0)),
        ):
            with self.subTest(field=field):
                rows = list(synthetic_receipts())
                rows[1] = replace(rows[1], **{field: value})
                with self.assertRaisesRegex(ValueError, "geometry drift"):
                    verify_preaction_replay(tuple(rows))

    def test_different_cases_identical_preaction_png_or_rgba_rejected(self):
        for field, value in (
            ("pre_png_sha256", "1"*64),
            ("pre_rgba_sha256", "a"*64),
        ):
            with self.subTest(field=field):
                rows = list(synthetic_receipts())
                for i in (2, 3):
                    rows[i] = replace(rows[i], **{field: value})
                with self.assertRaisesRegex(ValueError, "different preaction"):
                    verify_preaction_replay(tuple(rows))

    def test_model_gold_payload_not_allowed_in_strict_source_receipt(self):
        rows = list(synthetic_receipts())
        rows[0] = {**rows[0].__dict__, "private_gold_click": (10, 10)}
        with self.assertRaisesRegex(ValueError, "strict original source"):
            verify_preaction_replay(tuple(rows))

    def test_bad_visual_sha_and_untrusted_geometry_rejected(self):
        for field, value, error in (
            ("pre_png_sha256", "x"*64, "visual SHA"),
            ("pre_rgba_sha256", "abc", "visual SHA"),
            ("native_target_start", (-1.0, 200.0), "geometry invalid"),
            ("native_target_end", (float("nan"), 500.0), "geometry invalid"),
        ):
            with self.subTest(field=field):
                rows = list(synthetic_receipts())
                rows[0] = replace(rows[0], **{field: value})
                with self.assertRaisesRegex(ValueError, error):
                    verify_preaction_replay(tuple(rows))

    def test_known_old_png_or_rgba_sha_blocks_dev_promotion(self):
        for key, value in (
            ("image_sha256", "1"*64),
            ("visual_rgba_sha256", "b"*64),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                phase = Path(tmp) / "private-h141"
                phase.mkdir()
                (phase / "manifest.json").write_text(
                    json.dumps({key: value}), encoding="utf-8",
                )
                previous = scan_legacy_manifest_sha_fields([tmp])
                with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                    verify_preaction_replay(synthetic_receipts(), previous)

    def test_new_digest_no_history_complete_or_dev_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase = Path(tmp) / "private-h141"
            phase.mkdir()
            (phase / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8",
            )
            previous = scan_legacy_manifest_sha_fields([tmp])
            result = verify_preaction_replay(synthetic_receipts(), previous)
            self.assertTrue(result["owner_old_sha_candidates_screened"])
            self.assertEqual(result["dev_dataset_images_admitted"], 0)
            self.assertFalse(result["full_historical_visual_provenance"])


if __name__ == "__main__":
    unittest.main()
