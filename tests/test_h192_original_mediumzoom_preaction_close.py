"""H192 only synthetic source-receipt tests; original browser run separately."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h192.original_mediumzoom_preaction_close import (
    OriginalZoomPreaction, SCENES, validate_original_specs,
    validate_preaction_replay,
)


def original_shaped_receipts():
    return tuple(
        OriginalZoomPreaction(
            id=spec["id"], photo_index=spec["photo_index"],
            language=spec["language"], original_photo_opened=True,
            original_photo_closed=True,
            private_native_close_center=(380.0+index, 275.0),
            preaction_png_sha256=("1" if index == 0 else "2")*64,
            preaction_rgba_sha256=("a" if index == 0 else "b")*64,
        ) for index, spec in enumerate(SCENES) for _ in range(2)
    )


class OriginalMediumZoomCloseTests(unittest.TestCase):
    def test_source_is_research_exposed_unassigned_not_ood(self):
        validate_original_specs()
        self.assertEqual({s["photo_index"] for s in SCENES}, {0, 1})
        self.assertEqual({s["language"] for s in SCENES}, {"en", "ar"})
        result = validate_preaction_replay(original_shaped_receipts())
        self.assertEqual(result["native_open_close_action_successes"], 8)
        self.assertEqual(result["distinct_preaction_decoded_rgba"], 2)
        self.assertEqual(result["source_family_role"], "RESEARCH_EXPOSED_UNASSIGNED")
        self.assertEqual(result["train_dev_examples_admitted"], 0)
        self.assertEqual(result["ood_images_created"], 0)
        self.assertEqual(result["images_persisted"], 0)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["complete_historical_visual_registry_proven"])

    def test_must_have_both_original_independent_fresh_contexts(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_preaction_replay(original_shaped_receipts()[:-1])

    def test_wrong_original_photo_or_language_rejected(self):
        for key, value in (
            ("id", "invented"), ("photo_index", 1), ("language", "fr")
        ):
            with self.subTest(key=key):
                rows = list(original_shaped_receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    validate_preaction_replay(tuple(rows))

    def test_original_zoom_open_and_close_required(self):
        for key in ("original_photo_opened", "original_photo_closed"):
            with self.subTest(key=key):
                rows = list(original_shaped_receipts())
                rows[0] = replace(rows[0], **{key: False})
                with self.assertRaisesRegex(ValueError, "not proven"):
                    validate_preaction_replay(tuple(rows))

    def test_independent_context_png_rgba_or_geometry_drift_rejected(self):
        for key, value in (
            ("preaction_png_sha256", "f"*64),
            ("preaction_rgba_sha256", "e"*64),
            ("private_native_close_center", (11.0, 13.0)),
        ):
            with self.subTest(key=key):
                rows = list(original_shaped_receipts())
                rows[1] = replace(rows[1], **{key: value})
                with self.assertRaisesRegex(ValueError, "differs"):
                    validate_preaction_replay(tuple(rows))

    def test_identical_preaction_photographs_are_not_two_scenes(self):
        for key, value in (
            ("preaction_png_sha256", "1"*64),
            ("preaction_rgba_sha256", "a"*64),
        ):
            with self.subTest(key=key):
                rows = list(original_shaped_receipts())
                for i in (2, 3):
                    rows[i] = replace(rows[i], **{key: value})
                with self.assertRaisesRegex(ValueError, "duplicate pixels"):
                    validate_preaction_replay(tuple(rows))

    def test_bad_sha_and_offscreen_native_target_rejected(self):
        for key, value, err in (
            ("preaction_png_sha256", "G"*64, "malformed"),
            ("preaction_rgba_sha256", "abc", "malformed"),
            ("private_native_close_center", (-1.0, 20.0), "geometry"),
        ):
            with self.subTest(key=key):
                rows = list(original_shaped_receipts())
                rows[0] = replace(rows[0], **{key: value})
                with self.assertRaisesRegex(ValueError, err):
                    validate_preaction_replay(tuple(rows))

    def test_known_old_png_or_rgba_fingerprint_reuse_is_rejected(self):
        for field, value in (
            ("image_sha256", "1"*64),
            ("visual_rgba_sha256", "b"*64),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                phase = Path(folder) / "private-h140"
                phase.mkdir()
                (phase / "manifest.json").write_text(
                    json.dumps({field: value}), encoding="utf-8",
                )
                old = scan_legacy_manifest_sha_fields([folder])
                with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                    validate_preaction_replay(original_shaped_receipts(), old)


if __name__ == "__main__":
    unittest.main()
