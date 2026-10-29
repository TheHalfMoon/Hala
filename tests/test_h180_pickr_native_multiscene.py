"""H180 synthetic-only regression; actual original browser runs separately."""
from dataclasses import replace
from copy import deepcopy
import unittest

from h180.pickr_native_multiscene import (
    NativeScene, SCENES, validate_specs, verify_native_scenes,
)


def synthetic_rows():
    return (
        NativeScene("pickr-swatch-0", "en", 0, "#000", "#123",
                    "1"*64, "a"*64),
        NativeScene("pickr-swatch-0", "en", 0, "#000", "#123",
                    "1"*64, "a"*64),
        NativeScene("pickr-swatch-1", "ar", 1, "#000", "#345",
                    "2"*64, "b"*64),
        NativeScene("pickr-swatch-1", "ar", 1, "#000", "#345",
                    "2"*64, "b"*64),
    )


class OriginalPickrMultisceneTests(unittest.TestCase):
    def test_exact_bilingual_original_task_specs(self):
        validate_specs()
        self.assertEqual({s["language"] for s in SCENES}, {"en", "ar"})
        self.assertEqual({s["swatch_index"] for s in SCENES}, {0, 1})

    def test_synthetic_schema_pass_cannot_admit_data_or_model(self):
        proof = verify_native_scenes(synthetic_rows())
        self.assertEqual(proof["source_native_semantic_successes"], 4)
        self.assertEqual(proof["distinct_native_rgba_scenes"], 2)
        self.assertFalse(proof["train_dev_ood_admitted"])
        self.assertEqual(proof["model_calls"], 0)
        self.assertEqual(proof["native_rewards_model"], 0)

    def test_missing_replay_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "replay missing"):
            verify_native_scenes(synthetic_rows()[:-1])

    def test_duplicate_rgba_across_task_scene_fails(self):
        rows = list(synthetic_rows())
        rows[2] = replace(rows[2], rgba_sha256=rows[0].rgba_sha256)
        rows[3] = replace(rows[3], rgba_sha256=rows[0].rgba_sha256)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_native_scenes(tuple(rows))

    def test_duplicate_png_across_task_scene_fails(self):
        rows = list(synthetic_rows())
        rows[2] = replace(rows[2], png_sha256=rows[0].png_sha256)
        rows[3] = replace(rows[3], png_sha256=rows[0].png_sha256)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_native_scenes(tuple(rows))

    def test_identical_selected_native_value_across_tasks_fails(self):
        rows = list(synthetic_rows())
        rows[2] = replace(rows[2], after=rows[0].after)
        rows[3] = replace(rows[3], after=rows[0].after)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_native_scenes(tuple(rows))

    def test_repeat_native_value_drift_fails(self):
        rows = list(synthetic_rows())
        rows[1] = replace(rows[1], after="#456")
        with self.assertRaisesRegex(ValueError, "replay differs"):
            verify_native_scenes(tuple(rows))

    def test_repeat_rgba_drift_fails(self):
        rows = list(synthetic_rows())
        rows[1] = replace(rows[1], rgba_sha256="3"*64)
        with self.assertRaisesRegex(ValueError, "replay differs"):
            verify_native_scenes(tuple(rows))

    def test_no_actual_native_state_change_fails(self):
        rows = list(synthetic_rows())
        rows[0] = replace(rows[0], after=rows[0].before)
        with self.assertRaisesRegex(ValueError, "did not change"):
            verify_native_scenes(tuple(rows))

    def test_wrong_language_or_swatches_fails(self):
        for field, value in (("language", "ar"), ("swatch_index", 1)):
            rows = list(synthetic_rows())
            rows[0] = replace(rows[0], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "scene identity changed"
            ):
                verify_native_scenes(tuple(rows))

    def test_malformed_source_hashes_fail(self):
        for field, value in (("rgba_sha256", "BAD"),
                             ("png_sha256", "0"*63),
                             ("png_sha256", "Z"*64)):
            rows = list(synthetic_rows())
            rows[0] = replace(rows[0], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "fingerprint"
            ):
                verify_native_scenes(tuple(rows))

    def test_wrong_type_scene_and_extra_data_fields_fail(self):
        rows = list(synthetic_rows())
        rows[0] = {**rows[0].__dict__, "gold": "PRIVATE"}
        with self.assertRaisesRegex(ValueError, "named original"):
            verify_native_scenes(tuple(rows))

    def test_spec_duplicate_and_missing_bilingual_locale_fail(self):
        specs = tuple(deepcopy(s) for s in SCENES)
        specs[1]["language"] = "en"
        with self.assertRaises(ValueError):
            validate_specs(specs)
        specs = tuple(deepcopy(s) for s in SCENES)
        specs[1]["swatch_index"] = 0
        with self.assertRaises(ValueError):
            validate_specs(specs)


if __name__ == "__main__":
    unittest.main()
