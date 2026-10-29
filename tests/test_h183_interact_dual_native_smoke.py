"""H183 only synthetic source metadata tests; real native GUI runs separately."""
from copy import deepcopy
from dataclasses import replace
import unittest

from h183.interact_dual_native_smoke import (
    NativeDropScene, SCENES, validate_scenes, verify_native_replay,
)


def synthetic_rows() -> tuple[NativeDropScene, ...]:
    return (
        NativeDropScene("interact-drag1-to-drop1", "en", "drag1", "drop1",
                        "Drag me…", "Dropped", True, "1"*64, "a"*64),
        NativeDropScene("interact-drag1-to-drop1", "en", "drag1", "drop1",
                        "Drag me…", "Dropped", True, "1"*64, "a"*64),
        NativeDropScene("interact-drag2-to-drop2", "ar", "drag2", "drop2",
                        "Drag me…", "Dropped", True, "2"*64, "b"*64),
        NativeDropScene("interact-drag2-to-drop2", "ar", "drag2", "drop2",
                        "Drag me…", "Dropped", True, "2"*64, "b"*64),
    )


class OriginalInteractDualNativeTests(unittest.TestCase):
    def test_source_native_specs_exact_original_upstream_accept_rules(self):
        validate_scenes()
        self.assertEqual({row["drag"] for row in SCENES},
                         {"drag1", "drag2"})
        self.assertEqual({row["drop"] for row in SCENES},
                         {"drop1", "drop2"})
        self.assertEqual({row["language"] for row in SCENES}, {"en", "ar"})

    def test_synthetic_receipt_not_dataset_admission(self):
        proof = verify_native_replay(synthetic_rows())
        self.assertEqual(proof["native_source_drag_successes"], 4)
        self.assertEqual(proof["distinct_original_rgba_scenes"], 2)
        self.assertFalse(proof["train_dev_ood_data_admitted"])
        self.assertEqual(proof["new_ood_screenshots"], 0)
        self.assertEqual(proof["model_calls"], 0)

    def test_missing_native_drag_or_repeat_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing original"):
            verify_native_replay(synthetic_rows()[:-1])

    def test_source_identity_wrong_drag_or_drop_rejected(self):
        for field, value in (("drag", "drag3"), ("drop", "drop3"),
                             ("language", "xx"), ("id", "fake")):
            with self.subTest(field=field):
                rows = list(synthetic_rows())
                rows[0] = replace(rows[0], **{field: value})
                with self.assertRaises(ValueError):
                    verify_native_replay(tuple(rows))

    def test_native_drop_did_not_happen_rejected(self):
        for field, value in (("before", "Dropped"),
                             ("after", "Drag me…"),
                             ("unrelated_drag_unchanged", False)):
            with self.subTest(field=field):
                rows = list(synthetic_rows())
                rows[0] = replace(rows[0], **{field: value})
                with self.assertRaisesRegex(ValueError, "original native drop"):
                    verify_native_replay(tuple(rows))

    def test_same_png_different_semantic_source_rejected(self):
        rows = list(synthetic_rows())
        for i in (2, 3):
            rows[i] = replace(rows[i], png_sha256=rows[0].png_sha256)
        with self.assertRaisesRegex(ValueError, "identical visual"):
            verify_native_replay(tuple(rows))

    def test_same_rgba_different_semantic_source_rejected(self):
        rows = list(synthetic_rows())
        for i in (2, 3):
            rows[i] = replace(rows[i], rgba_sha256=rows[0].rgba_sha256)
        with self.assertRaisesRegex(ValueError, "identical visual"):
            verify_native_replay(tuple(rows))

    def test_replay_png_drift_rejected(self):
        rows = list(synthetic_rows())
        rows[1] = replace(rows[1], png_sha256="c"*64)
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            verify_native_replay(tuple(rows))

    def test_replay_rgba_drift_rejected(self):
        rows = list(synthetic_rows())
        rows[1] = replace(rows[1], rgba_sha256="d"*64)
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            verify_native_replay(tuple(rows))

    def test_nonhex_and_invalid_screenshot_digests_rejected(self):
        for field, value in (
            ("rgba_sha256", "X"*64), ("png_sha256", "A"*64),
            ("png_sha256", "abc"),
        ):
            with self.subTest(field=field):
                rows = list(synthetic_rows())
                rows[0] = replace(rows[0], **{field: value})
                with self.assertRaisesRegex(ValueError, "hash malformed"):
                    verify_native_replay(tuple(rows))

    def test_no_private_coordinates_or_gold_in_receipts(self):
        rows = list(synthetic_rows())
        rows[0] = {**rows[0].__dict__, "gold": [500, 400]}
        with self.assertRaisesRegex(ValueError, "strict native drop"):
            verify_native_replay(tuple(rows))

    def test_duplicate_or_swapped_source_task_spec_rejected(self):
        specs = tuple(deepcopy(row) for row in SCENES)
        specs[1]["drop"] = "drop1"
        with self.assertRaisesRegex(ValueError, "pinned original"):
            validate_scenes(specs)
        specs = (deepcopy(SCENES[1]), deepcopy(SCENES[0]))
        with self.assertRaisesRegex(ValueError, "pinned original"):
            validate_scenes(specs)

    def test_wrong_schema_and_extra_fields_in_spec_rejected(self):
        specs = tuple(deepcopy(row) for row in SCENES)
        specs[0]["owner_prediction"] = "SECRET"
        with self.assertRaisesRegex(ValueError, "exact metadata"):
            validate_scenes(specs)


if __name__ == "__main__":
    unittest.main()
