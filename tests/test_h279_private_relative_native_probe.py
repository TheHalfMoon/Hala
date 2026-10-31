"""H279 code-only synthetic checks; no GPU, private media, real model or action."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from h279.private_relative_native_probe import (
    SCENES, MODEL_SHA, RelativePrevalidationProcessor,
    classify_strict_model_text, summarize,
)
from h279.build_private_payload import build, SCRIPT_NAME, META_NAME

ROOT = Path(__file__).resolve().parents[1]


class H279PrivateRelativeNativeSafetyTests(unittest.TestCase):
    @staticmethod
    def row(scene, *, success=False):
        return {
            "scene": scene,
            "real_model_generation_count": 1,
            "real_token_filter_generate_calls": 1,
            "original_screenshot_unchanged": True,
            "format_class": "STRICT_EXACT_VALID",
            "strict_relative_action_valid": True,
            "native_action_kind": "click",
            "native_action_dispatched": True,
            "first_plus_target_hit": success,
            "firstparty_inventory_task_success": success,
            "other_rows_unchanged": True,
        }

    def test_safe_aggregate_only_bounded_boolean_fields(self):
        rows = [self.row(SCENES[0]), self.row(SCENES[1], success=True)]
        report = summarize(rows, {"free_t4": True, "devices": 1})
        self.assertEqual(report["real_pretrained_image_generations"], 2)
        self.assertEqual(report["native_actions_dispatched"], 2)
        self.assertEqual(report["first_plus_target_hits"], 1)
        self.assertEqual(report["actual_firstparty_task_success"], 1)
        self.assertEqual(report["pinned_model_revision"], MODEL_SHA)
        self.assertFalse(report["generalizable_hala_skill_proven"])
        self.assertFalse(report["independent_blind_ood"])
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn('"point":', encoded.lower())
        self.assertNotIn(".png", encoded.lower())
        self.assertNotIn("raw_model_text", encoded.lower())

    def test_reject_unsupported_or_fabricated_success(self):
        baseline = [self.row(x) for x in SCENES]
        invalid = [
            {"scene": SCENES[0]},  # no actual generation
            {**baseline[0], "firstparty_inventory_task_success": True},
            {**baseline[0], "real_model_generation_count": 0},
            {**baseline[0], "native_action_kind": "type_text"},
            {**baseline[0], "model_point": [12, 20]},
            {**baseline[0], "strict_relative_action_valid": False},
            {**baseline[0], "original_screenshot_unchanged": False},
        ]
        for row in invalid:
            with self.subTest(fields=sorted(row)):
                with self.assertRaisesRegex(ValueError, "H279 fail-closed"):
                    summarize([row, baseline[1]], {"free_t4": True, "devices": 1})

    def test_private_single_python_script_and_no_uploaded_media(self):
        source = ROOT / "h279" / "private_relative_native_probe.py"
        with TemporaryDirectory() as temp:
            target = Path(temp) / "payload"
            result = build(source, target)
            meta = json.loads((target / META_NAME).read_text("utf-8"))
            self.assertTrue(meta["is_private"] == "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], [])
            self.assertEqual(meta["model_sources"], [])
            self.assertEqual(meta["kernel_sources"], [])
            self.assertEqual(meta["competition_sources"], [])
            self.assertEqual(result["uploaded_model_or_media"], 0)
            self.assertEqual(sorted(x.name for x in target.iterdir()),
                             sorted([SCRIPT_NAME, META_NAME]))
            script = (target / SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(script.count("from __future__ import annotations"), 1)
            self.assertIn("execute_one(page,strict_native)", script)
            self.assertIn("RelativePrevalidationProcessor(", script)
            self.assertIn("native = self.converter(raw)", script)

    def test_relative_format_classes_stay_closed(self):
        self.assertEqual(
            classify_strict_model_text(
                '{"actions":[{"kind":"click","point":[500,400]}]}',
                r'\{"actions":\[\{"kind":"click","point":\[(0|[1-9][0-9]?|[1-9][0-9][0-9]),(0|[1-9][0-9]?|[1-9][0-9][0-9])\]\}\]\}'),
            "STRICT_EXACT_VALID",
        )
        self.assertEqual(classify_strict_model_text('BAD DATA', 'x'), "NOT_JSON_OR_NONCANONICAL_PREFIX")


    def test_relative_999_y_is_converted_before_native_validation(self):
        from h273.relative_native_bridge import (
            normalize_strict_action, NORMALIZED_REGEX)
        from h199.native_action_sft import parse_actions

        class FakeRelativeOutput:
            def batch_decode(self, *args, **kwargs):
                return ['{"actions":[{"kind":"click","point":[999,999]}]}']

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("not needed in decoding unit test")

        decoder = RelativePrevalidationProcessor(
            FakeRelativeOutput(), normalize_strict_action, NORMALIZED_REGEX)
        native = decoder.batch_decode([[]])[0]
        parsed = parse_actions(native)["actions"][0]
        self.assertEqual(parsed["kind"], "click")
        self.assertEqual(parsed["point"], [959, 719])
        self.assertEqual(decoder.converted_calls, 1)
        self.assertEqual(decoder.format_class, "STRICT_EXACT_VALID")
        self.assertFalse(decoder.format_rejected)

    def test_relative_decoder_never_repairs_invalid_json(self):
        from h273.relative_native_bridge import (
            normalize_strict_action, NORMALIZED_REGEX)

        class FakeInvalid:
            def batch_decode(self, *args, **kwargs):
                return ['{"actions":[{"kind":"click","point":[1000,999]}]}']

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("not needed in decoding unit test")

        decoder = RelativePrevalidationProcessor(
            FakeInvalid(), normalize_strict_action, NORMALIZED_REGEX)
        with self.assertRaisesRegex(ValueError, "normalized model format refused"):
            decoder.batch_decode([[]])
        self.assertTrue(decoder.format_rejected)
        self.assertEqual(decoder.converted_calls, 0)
        self.assertNotEqual(decoder.format_class, "STRICT_EXACT_VALID")



if __name__ == "__main__":
    unittest.main()
