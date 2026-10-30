"""H252 invented CPU-only diagnostic tests. Zero private data or model usage."""
import json
import unittest

from h252.native_action_diagnostics import (
    _encoded, diagnose_prediction, invented_cpu_example,
    synthetic_contract_summary,
)


class H252ActionDiagnosticsTests(unittest.TestCase):
    def test_all_five_kinds_invented_exact_dry_run(self):
        result = invented_cpu_example()
        self.assertEqual(result["diagnostic_cases"], 5)
        self.assertEqual(
            result["reference_kind_counts"],
            {"click": 1, "drag": 1, "key_press": 1,
             "scroll": 1, "type_text": 1},
        )
        self.assertEqual(result["valid_action_count"], 5)
        self.assertEqual(result["kind_match_count"], 5)
        self.assertEqual(result["exact_action_count"], 5)
        self.assertEqual(result["spatial_comparison_cases"], 4)
        self.assertEqual(result["spatial_within_12px_count"], 4)
        self.assertEqual(result["scroll_direction_match_count"], 1)
        self.assertIs(result["new_model_gain_proven"], False)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["optimizer_steps"], 0)

    def test_click_distance_not_task_success(self):
        ref = _encoded({"kind": "click", "point": [100, 100]})
        near = diagnose_prediction(
            _encoded({"kind": "click", "point": [108, 108]}), ref)
        far = diagnose_prediction(
            _encoded({"kind": "click", "point": [113, 100]}), ref)
        self.assertTrue(near["near_point"])
        self.assertFalse(near["exact"])
        self.assertFalse(far["near_point"])
        self.assertTrue(far["kind_match"])
        self.assertNotIn("task_success", near)

    def test_drag_both_endpoints_must_be_near(self):
        ref = _encoded({"kind": "drag", "start": [100, 100], "end": [400, 400]})
        good = _encoded({"kind": "drag", "start": [104, 103], "end": [402, 400]})
        bad = _encoded({"kind": "drag", "start": [104, 103], "end": [418, 400]})
        self.assertTrue(diagnose_prediction(good, ref)["near_point"])
        self.assertFalse(diagnose_prediction(bad, ref)["near_point"])

    def test_key_press_near_point_does_not_make_wrong_key_correct(self):
        ref = _encoded({"kind": "key_press", "point": [300, 300], "key": "Enter"})
        pred = _encoded({"kind": "key_press", "point": [300, 300], "key": "Space"})
        row = diagnose_prediction(pred, ref)
        self.assertTrue(row["near_point"])
        self.assertFalse(row["exact"])

    def test_unicode_text_exact_and_no_implicit_normalization(self):
        reference = _encoded({"kind": "type_text", "text": "اكتب الآن"})
        self.assertTrue(diagnose_prediction(reference, reference)["exact"])
        different = _encoded({"kind": "type_text", "text": "اكتب الان"})
        row = diagnose_prediction(different, reference)
        self.assertFalse(row["exact"])
        self.assertTrue(row["kind_match"])
        self.assertIsNone(row["near_point"])

    def test_scroll_direction_and_magnitude_separate(self):
        reference = _encoded({"kind": "scroll", "point": [480, 600], "dy": 480})
        opposite = _encoded({"kind": "scroll", "point": [480, 600], "dy": -480})
        close = _encoded({"kind": "scroll", "point": [480, 600], "dy": 360})
        wrong_delta = _encoded({"kind": "scroll", "point": [480, 600], "dy": 200})
        self.assertFalse(diagnose_prediction(opposite, reference)["direction_match"])
        self.assertTrue(diagnose_prediction(close, reference)["direction_match"])
        self.assertTrue(diagnose_prediction(close, reference)["scroll_delta_near"])
        self.assertFalse(diagnose_prediction(wrong_delta, reference)["scroll_delta_near"])

    def test_abstain_and_malformed_predictions_are_not_valid_actions(self):
        reference = _encoded({"kind": "click", "point": [100, 200]})
        abstain = diagnose_prediction(_encoded({"kind": "abstain"}), reference)
        invalid = diagnose_prediction(
            '{"actions":[{"kind":"click","point":[1,1],"point":[2,2]}]}',
            reference)
        self.assertTrue(abstain["abstained"])
        self.assertTrue(abstain["valid"])
        self.assertFalse(abstain["kind_match"])
        self.assertFalse(invalid["valid"])
        self.assertFalse(invalid["abstained"])
        self.assertFalse(invalid["exact"])

    def test_reference_and_synthetic_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            diagnose_prediction(
                _encoded({"kind": "click", "point": [1, 1]}),
                _encoded({"kind": "abstain"}))
        for bad in ([], ["wrong shape"], [("", "")] * 33, [("{}", 3)]):
            with self.subTest(bad=str(bad)[:25]), self.assertRaises(ValueError):
                synthetic_contract_summary(bad)

    def test_summary_never_contains_gold_text_images_or_coordinates(self):
        secretish_invented = "غرفة مجهولة لا تُنشَر"
        pairs = [
            (_encoded({"kind": "type_text", "text": secretish_invented}),
             _encoded({"kind": "type_text", "text": secretish_invented})),
            ("not valid JSON", _encoded({"kind": "scroll", "point": [321, 589],
                                        "dy": -120})),
        ]
        report = synthetic_contract_summary(pairs)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(secretish_invented, serialized)
        self.assertNotIn("321", serialized)
        self.assertNotIn("589", serialized)
        self.assertEqual(report["exact_action_count"], 1)
        self.assertFalse(report["ood_evaluated"])
        self.assertFalse(report["actual_model_predictions_measured"])
        self.assertFalse(report["actual_browser_task_success_measured"])


if __name__ == "__main__":
    unittest.main()
