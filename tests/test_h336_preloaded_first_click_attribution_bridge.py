"""H336 invented-only first-click model-wrapper contract tests.

No model weights, GPU, source screenshots, predictions or private receipts.
The browser/source oracle itself was independently qualified in H335.
"""
from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from h336.preloaded_first_click_attribution_bridge import (
    FirstClickBridgeRefused, ROLE, run_preloaded_first_click_source,
    run_preloaded_first_click_cohort, verify_first_click_wrapper,
)
from h335.original_first_click_error_attribution import ROLE as SOURCE_ROLE
from h319.beaconboard_dynamic_research import SCENES


def row(language="ar", category="correct_open_and_focus"):
    return {
        "status": SOURCE_ROLE, "language": language, "category": category,
        "source_public_exposed": True,
        "native_actions_dispatched": int(category != "abstain"),
        "genuine_model_identity_attested": False,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "can_release": False,
    }


def counters(category="correct_open_and_focus"):
    click = category != "abstain"
    return {
        "model_generations_via_wrapped_generate": 1,
        "model_time_token_filters_installed": 1,
        "distinct_original_pngs_observed": 1,
        "focus_clicks": int(click), "type_calls": 0,
        "abstained": not click, "protocol_terminal": not click,
        "protocol_failed_terminal": False,
        "pretrained_model_identity_attested": False,
        "real_owner_private_gpu_run_attested": False,
        "native_source_task_success_attested": False,
        "official_train_dev_admitted": False,
        "blind_ood_proven": False,
        "generalizable_hala_gain_proven": False,
    }


class FakePolicy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def safe_contract_summary(self):
        return counters()


class H336InventedOnly(TestCase):
    def test_valid_click_and_abstain_reconcile_exact_one_generation(self):
        for category in ("correct_open_and_focus", "correct_open_wrong_focus",
                         "wrong_station_open", "no_station_open", "abstain"):
            with self.subTest(category=category):
                self.assertEqual(
                    verify_first_click_wrapper(row(category=category),
                                               counters(category)), row(category=category))

    def test_reject_mismatched_actions_grammar_and_screenshot_count(self):
        for key, value in (
            ("focus_clicks", 0),
            ("type_calls", 1),
            ("model_generations_via_wrapped_generate", 0),
            ("model_time_token_filters_installed", 2),
            ("distinct_original_pngs_observed", 0),
            ("protocol_terminal", True),
            ("abstained", True),
            ("protocol_failed_terminal", True),
        ):
            with self.subTest(key=key):
                forged = counters()
                forged[key] = value
                with self.assertRaises(FirstClickBridgeRefused):
                    verify_first_click_wrapper(row(), forged)

    def test_reject_private_fields_false_attestations_and_bool_counts(self):
        bad = [
            (dict(row(), raw_model_prediction="PRIVATE"), counters()),
            (dict(row(), can_release=True), counters()),
            (dict(row(), native_actions_dispatched=True), counters()),
            (row(), dict(counters(), real_owner_private_gpu_run_attested=True)),
            (row(), dict(counters(), focus_clicks=True)),
            (row(), dict(counters(), raw_point=[11, 12])),
            (dict(row(), language="other"), counters()),
        ]
        for r, c in bad:
            with self.subTest(keys=len(r) + len(c)):
                with self.assertRaises(FirstClickBridgeRefused):
                    verify_first_click_wrapper(r, c)

    def test_control_preserves_three_arg_preloaded_constructor(self):
        with patch("h336.preloaded_first_click_attribution_bridge.PreloadedDynamicVisualPolicy",
                   side_effect=lambda *a, **kw: FakePolicy(*a, **kw)) as ctor, patch(
                   "h336.preloaded_first_click_attribution_bridge.source_first_click",
                   return_value=row(language="en")) as evaluator:
            got = run_preloaded_first_click_source(
                None, Path("unused"), object(), object(), SCENES[0])
            self.assertEqual(got["language"], "en")
            self.assertEqual(len(ctor.call_args.args), 3)
            self.assertEqual(ctor.call_args.kwargs, {})
            self.assertEqual(evaluator.call_args.kwargs["role"], SOURCE_ROLE)
            self.assertIsInstance(evaluator.call_args.args[3], FakePolicy)

    def test_optin_variant_is_explicit_and_bad_variant_refused(self):
        with patch("h336.preloaded_first_click_attribution_bridge.PreloadedDynamicVisualPolicy",
                   side_effect=lambda *a, **kw: FakePolicy(*a, **kw)) as ctor, patch(
                   "h336.preloaded_first_click_attribution_bridge.source_first_click",
                   return_value=row(language="ar")):
            run_preloaded_first_click_source(
                None, Path("unused"), object(), object(), SCENES[6],
                prompt_variant="visual_target_rtl")
            self.assertEqual(ctor.call_args.kwargs,
                             {"prompt_variant": "visual_target_rtl"})
        with self.assertRaises(FirstClickBridgeRefused):
            run_preloaded_first_click_source(
                None, Path("unused"), None, None, SCENES[0],
                prompt_variant="UNKNOWN")

    def test_full_cohort_aggregate_only_no_per_case_output(self):
        def source_case(browser, source, scene, predict, *, role):
            self.assertEqual(role, SOURCE_ROLE)
            return row(language=scene[1], category="correct_open_and_focus")
        with patch("h336.preloaded_first_click_attribution_bridge.checked_source"), patch(
             "h336.preloaded_first_click_attribution_bridge.PreloadedDynamicVisualPolicy",
             side_effect=lambda *a, **kw: FakePolicy(*a, **kw)), patch(
             "h336.preloaded_first_click_attribution_bridge.source_first_click",
             side_effect=source_case) as evaluator:
            result = run_preloaded_first_click_cohort(
                None, Path("unused"), object(), object())
        self.assertEqual(evaluator.call_count, 12)
        self.assertEqual(result["status"], ROLE)
        self.assertEqual(result["wrapped_model_time_generations_claimed"], 12)
        self.assertEqual(result["by_language"]["en"]["categories"]
                         ["correct_open_and_focus"], 6)
        self.assertEqual(result["by_language"]["ar"]["categories"]
                         ["correct_open_and_focus"], 6)
        self.assertFalse(result["genuine_model_identity_independently_attested"])
        self.assertFalse(result["can_release"])
        self.assertNotIn("scene_id", str(result))
        self.assertNotIn("point", str(result))

    def test_one_bad_private_case_refuses_entire_cohort_without_partial_result(self):
        n = 0
        def source_case(browser, source, scene, predict, *, role):
            nonlocal n
            n += 1
            result = row(language=scene[1])
            if n == 7:
                result["raw_point"] = [99, 99]
            return result
        with patch("h336.preloaded_first_click_attribution_bridge.checked_source"), patch(
             "h336.preloaded_first_click_attribution_bridge.PreloadedDynamicVisualPolicy",
             side_effect=lambda *a, **kw: FakePolicy(*a, **kw)), patch(
             "h336.preloaded_first_click_attribution_bridge.source_first_click",
             side_effect=source_case):
            with self.assertRaisesRegex(
                FirstClickBridgeRefused, "H336 FIRST_CLICK_COHORT_REFUSED"):
                run_preloaded_first_click_cohort(
                    None, Path("unused"), object(), object())
        self.assertEqual(n, 7)


if __name__ == "__main__":
    import unittest
    unittest.main()
