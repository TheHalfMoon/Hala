"""H331 invented-only preloaded prompt hypothesis tests (NO Qwen run)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h323.dynamic_grammar import STAGE_CLICK, STAGE_TYPE, STAGE_DONE, model_visible_prompt
from h323.preloaded_dynamic_bridge import (
    run_preloaded_dynamic_source_research, DynamicModelBridgeRefused,
)
from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
from h319.beaconboard_dynamic_research import SCENES, ROLE
from h331.click_prompt_candidate import (
    CONTROL, VISUAL_RTL, VARIANTS, FIRST_CLICK_ONLY, build_stage_prompt,
)


class FakeProcessor:
    tokenizer = object()
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("NO_MODEL_INVOCATION")
    def batch_decode(self, *args, **kwargs):
        raise AssertionError("NO_MODEL_INVOCATION")


class FakeModel:
    device = "cpu"
    def generate(self, *args, **kwargs):
        raise AssertionError("NO_MODEL_INVOCATION")
    def eval(self):
        return self


class H331PromptContracts(unittest.TestCase):
    def test_fixed_control_identical_for_both_stages(self):
        task = "Open fictional station B-11, then enter code SIM-21 in two steps."
        for stage in (STAGE_CLICK, STAGE_TYPE):
            self.assertEqual(build_stage_prompt(task, stage, variant=CONTROL),
                             model_visible_prompt(task, stage))

    def test_candidate_changes_first_click_and_not_second_stage(self):
        task = "افتح المحطة الخيالية B-11 ثم اكتب الرمز SIM-21 في الحقل الجديد."
        control = build_stage_prompt(task, STAGE_CLICK, variant=CONTROL)
        candidate = build_stage_prompt(task, STAGE_CLICK, variant=VISUAL_RTL)
        self.assertTrue(candidate.startswith(control))
        self.assertEqual(candidate[len(control):], FIRST_CLICK_ONLY)
        self.assertNotEqual(candidate, control)
        self.assertLessEqual(len(candidate), 2000)
        self.assertEqual(build_stage_prompt(task, STAGE_TYPE, variant=CONTROL),
                         build_stage_prompt(task, STAGE_TYPE, variant=VISUAL_RTL))

    def test_no_source_gold_or_station_specific_leak_in_fixed_delta(self):
        # The only target ID and fictional code are in the USER task; extra
        # instructions must never hardcode source scene answers or button IDs.
        for word in ("DOM", "oracle", "image", "RIGHT", "LEFT", "abstain"):
            self.assertIn(word, FIRST_CLICK_ONLY)
        for forbidden in ("B-11", "B-22", "B-33", "B-44", "SIM-",
                          "pixel_values", "beaconboard", "railkit",
                          "[[", "point":[,):
            self.assertNotIn(str(forbidden), FIRST_CLICK_ONLY.lower())
        self.assertIn("x=0", FIRST_CLICK_ONLY)
        self.assertIn("x=999", FIRST_CLICK_ONLY)

    def test_variant_and_stage_fail_closed(self):
        task = "Open fictional station B-11, then enter SIM-21."
        self.assertEqual(VARIANTS, {CONTROL, VISUAL_RTL})
        for bad in ("", "prompt_inject", "gold", "train", None, True, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_stage_prompt(task, STAGE_CLICK, variant=bad)
        for stage in (STAGE_DONE, "", "CLICK", None):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                build_stage_prompt(task, stage, variant=VISUAL_RTL)

    def test_fake_preloaded_policy_stores_variant_without_generating(self):
        for variant in (CONTROL, VISUAL_RTL):
            policy = PreloadedDynamicVisualPolicy(
                FakeProcessor(), FakeModel(),
                "Open fictional station B-11 and enter SIM-21",
                prompt_variant=variant)
            self.assertEqual(policy.prompt_variant, variant)
            self.assertEqual(policy.safe_contract_summary()
                             ["model_generations_via_wrapped_generate"], 0)
        with self.assertRaises(ValueError):
            PreloadedDynamicVisualPolicy(
                FakeProcessor(), FakeModel(),
                "Open fictional station B-11 and enter SIM-21",
                prompt_variant="unsafe")

    def test_bridge_explicit_optin_only_and_no_model_attestation(self):
        src = {
            "role": ROLE, "actions_observed": 2,
            "native_dispatched_actions": 2,
            "original_four_station_task_success": True,
            "categorical_failure": "none", "source_state_checked": True,
            "model_generations_proven": False,
            "official_train_admitted": False,
            "independent_blind_ood": False,
            "adapted_gain_proven": False, "can_release": False,
        }
        ctr = {
            "model_generations_via_wrapped_generate": 2,
            "model_time_token_filters_installed": 2,
            "distinct_original_pngs_observed": 2,
            "focus_clicks": 1, "type_calls": 1,
            "abstained": False, "protocol_terminal": True,
            "protocol_failed_terminal": False,
            "pretrained_model_identity_attested": False,
            "real_owner_private_gpu_run_attested": False,
            "native_source_task_success_attested": False,
            "official_train_dev_admitted": False,
            "blind_ood_proven": False,
            "generalizable_hala_gain_proven": False,
        }
        class MockPolicy:
            def __init__(self, *args, **kwargs):
                ctor_args.append((len(args), kwargs))
            def safe_contract_summary(self):
                return ctr
        ctor_args = []
        with patch("h323.preloaded_dynamic_bridge.PreloadedDynamicVisualPolicy",
                   MockPolicy), patch(
                   "h323.preloaded_dynamic_bridge.run_original_dynamic_sequence",
                   return_value=src):
            control = run_preloaded_dynamic_source_research(
                object(), object(), object(), object(), SCENES[0])
            candidate = run_preloaded_dynamic_source_research(
                object(), object(), object(), object(), SCENES[0],
                prompt_variant=VISUAL_RTL)
            with self.assertRaises(DynamicModelBridgeRefused):
                run_preloaded_dynamic_source_research(
                    object(), object(), object(), object(), SCENES[0],
                    prompt_variant="gold")
        self.assertEqual(ctor_args, [(3, {}), (3, {"prompt_variant": VISUAL_RTL})])
        for result in (control, candidate):
            self.assertTrue(result["source_original_task_success"])
            self.assertFalse(result["trained_model_gain_proven"])
            self.assertFalse(result["pretrained_model_identity_independently_attested"])
            self.assertFalse(result["can_release"])


if __name__ == "__main__":
    unittest.main()
