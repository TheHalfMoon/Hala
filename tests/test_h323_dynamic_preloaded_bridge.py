"""H323 invented-only contract tests: never claim mocked policy is Qwen."""
import unittest
from unittest.mock import patch

from h323.dynamic_grammar import (
    STAGE_CLICK, STAGE_TYPE, STAGE_DONE, DynamicPolicyState,
    TYPE_REGEX, model_visible_prompt,
)
from h323.preloaded_dynamic_bridge import (
    DynamicModelBridgeRefused, verify_counts,
    run_preloaded_dynamic_source_research,
)
from h319.beaconboard_dynamic_research import ROLE, SCENES


def outcome(n, dispatched, success, reason):
    return {
        "role": ROLE, "actions_observed": n,
        "native_dispatched_actions": dispatched,
        "original_four_station_task_success": success,
        "categorical_failure": reason,
        "source_state_checked": True,
        "model_generations_proven": False,
        "official_train_admitted": False,
        "independent_blind_ood": False,
        "adapted_gain_proven": False,
        "can_release": False,
    }


def counters(n, clicked, typed, abstained, terminal, failed=False):
    return {
        "model_generations_via_wrapped_generate": n,
        "model_time_token_filters_installed": n,
        "distinct_original_pngs_observed": n,
        "focus_clicks": clicked, "type_calls": typed,
        "abstained": abstained, "protocol_terminal": terminal,
        "protocol_failed_terminal": failed,
        "pretrained_model_identity_attested": False,
        "real_owner_private_gpu_run_attested": False,
        "native_source_task_success_attested": False,
        "official_train_dev_admitted": False,
        "blind_ood_proven": False,
        "generalizable_hala_gain_proven": False,
    }


class H323Tests(unittest.TestCase):
    def test_dynamic_click_prompt_mentions_button_not_fieldport_input(self):
        text = model_visible_prompt("Open station B-11 and type SIM-21", STAGE_CLICK)
        self.assertIn("STATION BUTTON", text)
        self.assertNotIn("FIC-", text)
        self.assertNotIn("correct coordinate", text)

    def test_dynamic_type_prompt_describes_newly_revealed_input(self):
        text = model_visible_prompt("Open station B-11 and type SIM-21", STAGE_TYPE)
        self.assertIn("NEW screenshot", text)
        self.assertIn("SIM-NN", text)
        self.assertNotIn("FIC-", text)
        with self.assertRaises(ValueError):
            model_visible_prompt("Open station B-11 and type SIM-21", STAGE_DONE)

    def test_native_strict_SIM_not_FIC_or_multiple_actions(self):
        import re
        self.assertIsNotNone(re.fullmatch(
            TYPE_REGEX, '{"actions":[{"kind":"type_text","text":"SIM-21"}]}'))
        for raw in ('{"actions":[{"kind":"type_text","text":"FIC-21"}]}',
                    '{"actions":[{"kind":"type_text","text":"SIM-2"}]}',
                    '{"actions":[{"kind":"type_text","text":"SIM-21"},{"kind":"abstain"}]}'):
            self.assertIsNone(re.fullmatch(TYPE_REGEX, raw))

    def test_exact_prospective_state_two_turns(self):
        state = DynamicPolicyState()
        click = state.consume_model_text(
            '{"actions":[{"kind":"click","point":[500,500]}]}')
        self.assertIn('"kind":"click"', click)
        self.assertEqual(state.stage, STAGE_TYPE)
        literal = '{"actions":[{"kind":"type_text","text":"SIM-21"}]}'
        self.assertEqual(state.consume_model_text(literal), literal)
        self.assertEqual(state.stage, STAGE_DONE)
        self.assertEqual(state.categorical_state()["generations"], 2)
        with self.assertRaises(ValueError):
            state.consume_model_text(literal)

    def test_fieldport_code_refused_and_no_stage_advance(self):
        state = DynamicPolicyState()
        state.consume_model_text('{"actions":[{"kind":"click","point":[1,2]}]}')
        with self.assertRaises(ValueError):
            state.consume_model_text(
                '{"actions":[{"kind":"type_text","text":"FIC-99"}]}')
        self.assertEqual(state.stage, STAGE_TYPE)
        self.assertEqual(state.generations, 1)

    def test_first_step_abstain_terminal(self):
        state = DynamicPolicyState()
        state.consume_model_text('{"actions":[{"kind":"abstain"}]}')
        self.assertEqual(state.stage, STAGE_DONE)
        self.assertTrue(state.abstained)
        self.assertEqual(state.generations, 1)

    def test_valid_success_is_structural_only(self):
        result = verify_counts(outcome(2, 2, True, "none"),
                               counters(2, 1, 1, False, True))
        self.assertTrue(result["source_original_task_success"])
        self.assertFalse(result["pretrained_model_identity_independently_attested"])
        self.assertFalse(result["trained_model_gain_proven"])
        self.assertFalse(result["can_release"])

    def test_abstention_first_step_not_fabricated(self):
        result = verify_counts(outcome(1, 0, False, "abstain"),
                               counters(1, 0, 0, True, True))
        self.assertFalse(result["source_original_task_success"])

    def test_wrong_focus_terminality(self):
        result = verify_counts(outcome(1, 1, False, "wrong_focus_or_state"),
                               counters(1, 1, 0, False, False, True))
        self.assertFalse(result["source_original_task_success"])

    def test_wrong_code_is_terminal_and_not_written(self):
        result = verify_counts(outcome(2, 1, False, "wrong_code"),
                               counters(2, 1, 1, False, True, True))
        self.assertEqual(result["native_actions_dispatched"], 1)

    def test_second_abstain(self):
        result = verify_counts(outcome(2, 1, False, "abstain"),
                               counters(2, 1, 0, True, True))
        self.assertFalse(result["source_original_task_success"])

    def test_mutated_and_fabricated_counters_rejected(self):
        base = outcome(2, 2, True, "none")
        valid = counters(2, 1, 1, False, True)
        for key, bad in (
            ("model_generations_via_wrapped_generate", 1),
            ("model_time_token_filters_installed", 1),
            ("distinct_original_pngs_observed", 1),
            ("pretrained_model_identity_attested", True),
            ("native_source_task_success_attested", True),
            ("blind_ood_proven", True),
            ("generalizable_hala_gain_proven", True),
            ("focus_clicks", 0),
            ("type_calls", 0),
        ):
            with self.subTest(key=key):
                forged = dict(valid, **{key: bad})
                with self.assertRaises(DynamicModelBridgeRefused):
                    verify_counts(base, forged)
        for result in (
            dict(base, role="BLIND_OOD"),
            dict(base, official_train_admitted=True),
            dict(base, categorical_failure="fake_success"),
            dict(base, source_state_checked=False),
        ):
            with self.assertRaises(DynamicModelBridgeRefused):
                verify_counts(result, valid)

    def test_no_unregistered_scene(self):
        with self.assertRaises(DynamicModelBridgeRefused):
            run_preloaded_dynamic_source_research(
                None, None, None, None,
                ("fake", "en", 0, False, 0, "SIM-00"))

    def test_fake_policy_is_never_model_evidence(self):
        # A mock-only test: source caller must still authenticate real model.
        src = outcome(2, 2, True, "none")
        ctr = counters(2, 1, 1, False, True)
        class MockPolicy:
            def __init__(self, *args): pass
            def safe_contract_summary(self): return ctr
        with patch("h323.preloaded_dynamic_bridge.PreloadedDynamicVisualPolicy",
                   MockPolicy), patch(
                   "h323.preloaded_dynamic_bridge.run_original_dynamic_sequence",
                   return_value=src):
            result = run_preloaded_dynamic_source_research(
                object(), object(), object(), object(), SCENES[0])
        self.assertFalse(result["pretrained_model_identity_independently_attested"])
        self.assertFalse(result["blind_ood_proven"])
        self.assertFalse(result["can_release"])


if __name__ == "__main__":
    unittest.main()
