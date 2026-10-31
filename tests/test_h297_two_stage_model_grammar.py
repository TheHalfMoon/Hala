"""H297 source-only grammar/state tests; synthetic literals, NO Qwen GPU."""
from __future__ import annotations

import json
import sys
import types
from unittest import TestCase, mock

from h199.native_action_sft import parse_actions
from h296.fieldport_sequence_research import SCENES, exact_task, IDS
from h297.two_stage_form_grammar import (
    STAGE_FOCUS, STAGE_TYPE, STAGE_DONE, STRICT_TYPE_REGEX,
    TwoStepFormPolicyState, model_visible_prompt, model_time_filter,
    strict_type_native,
)


class H297RealModelGrammarContractTests(TestCase):
    def test_two_step_relative_click_then_exact_native_type(self):
        state = TwoStepFormPolicyState()
        self.assertEqual(state.stage, STAGE_FOCUS)
        initial = '{"actions":[{"kind":"click","point":[500,500]}]}'
        converted = state.consume_model_text(initial)
        action = parse_actions(converted)["actions"][0]
        self.assertEqual(action, {"kind": "click", "point": [480, 360]})
        self.assertEqual(state.stage, STAGE_TYPE)
        final = '{"actions":[{"kind":"type_text","text":"FIC-23"}]}'
        self.assertEqual(strict_type_native(final), final)
        self.assertEqual(state.consume_model_text(final), final)
        self.assertEqual(state.stage, STAGE_DONE)
        self.assertEqual(state.categorical_state(), {
            "generations": 2, "focus_clicks": 1, "type_calls": 1,
            "abstained": False, "terminal": True,
            "actual_pretrained_model_generation_proven": False,
            "independent_blind_ood_or_train_admitted": False,
        })
        with self.assertRaises(ValueError):
            state.consume_model_text(final)

    def test_stage_restricted_abstention_never_claims_native_typing(self):
        abstain = '{"actions":[{"kind":"abstain"}]}'
        for stage in (STAGE_FOCUS, STAGE_TYPE):
            state = TwoStepFormPolicyState()
            if stage == STAGE_TYPE:
                state.consume_model_text(
                    '{"actions":[{"kind":"click","point":[0,999]}]}')
            self.assertEqual(state.consume_model_text(abstain), abstain)
            self.assertTrue(state.categorical_state()["terminal"])
            self.assertTrue(state.categorical_state()["abstained"])
            self.assertEqual(state.categorical_state()["type_calls"], 0)
            with self.assertRaises(ValueError):
                state.consume_model_text(abstain)

    def test_no_type_without_focus_and_no_second_click(self):
        state = TwoStepFormPolicyState()
        with self.assertRaises(ValueError):
            state.consume_model_text(
                '{"actions":[{"kind":"type_text","text":"FIC-11"}]}')
        self.assertEqual(state.generations, 0)
        self.assertEqual(state.stage, STAGE_FOCUS)
        state.consume_model_text('{"actions":[{"kind":"click","point":[20,20]}]}')
        for invalid in (
            '{"actions":[{"kind":"click","point":[20,20]}]}',
            '{"actions":[{"kind":"scroll","point":[20,20],"dy":240}]}',
            '{"actions":[{"kind":"key_press","point":[20,20],"key":"Enter"}]}',
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                state.consume_model_text(invalid)
            self.assertEqual(state.stage, STAGE_TYPE)
            self.assertEqual(state.generations, 1)

    def test_strict_text_rejects_raw_secret_extra_actions_and_whitespace(self):
        bad = (
            '{"actions":[{"kind":"type_text","text":"FIC-1"}]}',
            '{"actions":[{"kind":"type_text","text":"FIC-100"}]}',
            '{"actions":[{"kind":"type_text","text":"FIC-11 " }]}',
            '{"actions":[{"kind":"type_text","text":"FIC-11"},{"kind":"click","point":[1,2]}]}',
            '{"actions":[{"kind":"type_text","text":"FIC-11","other":"private"}]}',
            '{"actions":[{"kind":"type_text","text":"SECRETS"}]}',
            '{"actions":[{"kind":"type_text","text":"FIC-11"}],"debug":"raw"}',
            '{"actions":[{"kind":"type_text","text":"FIC-11"}]}\n',
            '{"actions":[{"kind":"type_text","text":"FIC-11"}]}<html>',
        )
        for raw in bad:
            with self.subTest(raw=raw[:35]), self.assertRaises(ValueError):
                strict_type_native(raw)
        for i in range(100):
            text = '{"actions":[{"kind":"type_text","text":"FIC-%02d"}]}' % i
            self.assertEqual(strict_type_native(text), text)
        self.assertIn("FIC-[0-9]{2}", STRICT_TYPE_REGEX)

    def test_model_visible_step_instructions_do_not_contain_source_gold(self):
        self.assertEqual(len(SCENES), 12)
        for scene in SCENES:
            task = exact_task(scene)
            first = model_visible_prompt(task, STAGE_FOCUS)
            second = model_visible_prompt(task, STAGE_TYPE)
            self.assertTrue(first.startswith(task))
            self.assertTrue(second.startswith(task))
            self.assertIn("RELATIVE", first)
            self.assertIn("already-focused", second)
            self.assertIn("FIC-00 through FIC-99", second)
            self.assertNotIn('data-field-id', first+second)
            self.assertNotIn("fieldPortState", first+second)
            self.assertNotIn("reference coordinates", first+second)
        for invalid in ("done", "", "CLICK"):
            with self.assertRaises(ValueError):
                model_visible_prompt(exact_task(SCENES[0]), invalid)
        with self.assertRaises(ValueError):
            model_visible_prompt("short", STAGE_FOCUS)

    def test_model_time_focus_filter_delegates_to_frozen_h273(self):
        token = object()
        with mock.patch(
            "h297.two_stage_form_grammar.build_normalized_token_filter",
            return_value=token
        ) as called:
            self.assertIs(model_time_filter("tokenizer", STAGE_FOCUS), token)
            called.assert_called_once_with("tokenizer")
        with self.assertRaises(ValueError):
            model_time_filter("tokenizer", STAGE_DONE)

    def test_stage_two_model_time_enforcer_is_real_regex_not_posthoc(self):
        class FakeTokenizer:
            eos_token_id = 9
            all_special_ids = [0, 9]
            def __len__(self):
                return 10
            def encode(self, content):
                return [1]
            def decode(self, ids):
                return "".join(str(x) for x in ids)

        class FakeRegex:
            def __init__(self, regex):
                self.regex = regex

        class FakeData:
            def __init__(self, tokens, decode, eos, is_vocab, size):
                self.tokens=tokens
                self.eos=eos
                self.size=size

        class FakeEnforcer:
            def __init__(self, data, grammar):
                self.data=data
                self.grammar=grammar
            def get_allowed_tokens(self, ids):
                return types.SimpleNamespace(allowed_tokens=[2, 3])

        fake = types.ModuleType("lmformatenforcer")
        fake.RegexParser=FakeRegex
        fake.TokenEnforcer=FakeEnforcer
        fake.TokenEnforcerTokenizerData=FakeData

        with mock.patch.dict(sys.modules, {"lmformatenforcer": fake}):
            prefix=model_time_filter(FakeTokenizer(), STAGE_TYPE)
            self.assertEqual(prefix(0, types.SimpleNamespace(tolist=lambda:[1])), [2,3])
            with self.assertRaises(ValueError):
                prefix(1, types.SimpleNamespace(tolist=lambda:[1]))
            with self.assertRaises(ValueError):
                prefix(0, types.SimpleNamespace(tolist=lambda:[]))
        with mock.patch.dict(sys.modules, {"lmformatenforcer": fake}):
            with self.assertRaises(ValueError):
                model_time_filter(object(), STAGE_TYPE)


if __name__ == "__main__":
    import unittest
    unittest.main()
