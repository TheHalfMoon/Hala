"""H298 original preloaded two-stage visual wiring, with FAKE VLM only.

These CPU tests do NOT prove genuine Qwen vision, model-time token
acceptance on its actual tokenizer, native action accuracy, GPU
execution, lawful TRAIN/DEV admission, blind OOD or skill gain.
"""
from __future__ import annotations

import types
from unittest import TestCase, mock

from h199.native_action_sft import source_task_prompt, parse_actions
from h296.fieldport_sequence_research import exact_task, SCENES
from h298.preloaded_form_two_stage_policy import (
    PreloadedTwoStageVisualPolicy, ModelContractRefused,
    run_fixed_original_source_research,
)

PNG_A = b"\x89PNG\r\n\x1a\n" + b"A" * 180
PNG_B = b"\x89PNG\r\n\x1a\n" + b"B" * 180
FOCUS = '{"actions":[{"kind":"click","point":[500,500]}]}'
ABSTAIN = '{"actions":[{"kind":"abstain"}]}'
TYPE = '{"actions":[{"kind":"type_text","text":"FIC-23"}]}'


class StubTokenizer:
    pass


class StubProcessor:
    tokenizer = StubTokenizer()
    def __init__(self, responses):
        self.responses = list(responses)
        self.apply_chat_template = lambda *a, **kw: None
    def batch_decode(self, *args, **kw):
        if not self.responses:
            raise ValueError("No more model responses")
        return [self.responses.pop(0)]


class StubModel:
    device = "cpu"
    def __init__(self):
        self.calls = 0
        self.filters = []
    def eval(self):
        return self
    def generate(self, **kwargs):
        assert callable(kwargs["prefix_allowed_tokens_fn"])
        self.calls += 1
        self.filters.append(kwargs["prefix_allowed_tokens_fn"])
        return [[101, 102]]


class FakeOnlyVisualPolicy:
    """Fake H208 unit adapter; NEVER loaded into real evaluation."""
    def __init__(self, processor, model, *, max_new_tokens=160):
        self.processor = processor
        self.model = model
        assert max_new_tokens == 160
    def __call__(self, png, prompt):
        assert type(png) is bytes and png.startswith(b"\x89PNG")
        assert "screenshot" in prompt.lower()
        self.model.eval()
        self.model.generate(input_ids=[[11]])
        return self.processor.batch_decode([[101, 102]])[0]


class H298PreloadedFormPolicyTests(TestCase):
    def setUp(self):
        self.task = exact_task(SCENES[0])
        self.native_prompt = source_task_prompt(self.task)
        self.patch_model = mock.patch(
            "h208.preloaded_vlm_policy.PreloadedVisionPolicy",
            FakeOnlyVisualPolicy)
        self.patch_filter = mock.patch(
            "h298.preloaded_form_two_stage_policy.model_time_filter",
            side_effect=lambda tok,stage: lambda batch_id,ids: [1])
        self.patch_model.start()
        self.mock_filter = self.patch_filter.start()
        self.addCleanup(self.patch_filter.stop)
        self.addCleanup(self.patch_model.stop)

    def test_two_actual_wrapped_generate_calls_and_clean_source_result(self):
        proc = StubProcessor([FOCUS, TYPE])
        model = StubModel()
        policy = PreloadedTwoStageVisualPolicy(proc, model, self.task)
        focus = policy(PNG_A, self.native_prompt)
        self.assertEqual(parse_actions(focus)["actions"][0], {
            "kind": "click", "point": [480, 360]})
        typed = policy(PNG_B, self.native_prompt)
        self.assertEqual(typed, TYPE)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(model.filters), 2)
        self.assertEqual(self.mock_filter.call_count, 2)
        self.assertEqual([c.args[1] for c in self.mock_filter.call_args_list],
                         ["focus", "type"])
        safe = policy.safe_contract_summary()
        self.assertEqual(safe["model_generations_via_wrapped_generate"], 2)
        self.assertEqual(safe["model_time_token_filters_installed"], 2)
        self.assertEqual(safe["distinct_original_pngs_observed"], 2)
        self.assertEqual(safe["focus_clicks"], 1)
        self.assertEqual(safe["type_calls"], 1)
        self.assertFalse(safe["pretrained_model_identity_attested"])
        self.assertFalse(safe["real_owner_private_gpu_run_attested"])
        self.assertFalse(safe["native_source_task_success_attested"])
        self.assertFalse(safe["blind_ood_proven"])
        with self.assertRaises(ModelContractRefused):
            policy(PNG_A, self.native_prompt)
        self.assertNotIn("point", str(safe).lower())
        self.assertNotIn("FIC-23", str(safe))

    def test_first_stage_abstention_never_permits_more_predictions(self):
        model = StubModel()
        policy = PreloadedTwoStageVisualPolicy(
            StubProcessor([ABSTAIN]), model, self.task)
        self.assertEqual(policy(PNG_A, self.native_prompt), ABSTAIN)
        self.assertEqual(model.calls, 1)
        state = policy.safe_contract_summary()
        self.assertTrue(state["protocol_terminal"])
        self.assertTrue(state["abstained"])
        self.assertEqual(state["type_calls"], 0)
        with self.assertRaises(ModelContractRefused):
            policy(PNG_B, self.native_prompt)

    def test_second_stage_abstention_never_becomes_a_success(self):
        policy = PreloadedTwoStageVisualPolicy(
            StubProcessor([FOCUS, ABSTAIN]), StubModel(), self.task)
        policy(PNG_A, self.native_prompt)
        self.assertEqual(policy(PNG_B, self.native_prompt), ABSTAIN)
        safe = policy.safe_contract_summary()
        self.assertEqual(safe["focus_clicks"], 1)
        self.assertEqual(safe["type_calls"], 0)
        self.assertTrue(safe["abstained"])
        self.assertTrue(safe["protocol_terminal"])

    def test_duplicate_or_fake_screenshot_and_prompt_refused_without_generation(self):
        model = StubModel()
        policy = PreloadedTwoStageVisualPolicy(
            StubProcessor([FOCUS, TYPE]), model, self.task)
        for png,prompt in (
            (b"not PNG" * 50, self.native_prompt),
            (PNG_A, source_task_prompt(exact_task(SCENES[1]))),
            (PNG_A, self.task),
        ):
            with self.subTest(reason=prompt[:12]), self.assertRaises(ModelContractRefused):
                policy(png, prompt)
        self.assertEqual(model.calls, 0)
        policy(PNG_A, self.native_prompt)
        with self.assertRaises(ModelContractRefused):
            policy(PNG_A, self.native_prompt)
        self.assertEqual(model.calls, 1)

    def test_model_raw_format_error_is_sanitized_with_no_success(self):
        model = StubModel()
        policy = PreloadedTwoStageVisualPolicy(
            StubProcessor(['{"actions":[{"kind":"click","point":[1000,12]}]}']),
            model,self.task)
        with self.assertRaisesRegex(ModelContractRefused, "model-time visual grammar"):
            policy(PNG_A, self.native_prompt)
        self.assertEqual(model.calls, 1)
        safe = policy.safe_contract_summary()
        self.assertEqual(safe["model_generations_via_wrapped_generate"], 0)
        self.assertEqual(safe["focus_clicks"], 0)
        self.assertFalse(safe["protocol_terminal"])

    def test_second_stage_does_not_accept_click_or_private_text(self):
        for invalid in (
            '{"actions":[{"kind":"click","point":[10,20]}]}',
            '{"actions":[{"kind":"type_text","text":"private secret"}]}',
            '{"actions":[{"kind":"type_text","text":"FIC-23"}],"debug":true}',
        ):
            model = StubModel()
            policy = PreloadedTwoStageVisualPolicy(
                StubProcessor([FOCUS, invalid]), model, self.task)
            policy(PNG_A, self.native_prompt)
            with self.subTest(raw=invalid[:12]), self.assertRaises(ValueError):
                policy(PNG_B, self.native_prompt)
            self.assertEqual(policy.safe_contract_summary()["type_calls"], 0)
            self.assertFalse(policy.safe_contract_summary()["protocol_terminal"])

    def test_unregistered_model_source_and_task_never_fake_attestation(self):
        for fake in (object(), None):
            with self.assertRaises(ModelContractRefused):
                PreloadedTwoStageVisualPolicy(
                    StubProcessor([FOCUS]), fake,self.task)
        with self.assertRaises(ModelContractRefused):
            PreloadedTwoStageVisualPolicy(
                object(), StubModel(),self.task)
        with self.assertRaises(ModelContractRefused):
            run_fixed_original_source_research(
                None, None, StubProcessor([FOCUS]), StubModel(),
                ("invented", "en", 0, False, 0, "FIC-11"))


if __name__ == "__main__":
    import unittest
    unittest.main()
