"""H301: unexpected processor/tokenizer/model failures permanently latch the policy.

All data are invented CPU stubs. These tests do NOT attest genuine Qwen
inference, training, GUI success, blind OOD, or release qualification.
"""
from __future__ import annotations

import unittest
from io import BytesIO
from PIL import Image
from unittest import mock

from h199.native_action_sft import source_task_prompt
from h298.preloaded_form_two_stage_policy import (
    ModelContractRefused,
    PreloadedTwoStageVisualPolicy,
)

_image_buffer = BytesIO()
Image.new("RGB", (960, 720), (13, 17, 19)).save(_image_buffer, format="PNG")
PNG = _image_buffer.getvalue()
TASK = "Enter fictional code FIC-11 into field F-10."


class DummyProcessor:
    tokenizer = object()

    def apply_chat_template(self, *args, **kwargs):
        return None

    def batch_decode(self, *args, **kwargs):
        return ['{"actions":[{"kind":"abstain"}]}']


class DummyModel:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls += 1
        return [[101]]


class UnexpectedModelFailureTests(unittest.TestCase):
    def new_policy(self):
        model = DummyModel()
        return PreloadedTwoStageVisualPolicy(
            DummyProcessor(), model, TASK
        ), model

    def assert_terminal_without_retry(self, policy, model, calls):
        self.assertTrue(policy.safe_contract_summary()["protocol_failed_terminal"])
        self.assertEqual(model.calls, calls)
        with self.assertRaises(ModelContractRefused):
            policy(PNG, source_task_prompt(TASK))
        self.assertEqual(model.calls, calls)
        self.assertEqual(
            policy.safe_contract_summary()["model_generations_via_wrapped_generate"], 0
        )

    def test_unexpected_tokenizer_key_error_is_sanitized_and_terminal(self):
        policy, model = self.new_policy()
        with mock.patch(
            "h298.preloaded_form_two_stage_policy.model_time_filter",
            side_effect=KeyError("DO_NOT_LEAK_PRIVATE_MODEL_OUTPUT"),
        ):
            with self.assertRaises(ModelContractRefused) as ctx:
                policy(PNG, source_task_prompt(TASK))
        self.assertNotIn("DO_NOT_LEAK", str(ctx.exception))
        self.assert_terminal_without_retry(policy, model, 0)

    def test_unexpected_processor_attribute_error_is_sanitized_and_terminal(self):
        policy, model = self.new_policy()
        with mock.patch(
            "h298.preloaded_form_two_stage_policy.model_time_filter",
            return_value=lambda batch_id, ids: [1],
        ), mock.patch(
            "h208.preloaded_vlm_policy.PreloadedVisionPolicy",
            side_effect=AttributeError("DO_NOT_LEAK_PRIVATE_IMAGE"),
        ):
            with self.assertRaises(ModelContractRefused) as ctx:
                policy(PNG, source_task_prompt(TASK))
        self.assertNotIn("DO_NOT_LEAK", str(ctx.exception))
        self.assert_terminal_without_retry(policy, model, 0)

    def test_unexpected_model_bridge_lookup_error_is_terminal(self):
        policy, model = self.new_policy()
        with mock.patch(
            "h298.preloaded_form_two_stage_policy.model_time_filter",
            return_value=lambda batch_id, ids: [1],
        ), mock.patch(
            "h208.preloaded_vlm_policy.PreloadedVisionPolicy",
            return_value=mock.Mock(side_effect=LookupError("PRIVATE_TOKEN_IDS")),
        ):
            with self.assertRaises(ModelContractRefused) as ctx:
                policy(PNG, source_task_prompt(TASK))
        self.assertNotIn("PRIVATE_TOKEN_IDS", str(ctx.exception))
        self.assert_terminal_without_retry(policy, model, 0)


if __name__ == "__main__":
    unittest.main()
