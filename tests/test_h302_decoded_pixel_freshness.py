"""H302: compressed-PNG re-encoding must not forge a fresh Qwen observation.

Only invented in-memory images and fake model outputs. No private source,
neural inference, task scores, training, or OOD material.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from unittest import TestCase, mock

from PIL import Image, PngImagePlugin

from h199.native_action_sft import source_task_prompt
from h298.preloaded_form_two_stage_policy import (
    ModelContractRefused, PreloadedTwoStageVisualPolicy,
    _model_visible_rgb_digest,
)

TASK = "Enter fictional code FIC-11 into field F-10."
FOCUS = '{"actions":[{"kind":"click","point":[500,500]}]}'
TYPE = '{"actions":[{"kind":"type_text","text":"FIC-11"}]}'


def invented_png(color, *, metadata=False):
    output = BytesIO()
    image = Image.new("RGB", (960, 720), color)
    kwargs = {}
    if metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text("synthetic", "nonvisual ancillary metadata only")
        kwargs["pnginfo"] = info
    image.save(output, format="PNG", **kwargs)
    return output.getvalue()


PNG_A = invented_png((20, 30, 40))
PNG_A_METADATA = invented_png((20, 30, 40), metadata=True)
PNG_B = invented_png((40, 30, 20))


class StubProcessor:
    tokenizer = object()

    def __init__(self, responses):
        self.responses = list(responses)

    def apply_chat_template(self, *args, **kwargs):
        return None

    def batch_decode(self, *args, **kwargs):
        return [self.responses.pop(0)]


class StubModel:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        assert callable(kwargs["prefix_allowed_tokens_fn"])
        self.calls += 1
        return [[10, 11]]


class FakeVision:
    def __init__(self, processor, model, *, max_new_tokens):
        assert max_new_tokens == 160
        self.processor = processor
        self.model = model

    def __call__(self, png, prompt):
        self.model.eval()
        self.model.generate(input_ids=[[10]])
        return self.processor.batch_decode([[10, 11]])[0]


class PixelFreshnessTests(TestCase):
    def setUp(self):
        self.patches = [
            mock.patch("h208.preloaded_vlm_policy.PreloadedVisionPolicy", FakeVision),
            mock.patch(
                "h298.preloaded_form_two_stage_policy.model_time_filter",
                side_effect=lambda tokenizer, stage: lambda *args: [1],
            ),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def new_policy(self):
        model = StubModel()
        policy = PreloadedTwoStageVisualPolicy(
            StubProcessor([FOCUS, TYPE]), model, TASK
        )
        return policy, model

    def test_ancillary_png_bytes_differ_but_model_visible_pixels_match(self):
        self.assertNotEqual(sha256(PNG_A).digest(), sha256(PNG_A_METADATA).digest())
        self.assertEqual(
            _model_visible_rgb_digest(PNG_A),
            _model_visible_rgb_digest(PNG_A_METADATA),
        )
        self.assertNotEqual(
            _model_visible_rgb_digest(PNG_A),
            _model_visible_rgb_digest(PNG_B),
        )

    def test_metadata_only_second_screenshot_is_terminal_without_second_model_call(self):
        policy, model = self.new_policy()
        policy(PNG_A, source_task_prompt(TASK))
        self.assertEqual(model.calls, 1)
        with self.assertRaises(ModelContractRefused):
            policy(PNG_A_METADATA, source_task_prompt(TASK))
        self.assertEqual(model.calls, 1)
        self.assertTrue(policy.safe_contract_summary()["protocol_failed_terminal"])
        with self.assertRaises(ModelContractRefused):
            policy(PNG_B, source_task_prompt(TASK))
        self.assertEqual(model.calls, 1)

    def test_visually_distinct_second_scene_allows_exact_type_stage(self):
        policy, model = self.new_policy()
        policy(PNG_A, source_task_prompt(TASK))
        self.assertEqual(policy(PNG_B, source_task_prompt(TASK)), TYPE)
        self.assertEqual(model.calls, 2)
        self.assertEqual(
            policy.safe_contract_summary()["distinct_original_pngs_observed"], 2
        )
        self.assertTrue(policy.safe_contract_summary()["protocol_terminal"])

    def test_malformed_but_png_prefixed_media_refused_before_generate(self):
        policy, model = self.new_policy()
        fake = b"\\x89PNG\\r\\n\\x1a\\n" + b"not-decoded-image" * 12
        with self.assertRaises(ModelContractRefused):
            policy(fake, source_task_prompt(TASK))
        self.assertEqual(model.calls, 0)
        self.assertTrue(policy.safe_contract_summary()["protocol_failed_terminal"])

    def test_wrong_viewport_refused_before_generate(self):
        output = BytesIO()
        Image.new("RGB", (320, 240), (20, 30, 40)).save(output, "PNG")
        policy, model = self.new_policy()
        with self.assertRaises(ModelContractRefused):
            policy(output.getvalue(), source_task_prompt(TASK))
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    import unittest
    unittest.main()
