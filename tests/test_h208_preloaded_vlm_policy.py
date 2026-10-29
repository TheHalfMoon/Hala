"""H208 deterministic mock-processor protocol tests, NOT real VLM inference."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from h201.original_native_action_executor import exact_original_loopback_url
from h207.screenshot_prediction_bridge import run_screenshot_only_policy
from h208.preloaded_vlm_policy import PreloadedVisionPolicy
from test_h203_screenshot_bound_dispatch import Page, generated_960_png

PROMPT = "You are Hala. 960x720 screenshot. Task: Click the red square."
CLICK = json.dumps({"actions":[{"kind":"click","point":[300,200]}]})


class Batch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self, *, pixels=True):
        self.pixels = pixels
        self.messages = None
        self.batch = None
        self.decode_args = None
        self.image_pixel = None
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": True, "add_generation_prompt": True,
            "return_dict": True, "return_tensors": "pt",
        }
        content = messages[0]["content"]
        assert messages[0]["role"] == "user"
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == PROMPT
        image = content[0]["image"]
        assert image.mode == "RGB" and image.size == (960, 720)
        self.image_pixel = image.getpixel((0, 0))
        b = Batch({"input_ids": [[123,456,789]]})
        if self.pixels:
            b["pixel_values"] = [self.image_pixel]
            b["image_grid_thw"] = [(1, 27, 34)]
        self.batch = b
        return b
    def batch_decode(self, encoded, **kwargs):
        self.decode_args = kwargs
        assert len(encoded) == 1
        return ["".join(chr(x) for x in encoded[0])]


class FakeModel:
    device = "cpu"
    def __init__(self, text=CLICK):
        self.text = text
        self.prefix = [123,456,789]
        self.generate_calls = 0
        self.evaluated = False
        self.last_batch = None
        self.extra_prefix = None
        self.on_generate = lambda: None
    def eval(self):
        self.evaluated = True
        return self
    def generate(self, **kwargs):
        self.generate_calls += 1
        self.last_batch = kwargs
        self.on_generate()
        if self.extra_prefix is not None:
            prefix = self.extra_prefix
        else:
            prefix = self.prefix
        return [prefix + [ord(x) for x in self.text]]


class PreloadedVisionTests(unittest.TestCase):
    def setUp(self):
        self.png = generated_960_png(21)
        self.processor = FakeProcessor()
        self.model = FakeModel()
        self.model_policy = PreloadedVisionPolicy(self.processor, self.model)
        self.guard = patch.dict(sys.modules, {"torch": SimpleNamespace(
            inference_mode=nullcontext,
        )})

    def invoke(self, png=None):
        with self.guard:
            return self.model_policy(self.png if png is None else png, PROMPT)

    def test_real_decoded_png_pixels_reach_multimodal_template(self):
        action = self.invoke()
        self.assertEqual(action, CLICK)
        self.assertIsNotNone(self.processor.image_pixel)
        self.assertEqual(self.processor.messages[0]["role"], "user")
        self.assertEqual(self.model.generate_calls, 1)
        self.assertTrue(self.model.evaluated)
        self.assertEqual(self.processor.batch.device, "cpu")
        self.assertIsNotNone(self.model.last_batch["pixel_values"])
        self.assertEqual(self.processor.decode_args, {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        })
        self.assertFalse("page" in str(self.processor.messages))
        self.assertFalse("DOM" in PROMPT)

    def test_different_real_images_reach_processor_differently(self):
        a = self.invoke(generated_960_png(21))
        one = self.processor.image_pixel
        b = self.invoke(generated_960_png(22))
        two = self.processor.image_pixel
        self.assertEqual(a, b)
        # This proves the processor sees actual distinct raster input,
        # NOT that the fake model learns or reacts to those pixels.
        self.assertNotEqual(one, two)

    def test_actual_h207_bridge_accepts_model_shaped_callback(self):
        page = Page(self.png)
        url = exact_original_loopback_url(8201)
        # H207 supplies its full source_task_prompt, not our short test string.
        previous = self.processor.apply_chat_template
        def accept_any_task(messages, **kwargs):
            self.processor.messages = messages
            assert "screenshot-only computer-use" in messages[0]["content"][1]["text"]
            self.processor.image_pixel = messages[0]["content"][0]["image"].getpixel((0,0))
            self.processor.batch = Batch({
                "input_ids": [[123,456,789]],
                "pixel_values": [self.processor.image_pixel],
            })
            return self.processor.batch
        self.processor.apply_chat_template = accept_any_task
        try:
            with self.guard:
                result = run_screenshot_only_policy(
                    page, "Click the red square.", url, self.model_policy)
        finally:
            self.processor.apply_chat_template = previous
        self.assertTrue(result["preaction_png_verified"])
        self.assertFalse(result["model_identity_attested"])
        self.assertEqual(result["native_events"], 1)
        self.assertEqual(page.events, [("click",300,200)])

    def test_missing_visual_tensor_fail_closed_before_generation(self):
        self.processor.pixels = False
        with self.assertRaisesRegex(ValueError,"multimodal processor"):
            self.invoke()
        self.assertEqual(self.model.generate_calls, 0)

    def test_empty_visual_tensor_fail_closed_before_generation(self):
        original = self.processor.apply_chat_template
        def missing_pixels(messages, **kwargs):
            result = original(messages, **kwargs)
            result["pixel_values"] = []
            return result
        self.processor.apply_chat_template = missing_pixels
        with self.assertRaisesRegex(ValueError, "multimodal processor"):
            self.invoke()
        self.assertEqual(self.model.generate_calls, 0)

    def test_output_prompt_prefix_mismatch_fail_closed(self):
        self.model.extra_prefix = [456,123,789]
        with self.assertRaisesRegex(ValueError, "generated action"):
            self.invoke()

    def test_non_json_continuation_rejected(self):
        for response in ("Clicked!", '{"actions":[{"kind":"click","point":[NaN,3]}]}',
                         '{"actions":[{"kind":"click","point":[1,3],"point":[2,4]}]}'):
            with self.subTest(response=response):
                self.model.text = response
                with self.assertRaisesRegex(ValueError, "generated action"):
                    self.invoke()

    def test_inference_exception_redacted(self):
        def broken():
            raise RuntimeError("secret-model-training-output")
        self.model.on_generate = broken
        with self.assertRaises(ValueError) as ctx:
            self.invoke()
        self.assertNotIn("secret-model-training-output", str(ctx.exception))

    def test_invalid_synthetic_png_stops_before_processor(self):
        with self.assertRaises(ValueError):
            self.invoke(b"not_png")
        self.assertEqual(self.model.generate_calls, 0)

    def test_malformed_valid_header_png_rejected_by_full_decoder(self):
        fake = self.png[:24] + b"invalid-pixels"
        with self.assertRaisesRegex(ValueError, "malformed preaction PNG"):
            self.invoke(fake)
        self.assertIsNone(self.processor.messages)

    def test_malformed_preloaded_dependencies_fail_before_input(self):
        for tokens in (0, 193, True):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError,"preloaded processor/model"):
                    PreloadedVisionPolicy(self.processor,self.model,max_new_tokens=tokens)

    def test_empty_generated_continuation_rejected(self):
        self.model.text = ""
        with self.assertRaisesRegex(ValueError, "generated action"):
            self.invoke()

    def test_abstain_generates_no_native_input(self):
        self.model.text = '{"actions":[{"kind":"abstain"}]}'
        page = Page(self.png)
        url = exact_original_loopback_url(8201)
        # bypass FakeProcessor's literal prompt expectation with a wrapper
        def relaxed(messages, **kwargs):
            self.processor.messages = messages
            self.processor.batch = Batch({
                "input_ids": [[123,456,789]], "pixel_values":[1],
            })
            return self.processor.batch
        self.processor.apply_chat_template = relaxed
        with self.guard:
            receipt=run_screenshot_only_policy(page,"Don't change the screen.",url,self.model_policy)
        self.assertEqual(receipt["native_events"],0)
        self.assertEqual(page.events,[])


if __name__ == "__main__":
    unittest.main()
