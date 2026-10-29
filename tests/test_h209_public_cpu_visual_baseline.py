"""H209 synthetic self-generated visual baseline contract tests, no model weights."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import nullcontext
import sys

from h209.public_cpu_visual_baseline import (
    MODEL_ID, MODEL_REVISION, generated_continuation, synthetic_image,
    COLORS, require,
)


class Tokens:
    def __init__(self, ids):
        self.ids = ids
    def tolist(self):
        return self.ids


class Batch(dict):
    def to(self, dev):
        if dev != "cpu":
            raise ValueError("GPU prohibited")
        return self


class FakeVisual:
    def numel(self):
        return 36


class Processor:
    def __init__(self, has_pixels=True):
        self.has_pixels = has_pixels
        self.seen_image = None
    def apply_chat_template(self, msgs, **kw):
        assert kw["tokenize"] is True and kw["return_tensors"] == "pt"
        self.seen_image = msgs[0]["content"][0]["image"]
        assert self.seen_image.size == (960,720)
        result = Batch({"input_ids":[Tokens([7,8,9])]})
        if self.has_pixels:
            result["pixel_values"] = FakeVisual()
        return result
    def batch_decode(self, answer, **kw):
        return ["".join(chr(z) for z in answer[0])]


class Model:
    def __init__(self, result="RED"):
        self.result = result
        self.calls = 0
    def generate(self, **kw):
        self.calls += 1
        return [Tokens([7,8,9] + [ord(x) for x in self.result])]


class H209BaselineTests(unittest.TestCase):
    def test_public_model_has_immutable_sha_pin_and_expected_id(self):
        self.assertEqual(MODEL_ID, "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.assertEqual(len(MODEL_REVISION),40)
        self.assertTrue(all(x in "0123456789abcdef" for x in MODEL_REVISION))
    def test_synthetic_color_cases_have_no_owner_media_input(self):
        from PIL import Image
        a = synthetic_image("RED")
        b = synthetic_image("BLUE")
        self.assertEqual(a.size,(960,720))
        self.assertEqual(b.size,(960,720))
        self.assertEqual(a.getpixel((0,0)),(255,255,255))
        self.assertNotEqual(a.getpixel((450,320)),b.getpixel((450,320)))
        self.assertEqual(a.getpixel((450,320)),(255,0,0))
        self.assertEqual(b.getpixel((450,320)),(0,0,255))
        self.assertEqual(tuple(COLORS),("RED","BLUE"))
    def test_unapproved_synthetic_scene_is_denied(self):
        for invalid in ("green","RED ",None,25):
            with self.subTest(value=str(invalid)):
                with self.assertRaises(ValueError):
                    synthetic_image(invalid)
    def test_true_pixel_message_to_model_and_continuation_only(self):
        processor=Processor()
        model=Model("BLUE")
        with patch.dict(sys.modules,{"torch": SimpleNamespace(inference_mode=nullcontext)}):
            answer,tokens,numel = generated_continuation(
                model,processor,synthetic_image("BLUE"),"Describe color.",max_new_tokens=10)
        self.assertEqual((answer,tokens,numel),("BLUE",4,36))
        self.assertEqual(model.calls,1)
    def test_no_text_only_fallback_when_image_missing(self):
        model=Model()
        with patch.dict(sys.modules,{"torch": SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaisesRegex(ValueError,"missing actual vision tensor"):
                generated_continuation(model,Processor(False),synthetic_image("RED"),"color")
        self.assertEqual(model.calls,0)
    def test_empty_model_continuation_is_rejected(self):
        with patch.dict(sys.modules,{"torch": SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaisesRegex(ValueError,"invalid model prompt prefix"):
                generated_continuation(Model(""),Processor(),synthetic_image("RED"),"color")
    def test_rejection_reason_does_not_echo_private_strings(self):
        with self.assertRaisesRegex(ValueError,"H209 real public baseline"):
            require(False,"invalid self-generated public color")


if __name__ == "__main__":
    unittest.main()
