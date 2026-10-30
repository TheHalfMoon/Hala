"""H239 synthetic-only tests for bilingual Qwen processor alignment policy.

No Qwen weights, no owner media, no network. The fake below exercises
the contract; ONLY the optional separate real processor run verifies HF.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h239.bilingual_processor_parity import (
    SAMPLES, audit_type_text_sample, make_synthetic_image,
    run_offline_real_processor,
)


class TinyTensor:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def numel(self):
        return len(self.values)


class SyntheticProcessor:
    def __init__(self):
        self.last_target = None
        self.mutate_prefix = False
        self.mutate_vision = False
        self.mutate_grid = False
        self.remove_vision = False
        self.malformed_decode = False
        self.make_two_rows = False

    def apply_chat_template(self, messages, *, add_generation_prompt,
                            tokenize, return_dict, return_tensors):
        if not tokenize or not return_dict or return_tensors != "pt":
            raise ValueError("invalid processor options")
        if len(messages) == 2:
            self.last_target = messages[-1]["content"][0]["text"]
        source = messages[0]["content"][0]
        if source.get("type") != "image" or source["image"].size != (960, 720):
            raise ValueError("missing synthetic source image")
        prefix = [101, 102, 103, 104, 105, 106]
        sequence = prefix if add_generation_prompt else prefix + [40, 41, 42, 43, 44, 45]
        if self.mutate_prefix and not add_generation_prompt:
            sequence[2] = 999
        vision = [11, 12, 13, 14]
        if self.mutate_vision and not add_generation_prompt:
            vision[-1] = 99
        grid = [1, 2, 3]
        if self.mutate_grid and not add_generation_prompt:
            grid[-1] = 99
        rows = [TinyTensor(sequence)]
        if self.make_two_rows:
            rows.append(TinyTensor(sequence))
        output = {"input_ids": rows, "image_grid_thw": TinyTensor(grid)}
        if not self.remove_vision:
            output["pixel_values"] = TinyTensor(vision)
        return output

    def batch_decode(self, rows, **kwargs):
        if kwargs.get("skip_special_tokens") is not True:
            raise ValueError("special tokens must be stripped")
        return ["not JSON" if self.malformed_decode else self.last_target]


class H239BilingualProcessorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.p = SyntheticProcessor()
        fake = types.SimpleNamespace(equal=lambda a, b: a.values == b.values)
        self.torch = patch.dict(sys.modules, {"torch": fake})
        self.torch.start()
        self.addCleanup(self.torch.stop)

    def example(self, index=0):
        language, task, text = SAMPLES[index]
        return audit_type_text_sample(
            self.p, language=language, task=task, text=text)

    def test_bilingual_screenshot_action_completion_only_mask(self):
        for index, (lang, _task, _text) in enumerate(SAMPLES):
            with self.subTest(lang=lang):
                result = self.example(index)
                self.assertEqual(result["language"], lang)
                self.assertTrue(result["input_prefix_exact"])
                self.assertTrue(result["assistant_completion_only_mask"])
                self.assertTrue(result["text_action_roundtrip"])
                self.assertTrue(result["vision_identical"])
                self.assertEqual(result["prompt_tokens"], 6)
                self.assertEqual(result["completion_tokens"], 6)
                self.assertEqual(result["vision_elements"], 4)
                self.assertFalse(result["model_loaded"])
                self.assertFalse(result["training_approved"])
                self.assertFalse(result["ood_evaluation_approved"])
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertEqual(result["model_calls"], 0)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())

    def test_invalid_source_image_dimensions_denied(self):
        from PIL import Image
        for size in ((300, 300), (960, 500)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                lang, task, text = SAMPLES[0]
                audit_type_text_sample(
                    self.p, language=lang, task=task, text=text,
                    image=Image.new("RGB", size))

    def test_malformed_decode_denied(self):
        self.p.malformed_decode = True
        with self.assertRaises(ValueError):
            self.example(1)

    def test_mutated_prefix_denied(self):
        self.p.mutate_prefix = True
        with self.assertRaises(ValueError):
            self.example()

    def test_mutated_visual_values_denied(self):
        self.p.mutate_vision = True
        with self.assertRaises(ValueError):
            self.example()

    def test_mutated_visual_grid_denied(self):
        self.p.mutate_grid = True
        with self.assertRaises(ValueError):
            self.example()

    def test_removed_visual_tensor_denied(self):
        self.p.remove_vision = True
        with self.assertRaises(ValueError):
            self.example()

    def test_multiple_processor_rows_denied(self):
        self.p.make_two_rows = True
        with self.assertRaises(ValueError):
            self.example()

    def test_task_containing_reference_answer_denied(self):
        lang, _task, text = SAMPLES[1]
        with self.assertRaisesRegex(ValueError, "leaked"):
            audit_type_text_sample(
                self.p, language=lang,
                task="المطلوب كتابة " + text, text=text)

    def test_noncanonical_text_rejected_before_tokenization(self):
        for text in ("bad\u202etext", "a\u0301", "x\nmore", ""):
            with self.subTest(value=repr(text)), self.assertRaises(ValueError):
                audit_type_text_sample(
                    self.p, language="en",
                    task="Enter safe demo value.", text=text)

    def test_unconfigured_offline_environment_denied_before_hf_import(self):
        with patch.dict("os.environ", {
            "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
        }):
            with self.assertRaisesRegex(ValueError, "network disabled"):
                run_offline_real_processor(__import__("pathlib").Path("/missing"))

    def test_invented_image_is_deterministic_in_memory(self):
        a = make_synthetic_image()
        b = make_synthetic_image()
        self.assertEqual(a.size, (960, 720))
        self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == "__main__":
    unittest.main()
