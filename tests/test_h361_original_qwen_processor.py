"""H361 processor interface contract tests; fakes are NEVER real model evidence."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h357.source_candidate import capture_source_candidate
from h358.source_candidate import capture_candidate
from h361.original_qwen_processor import (
    SELECTED, ROLE, SCHEMA, real_processor_messages,
    run_pinned_offline_candidate_processor,
)


class Tensor:
    def __init__(self, values):
        self.values = list(values)
    def tolist(self):
        return list(self.values)
    def numel(self):
        return len(self.values)


class FakeProcessorForContractOnly:
    """No real tokenizer, no model. Simulated fixed IDs test fail-closed path."""
    def __init__(self):
        self.bad_prefix = False
        self.bad_pixel = False
        self.bad_grid = False
        self.bad_decoding = False
        self.omit_visual = False
        self.answer_text = None
        self.seen = []

    def apply_chat_template(self, messages, *, add_generation_prompt,
                            tokenize, return_dict, return_tensors):
        assert tokenize and return_dict and return_tensors == "pt"
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[0]["content"][0]["image"].size == (960, 720)
        assert len(messages) == (1 if add_generation_prompt else 2)
        self.seen.append(add_generation_prompt)
        prefix = [1, 4, 8, 12, 20, 28]
        full = prefix + [90, 91, 92, 93, 94, 95, 96]
        if not add_generation_prompt:
            self.answer_text = messages[-1]["content"][0]["text"]
            if self.bad_prefix:
                full[2] += 1
        obj = {
            "input_ids": [Tensor(prefix if add_generation_prompt else full)],
            "image_grid_thw": Tensor([1, 2, 3] if not self.bad_grid or add_generation_prompt
                                     else [0, 2, 3]),
        }
        if not self.omit_visual:
            obj["pixel_values"] = Tensor([3, 4, 5, 6] if not self.bad_pixel
                                          or add_generation_prompt else [3, 4, 5, 0])
        return obj

    def batch_decode(self, token_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        assert kwargs["clean_up_tokenization_spaces"] is False
        return ["invalid json" if self.bad_decoding else self.answer_text]


class NativeQwenProcessorInterfaceTests(unittest.TestCase):
    def test_required_real_processor_is_unavailable_without_authorized_source(self):
        self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)
        with self.assertRaises(ValueError):
            run_pinned_offline_candidate_processor(Path("not-real"), {})
        with self.assertRaises(ValueError):
            run_pinned_offline_candidate_processor(Path("not-real"),
                                                   {"controlgarden-single-app":Path(".")})
        self.assertEqual(tuple(SELECTED["controlgarden-single-app"]), (0, 1, 2, 3))
        self.assertEqual(len(SELECTED["noterelay-original-multistep"]), 8)

    @unittest.skipUnless(
        os.environ.get("HALA_H361_BROWSER_QA") == "1"
        and importlib.util.find_spec("playwright") is not None
        and importlib.util.find_spec("PIL") is not None,
        "real source browser and Pillow required; opt-in only in H361 CI")
    def test_real_source_screens_with_fake_processor_interface_are_not_training(self):
        with TemporaryDirectory() as temp, patch.dict(sys.modules, {
            "torch": types.SimpleNamespace(
                equal=lambda a, b: a.values == b.values)
        }):
            a = Path(temp) / "controlgarden"
            b = Path(temp) / "noterelay"
            browser = "msedge" if os.name == "nt" else "chromium"
            capture_source_candidate(a, browser_kind=browser)
            capture_candidate(b, browser_kind=browser)
            f = FakeProcessorForContractOnly()
            receipts = (real_processor_messages(f, a, "controlgarden-single-app")
                        + real_processor_messages(f, b, "noterelay-original-multistep"))
            self.assertEqual(len(receipts), 12)
            self.assertEqual(Counter(x["language"] for x in receipts),
                             {"en": 6, "ar": 6})
            self.assertEqual(Counter(x["kind"] for x in receipts),
                             {"click": 10, "type_text": 2})
            self.assertEqual(f.seen, [True, False]*12)
            self.assertTrue(all(x["assistant_only_mask"]
                                and x["image_identical"]
                                and x["action_roundtrip"] for x in receipts))
            self.assertTrue(all(x["model_calls"] == 0 and
                                x["optimizer_steps"] == 0 and
                                not x["training_authorized"] for x in receipts))
            for mode in ("bad_prefix", "bad_pixel", "bad_grid",
                         "bad_decoding", "omit_visual"):
                bad = FakeProcessorForContractOnly()
                setattr(bad, mode, True)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    real_processor_messages(bad, a, "controlgarden-single-app")
            # Exact candidate image/answer source is part of the admission fence.
            manifest = a / "candidate-manifest.json"
            original = manifest.read_bytes()
            try:
                doc = json.loads(original)
                doc["official_train_admitted"] = True
                manifest.write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaises(ValueError):
                    real_processor_messages(f, a, "controlgarden-single-app")
            finally:
                manifest.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
