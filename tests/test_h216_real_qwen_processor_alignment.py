"""H216 CPU-only synthetic contracts; NO real model or owner data in CI.

Fake processor tests verify the fail-closed audit. Actual HF/Qwen3-VL
processor parity is a SEPARATE opt-in Windows-only local experiment.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h216.real_qwen_processor_alignment import (
    CASES, QWEN_PROCESSOR_REV, audit_sample, token_ids,
    verify_cached_processor_root, visual_tensor_receipt,
    run_real_processor_audit,
)


class Tensor:
    def __init__(self, values):
        self.values=list(values)
    def tolist(self):
        return list(self.values)
    def numel(self):
        return len(self.values)


class FakeProcessor:
    """Only tests policy plumbing; never infer genuine tokenization."""
    def __init__(self):
        self.drop_pixel=False
        self.mutate_pixel=False
        self.mutate_prefix=False
        self.mutate_grid=False
        self.malformed_decode=False
        self.two_rows=False
        self.target=None

    def apply_chat_template(self, messages, *, tokenize, return_dict,
                            return_tensors, add_generation_prompt):
        assert tokenize and return_dict and return_tensors=="pt"
        assert len(messages) in (1,2)
        assert messages[0]["content"][0]["type"]=="image"
        assert messages[0]["content"][0]["image"].size==(960,720)
        prefix=[101,102,103,104,105,106]
        full=prefix+[31,32,33,34,35,36,37]
        if len(messages)==2:
            self.target=messages[-1]["content"][0]["text"]
            if self.mutate_prefix:
                full[1]=999
        ids=prefix if add_generation_prompt else full
        row=[Tensor(ids)]
        if self.two_rows:
            row.append(Tensor(ids))
        values=[1,2,3,4] if not self.mutate_pixel or add_generation_prompt else [1,2,3,8]
        grid=[1,2,3] if not self.mutate_grid or add_generation_prompt else [5,5,5]
        result={"input_ids":row,
                "image_grid_thw":Tensor(grid)}
        if not self.drop_pixel:
            result["pixel_values"]=Tensor(values)
        return result

    def batch_decode(self, tokens, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        return ["NOT JSON" if self.malformed_decode else self.target]


class RealProcessorPolicyUnitTests(unittest.TestCase):
    def setUp(self):
        self.p=FakeProcessor()
        # Host CI has no torch; fake only its deterministic equal()
        self.fake_torch=types.SimpleNamespace(
            equal=lambda a,b: a.values==b.values,
        )
        self.patch=patch.dict(sys.modules,{"torch":self.fake_torch})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_synthetic_english_arabic_click_examples(self):
        for i in CASES:
            with self.subTest(case=i):
                self.p.target=None
                result=audit_sample(self.p,i)
                self.assertTrue(result["input_prefix_exact"])
                self.assertTrue(result["assistant_native_click_roundtrip"])
                self.assertEqual(result["masked_prompt_tokens"],6)
                self.assertEqual(result["supervised_tokens"],7)
                self.assertEqual(result["vision_tensor_elements"],4)
                self.assertFalse(result["model_loaded"])
                self.assertFalse(result["training_step_performed"])

    def test_all_four_layout_families_eligible_for_research_only(self):
        # Synthetic fake-processor guard only. The real owner processor
        # separately passed all 96 genuine image and token sequences.
        for index in (0,31,32,63,64,79,80,95):
            with self.subTest(index=index):
                observed=audit_sample(self.p,index)
                self.assertTrue(observed["assistant_native_click_roundtrip"])
                self.assertFalse(observed["training_step_performed"])

    def test_real_hf_prompt_prefix_mismatch_fails_closed(self):
        self.p.mutate_prefix=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,0)

    def test_visual_pixels_cannot_be_removed(self):
        self.p.drop_pixel=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,1)

    def test_visual_values_cannot_change(self):
        self.p.mutate_pixel=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,64)

    def test_visual_grid_cannot_change(self):
        self.p.mutate_grid=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,65)

    def test_malformed_assistant_action_refused(self):
        self.p.malformed_decode=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,0)

    def test_multiple_rows_refused(self):
        self.p.two_rows=True
        with self.assertRaises(ValueError):
            audit_sample(self.p,0)

    def test_out_of_roster_cases_refused(self):
        for i in (-1,96,10000,None,True):
            with self.subTest(index=str(i)):
                with self.assertRaises(ValueError):
                    audit_sample(self.p,i)

    def test_invalid_token_types_and_rows_refused(self):
        for batch in ({}, {"input_ids":[]}, {"input_ids":[Tensor([-1])]},
                      {"input_ids":[Tensor([1.1,2])]}):
            with self.subTest(batch=str(batch)):
                with self.assertRaises(ValueError):
                    token_ids(batch,what="adversarial")

    def test_processor_snapshot_and_offline_mode_must_be_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/QWEN_PROCESSOR_REV
            root.mkdir()
            with patch.dict(os.environ, {
                    "HF_HUB_OFFLINE":"1",
                    "TRANSFORMERS_OFFLINE":"1"}):
                with self.assertRaises(ValueError):
                    verify_cached_processor_root(root)
            with patch.dict(os.environ, {
                    "HF_HUB_OFFLINE":"0",
                    "TRANSFORMERS_OFFLINE":"0"}):
                with self.assertRaises(ValueError):
                    verify_cached_processor_root(root)

    def test_exact_qwen3vl_processor_class_required_before_audit(self):
        # Synthetic import seam only: no transformers package, weights, or network.
        class ExpectedQwenProcessor(FakeProcessor):
            pass

        class ImpostorProcessor(FakeProcessor):
            pass

        class DerivedQwenProcessor(ExpectedQwenProcessor):
            pass

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.Qwen3VLProcessor = ExpectedQwenProcessor
        fake_transformers.AutoProcessor = types.SimpleNamespace()
        safe_source = {
            "candidate_cases_checked": 96,
            "rgba_sha_matches": 96,
            "official_train_admitted": False,
        }
        with patch.dict(sys.modules, {"transformers": fake_transformers}), \
             patch("h216.real_qwen_processor_alignment.verify_cached_processor_root"), \
             patch("h216.real_qwen_processor_alignment.verify_checked_in_source",
                   return_value=safe_source):
            for wrong in (ImpostorProcessor, DerivedQwenProcessor):
                with self.subTest(processor=wrong.__name__):
                    fake_transformers.AutoProcessor.from_pretrained = (
                        lambda *args, cls=wrong, **kwargs: cls()
                    )
                    with self.assertRaisesRegex(ValueError, "exact Qwen3VLProcessor"):
                        run_real_processor_audit(Path("/synthetic-no-cache"))
            fake_transformers.AutoProcessor.from_pretrained = (
                lambda *args, **kwargs: ExpectedQwenProcessor()
            )
            receipts = run_real_processor_audit(Path("/synthetic-no-cache"))
            self.assertEqual(len(receipts), len(CASES))
            self.assertTrue(all(r["model_loaded"] is False for r in receipts))

    def test_no_admitted_training_release_bypass(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS,frozenset())
        self.assertEqual(len(CASES),4)
        self.assertEqual(CASES,(0,1,64,65))


if __name__=="__main__":
    unittest.main()
