"""H426 invented bilingual screenshots processed by the ORIGINAL H425 contract.

CPU-only original code integration and H426 fake pixel-path falsifications.
These tests DO NOT qualify a model, GPU training, admission or external OOD.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest
from unittest.mock import patch

from PIL import Image

from h426 import synthetic_h425_processor_bridge as bridge
from h426.synthetic_drag_curriculum import DOMAINS, generate

ROOT=Path(__file__).resolve().parents[1]


def fake_processor():
    source=ROOT/"tests/test_h426_full_processor_contract.py"
    spec=importlib.util.spec_from_file_location(
        "h426_existing_fake_processor_replay", source)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FakeImageProcessor()


class SyntheticH425InputParityTests(unittest.TestCase):
    def test_all_split_domain_language_pairs_use_actual_h425_prompt_target(self):
        from h426.h425_loss_bridge import H425_SOURCE, _load_source_module
        h425=_load_source_module("h426_test_authentic_h425", H425_SOURCE)
        for split in ("train","dev"):
            for domain in DOMAINS:
                for language in ("en","ar"):
                    with self.subTest(split=split,domain=domain,language=language):
                        sample=bridge.pack_generated(split,11,domain,language)
                        generated=generate(split,11,domain)
                        row=next(a for a in generated["actions"]
                                 if a["language"]==language)
                        self.assertEqual(sample["image"].size,(600,400))
                        self.assertEqual(sample["image"].mode,"RGB")
                        self.assertEqual(sample["request"],h425.prompt(row))
                        self.assertEqual(
                            json.loads(sample["answer"]), row["target"])
                        self.assertEqual(sample["source_sha256"],
                                         generated["image_sha256"])
                        self.assertEqual(len(sample["source_case_identity_sha256"]),64)
                        self.assertEqual(sample["family"],generated["family"])
                        self.assertFalse(sample["source_is_admitted_to_train"])
                        self.assertFalse(sample["new_model_trained"])

    def test_thumbnail_matches_original_h425_image_resize_exactly(self):
        for domain in DOMAINS:
            sample=bridge.pack_generated("dev",2,domain,"en")
            generated=generate("dev",2,domain)
            with Image.open(io.BytesIO(generated["original_png_bytes"])) as png:
                original=png.convert("RGB")
            original.thumbnail((640,400),Image.Resampling.LANCZOS)
            self.assertEqual(sample["image"].size,original.size)
            self.assertEqual(sample["image"].tobytes(),original.tobytes())

    def test_source_image_same_for_two_different_bilingual_actions(self):
        for domain in DOMAINS:
            en=bridge.pack_generated("train",40,domain,"en")
            ar=bridge.pack_generated("train",40,domain,"ar")
            self.assertEqual(en["source_sha256"],ar["source_sha256"])
            self.assertEqual(en["image"].tobytes(),ar["image"].tobytes())
            self.assertNotEqual(en["answer"],ar["answer"])
            self.assertNotEqual(en["request"],ar["request"])

    def test_synthetic_h425_bridge_passes_existing_fake_full_processor(self):
        proc=fake_processor()
        for domain in DOMAINS:
            for language in ("en","ar"):
                with self.subTest(domain=domain,language=language):
                    result=bridge.check_generated_with_real_processor(
                        proc,"train",5,domain,language)
                    self.assertEqual(result["domain"],domain)
                    self.assertEqual(result["language"],language)
                    self.assertGreater(result["image_patch_tokens"],0)
                    self.assertGreater(result["weighted_coordinate_tokens"],0)
                    self.assertFalse(result["new_model_trained"])
                    self.assertFalse(result["source_is_admitted_to_train"])
                    self.assertEqual(len(result["synthetic_source_image_sha256"]),64)

    def test_bad_language_and_split_refused_before_any_processor(self):
        for split,index,domain,language in (
            ("train",1,"slider","fr"),
            ("train",1,"slider",True),
            ("hidden",1,"slider","en"),
            ("train",-1,"slider","en"),
            ("train",2,"click","ar"),
        ):
            with self.subTest(language=language,domain=domain):
                with self.assertRaises(ValueError):
                    bridge.pack_generated(split,index,domain,language)

    def test_forged_source_png_digest_is_refused(self):
        real=generate("train",2,"slider")
        tampered=dict(real,image_sha256="0"*64)
        with patch.object(bridge,"generate",return_value=tampered):
            with self.assertRaisesRegex(ValueError,"PNG byte provenance"):
                bridge.pack_generated("train",2,"slider","en")

    def test_promoting_candidate_source_role_to_admitted_is_refused(self):
        real=generate("train",3,"slider")
        tampered=dict(real,actions=[dict(row) for row in real["actions"]])
        tampered["actions"][0]["source_role"]="admitted_original_train"
        with patch.object(bridge,"generate",return_value=tampered):
            with self.assertRaisesRegex(ValueError,"source role/provenance"):
                bridge.pack_generated("train",3,"slider","en")

    def test_mismatched_h425_native_target_is_refused(self):
        fake=SimpleNamespace(
            target=lambda row,w,h:{"wrong":"target"},
            prompt=lambda row:row["body"],
        )
        with patch.object(bridge,"_load_source_module",return_value=fake):
            with self.assertRaisesRegex(ValueError,"differs from H425"):
                bridge.pack_generated("dev",1,"slider","ar")

    def test_real_qwen_cpu_preflight_registers_native_generated_guis(self):
        # The actual Qwen process is only executed in the owner's pinned
        # offline CPU environment, never by lightweight GitHub/source CI.
        source=(ROOT/"h426/full_processor_contract.py").read_text("utf8")
        self.assertIn("check_generated_with_real_processor(",source)
        self.assertIn("actual_h425_synthetic_gui_test_count",source)
        self.assertIn('for split in ("train", "dev")',source)
        self.assertIn('for language in ("en", "ar")',source)
        self.assertIn("verified_processor_handoff(root)",source)

    def test_this_bridge_cannot_run_training_or_publish_weights(self):
        source=(ROOT/"h426/synthetic_h425_processor_bridge.py").read_text("utf8")
        for phrase in ("optimizer.step(", "model.generate(",
                       "torch.optim.", "requests.get(", "kaggle kernels push",
                       "model.save_pretrained(", "cuda:0"):
            self.assertNotIn(phrase,source)
        self.assertIn("source_is_admitted_to_train",source)
        self.assertIn("check(processor",source)


if __name__=="__main__":
    unittest.main()
