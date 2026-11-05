"""H425 H421 384-case source-original verifier negative contracts.

Pure synthetic metadata exercises false claim rejection; not GPU model evidence.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "h425_pair_receipt",ROOT/"h425/validate_h421_pair_receipt.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
h=v.h

def temporary_fake():
    return {
      "schema":v.SCHEMA,
      "status":v.COMPLETE,
      "official_base":h.BASE,"official_revision":h.REV,
      "h417_parent_adapter_sha256":h.PARENT_ADAPTER_SHA,
      "source_pairs_jsonl_sha256":h.SOURCE_JSONL_SHA,
      "source_pairs_receipt_sha256":h.SOURCE_RECEIPT_SHA,
      "h425_completed_source_optimizer_steps":12696,
      "h425_source_only_research_gate_passed":True,
      "original_heldout_images":192,
      "native_contrastive_original_instructions":384,
      "device_name":"Tesla T4",
      "h425_authentic_trained_adapter_sha256":"a"*64,
      "h425_authentic_private_receipt_sha256":"b"*64,
      "model_training_executed_in_this_kernel":False,
      "independent_external_ood":False,
      "model_release_qualified":False,
      "weights_public":False,
      "research_pair_improved":False}

def make_receipt(folder,data):
    raw=json.dumps(data,sort_keys=True).encode()
    Path(folder,"receipt.json").write_bytes(raw)
    Path(folder,"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())

class H425PairOriginalVerifier(unittest.TestCase):
    def test_original_eval_scorer_and_schema_pins(self):
        self.assertEqual((h.IMAGE_PAIRS,h.SAMPLE_N),(192,384))
        self.assertEqual(h.SOURCE_JSONL_SHA,
           "42d93219ec4083920462d4194b175154f1f12d9b3e7b36ffd1068d8f8a4d96b9")
        self.assertEqual(h.SOURCE_RECEIPT_SHA,
           "02a4d17ead71fc06019a83669a04c9779430de5a7a48f800429ad2241d2ebaf4")
        v._validate_metadata(temporary_fake())

    def test_private_model_not_trained_or_gate_false_is_rejected(self):
        for field,bad in (
            ("h425_source_only_research_gate_passed",False),
            ("h425_completed_source_optimizer_steps",9452),
            ("h425_authentic_trained_adapter_sha256","broken"),
            ("device_name","CPU"),
            ("research_pair_improved","true")):
            with self.subTest(field=field):
                m=temporary_fake();m[field]=bad
                with self.assertRaisesRegex(ValueError,"metadata invalid"):
                    v._validate_metadata(m)

    def test_external_ood_and_release_false_in_all_cases(self):
        for field in ("model_release_qualified","weights_public",
                      "independent_external_ood","model_training_executed_in_this_kernel"):
            with self.subTest(field=field):
                m=temporary_fake();m[field]=True
                with self.assertRaisesRegex(ValueError,"metadata invalid"):
                    v._validate_metadata(m)

    def test_exact_original_private_model_provider_sha_required(self):
        with tempfile.TemporaryDirectory() as folder:
            fake=temporary_fake()
            make_receipt(folder,fake)
            Path(folder,"receipt.sha256").write_text("0"*64)
            with self.assertRaisesRegex(ValueError,"sidecar"):
                v.verify(folder,"nonexistent-source.jsonl","none","none")

    def test_failed_original_gpu_not_misrepresented_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            make_receipt(folder,{"status":v.FAILED,"failure_class":"RuntimeError"})
            result=v.verify(folder,"not-exists","not-exists","not-exists")
        self.assertEqual(result["status"],v.FAILED)
        self.assertFalse(result["model_release_qualified"])
