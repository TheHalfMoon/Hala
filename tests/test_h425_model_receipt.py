"""H425 receipt validator NEGATIVE tests; fake metrics are synthetic ONLY.

No synthetic 12696-step fake is training evidence. Owner provider receipts and
original media never enter GitHub CI.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("h425_receipt_validator",
    ROOT/"h425/validate_receipt.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
h=v.MODEL

def false_fixture():
    """Synthetic temporary counterfeit used ONLY to test rejection boundaries."""
    d={
        "schema":h.H425_SCHEMA,"status":h.H425_DONE,
        "official_base":h.BASE,"official_revision":h.REV,
        "h417_source_parent_sha256":h.PARENT_ADAPTER_SHA,
        "h417_source_parent_receipt_sha256":h.PARENT_RECEIPT_SHA,
        "h418_new_original_30k_source_manifest_sha256":h.H418_RECEIPT_SHA,
        "h422_original_pair_rows_sha256":h.H422_ROWS_SHA,
        "h422_original_paired_source_receipt_sha256":h.H422_RECEIPT_SHA,
        "h421_reserved_counterfactual_dev_sha256":h.H421_DEV_SHA,
        "original_nine_H60_source_zip_SHA256":h.H419_ALL_SOURCE_SHAS,
        "independent_external_ood":False,
        "source_rights_review_qualified":False,
        "release_qualified":False,
        "weights_public":False,
        "replaced_failed_h419_parent":True,
        "h425_unique_original_training_screens":27952,
        "h425_genuine_distinct_native_action_rows":43952,
        "h425_original_same_screenshot_two_instructions":16000,
        "h425_original_paired_optimizer_steps":8000,
        "h425_old_h417_english_replay_examples":10976,
        "h425_old_h417_arabic_replay_unique":976,
        "h425_old_h417_arabic_passes":8,
        "h425_gradients_expected":50784,
        "h425_optimizer_steps_expected":12696,
        "untouched_old_h417_and_fresh_h419_original_dev_images":240,
        "actual_optimizer_steps":12696,
        "completed_optimizer_steps":12696,
        "processed_training_examples":50784,
        "actual_50784_gradient_examples":50784,
        "device_name":"Tesla T4",
        "changed_lora_tensors":2,
        "model_weights_sha256":"a"*64,
        "model_weights_bytes":13000,
        "elapsed_seconds":2.5,
        "weighted_loss_history":[0.9]*12696,
        "private_training_checkpoints_sha256":{
          str(step):{"optimizer_step":step,"model_weights_bytes":13000,
                     "model_weights_sha256":hashlib.sha256(
                         f"SYNTHETIC_NOT_REAL_CHECKPOINT_{step}".encode()).hexdigest()}
          for step in range(1500,12697,1500)},
        "last_private_checkpoint_step":12000,
        "original_dev_old_parent_correct":76,
        "original_dev_old_after_correct":71,
        "original_dev_fresh_parent_correct":18,
        "original_dev_fresh_after_correct":30,
        "source_only_H419_frozen_research_gate":False,
        "source_only_H425_stronger_improvement_gate":False,
    }
    before=[];after=[]
    for i in range(240):
        ar=i<24
        valid=True
        prev=(i<76 if i<144 else i<162)
        new=(i<71 if i<144 else i<174)
        base={"id":f"synthetic-case-{i}","family":"firstparty" if ar else "original",
              "domain":"slider" if i%2 else "slide_resize",
              "language":"ar" if ar else "en",
              "image_sha256":hashlib.sha256(f"NOT_REAL_SOURCE_IMAGE_{i}".encode()).hexdigest(),
              "raw_prediction":"{}","valid":valid}
        before.append({**base,"correct":prev,"error":None if prev else "coordinates"})
        after.append({**base,"correct":new,"error":None if new else "coordinates"})
    d["h417_parent_dev_exact"]=before
    d["h425_after_dev_exact"]=after
    d["h417_parent_dev"]=v._metrics(before)
    d["h425_after_dev"]=v._metrics(after)
    d["original_dev_parent_ar_correct"]=d["h417_parent_dev"]["ar"]["correct"]
    d["original_dev_after_ar_correct"]=d["h425_after_dev"]["ar"]["correct"]
    return d

def write_receipt(folder,data):
    raw=json.dumps(data,sort_keys=True).encode("utf8")
    Path(folder,"receipt.json").write_bytes(raw)
    Path(folder,"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())

class H425FailClosedReceiptContract(unittest.TestCase):
    def test_synthetic_counterfeit_is_research_fail_not_release(self):
        d=false_fixture()
        with tempfile.TemporaryDirectory() as directory:
            write_receipt(directory,d)
            state=v.verify(directory)
        self.assertEqual((state["old_parent_correct"],state["fresh_parent_correct"]),(76,18))
        self.assertEqual((state["old_after_correct"],state["fresh_after_correct"]),(71,30))
        self.assertFalse(state["original_source_only_gate"])
        self.assertFalse(state["model_release_qualified"])
        self.assertFalse(state["trained_adapter_sha_verified"])
        self.assertFalse(state["original_source_media_replayed"])

    def test_claiming_false_gate_as_true_rejected(self):
        d=false_fixture();d["source_only_H419_frozen_research_gate"]=True
        with self.assertRaisesRegex(ValueError,"gate"):
            v._eval_contract(d)

    def test_wrong_optimizer_budget_and_loss_count_rejected(self):
        for field,bad in (("actual_optimizer_steps",12695),
                          ("processed_training_examples",50780),
                          ("weighted_loss_history",[0.1]*4)):
            with self.subTest(field=field):
                d=false_fixture();d[field]=bad
                with self.assertRaises(ValueError):v._eval_contract(d)

    def test_model_private_safety_claims_fail_closed(self):
        for field in ("independent_external_ood",
                       "source_rights_review_qualified",
                       "release_qualified","weights_public"):
            with self.subTest(field=field):
                d=false_fixture();d[field]=True
                with self.assertRaisesRegex(ValueError,"release guard"):
                    v._eval_contract(d)

    def test_corrupt_case_or_score_rejected(self):
        d=false_fixture();d["h425_after_dev_exact"][0]["correct"]=False
        with self.assertRaisesRegex(ValueError,"aggregate"):
            v._eval_contract(d)

    def test_old_h417_parent_baseline_not_mutable(self):
        d=false_fixture()
        d["h417_parent_dev_exact"][0]["correct"]=False
        d["h417_parent_dev"]=v._metrics(d["h417_parent_dev_exact"])
        d["original_dev_old_parent_correct"]=75
        d["original_dev_parent_ar_correct"]=d["h417_parent_dev"]["ar"]["correct"]
        with self.assertRaisesRegex(ValueError,"base failed"):
            v._eval_contract(d)

    def test_receipt_sha_sidecar_tamper_rejected(self):
        d=false_fixture()
        with tempfile.TemporaryDirectory() as directory:
            write_receipt(directory,d)
            Path(directory,"receipt.sha256").write_text("f"*64)
            with self.assertRaisesRegex(ValueError,"sidecar"):
                v.verify(directory)

    def test_private_trained_weights_must_match_exact_sha(self):
        d=false_fixture()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory)/"adapter_model.safetensors"
            tmp.write_bytes(b"fake-safetensors-not-actual-weights")
            with self.assertRaisesRegex(ValueError,"safetensors SHA or size"):
                v._safetensors_sha(tmp,d)

    def test_failed_owner_gpu_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_receipt(directory,{"status":h.H425_FAIL,"failure_class":"ValueError"})
            info=v.verify(directory)
        self.assertFalse(info["source_only_qualified"])
        self.assertFalse(info["model_release_qualified"])

    def test_complete_requires_all_private_source_args_to_replay(self):
        d=false_fixture()
        with tempfile.TemporaryDirectory() as directory:
            write_receipt(directory,d)
            with self.assertRaisesRegex(ValueError,"requires private"):
                v.verify(directory,source_root=directory)

    def test_original_intermediate_checkpoint_sha_and_optimizer_step_rejected(self):
        d=false_fixture()
        del d["private_training_checkpoints_sha256"]["3000"]
        with self.assertRaisesRegex(ValueError,"all eight"):
            v._eval_contract(d)
        d=false_fixture()
        d["private_training_checkpoints_sha256"]["3000"]["optimizer_step"]=3001
        with self.assertRaisesRegex(ValueError,"checkpoint SHA/step"):
            v._eval_contract(d)
        d=false_fixture()
        d["private_training_checkpoints_sha256"]["3000"]["model_weights_sha256"]="bad"
        with self.assertRaisesRegex(ValueError,"checkpoint SHA/step"):
            v._eval_contract(d)

    def test_checkpoint_archive_actual_bytes_absent_not_verified(self):
        d=false_fixture()
        with tempfile.TemporaryDirectory() as directory:
            write_receipt(directory,d)
            state=v.verify(directory)
            self.assertFalse(state["all_eight_intermediate_checkpoint_shas_verified"])
            with self.assertRaisesRegex(ValueError,"safetensors SHA or size"):
                v.verify(directory,checkpoints_root=directory)
