"""H406 source-only anti-fabrication tests; no Kaggle/PyTorch needed."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h406.validate_receipt import verify

def failure_receipt():
    return {
        "schema":"hala.h406.private-first-party-ambiguous-refusal-lora.v1",
        "model":"Qwen/Qwen3-VL-2B-Instruct",
        "base_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
        "parent_adapter_sha256":"01f1c1846c76a5e1e785fcc431faed5ee2774638a0c46d84fedfbd2486ebcb4b",
        "train_source":"first-party algorithmic source-only rectangles",
        "train_uses_h405_eval":False,
        "independent_external_ood":False,
        "original_gui_train_rights_approved":False,
        "official_hala_trained":False,
        "public_weights":False,
        "status":"FAILED_PRIVATE_SYNTHETIC_NO_RELEASE",
        "synthetic_safety_gate_pass":False,
        "failure_class":"ValueError",
        "optimizer_steps":0,
    }

def write(path,obj):
    data=json.dumps(obj,sort_keys=True).encode()
    (path/"receipt.json").write_bytes(data)
    (path/"receipt.sha256").write_text(hashlib.sha256(data).hexdigest())

class H406SourceReceiptTests(unittest.TestCase):
    def test_original_owner_kaggle_training_and_unsafe_stress_receipt(self):
        data=verify()
        self.assertEqual(data["status"],
                         "COMPLETE_PRIVATE_FIRST_PARTY_AMBIGUITY_LORA")
        self.assertEqual(data["h403"]["ambiguous"]["unsafe_clicks"],13)
        self.assertEqual(data["h406"]["ambiguous"]["unsafe_clicks"],2)
        self.assertEqual(data["h406"]["unique"]["correct"],11)
        self.assertFalse(data["toy_safety_gate"])
        self.assertEqual(data["adapter_sha256"],
             "26ac3b14380eb7dfb1709d67b771b7bfe3d383ec37d02e3cae5b1c48197936b6")

    def test_failed_gpu_receipt_is_not_a_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p,failure_receipt())
            doc=verify(p)
            self.assertEqual(doc["status"],"FAILED_PRIVATE_SYNTHETIC_NO_RELEASE")
            self.assertFalse(doc["toy_safety_gate"])
    def test_tampered_receipt_fails_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p,failure_receipt())
            (p/"receipt.json").write_bytes(
                (p/"receipt.json").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(p)
    def test_forged_rights_or_ood_flags_always_refused(self):
        for flag in ("train_uses_h405_eval","independent_external_ood",
                     "original_gui_train_rights_approved",
                     "official_hala_trained","public_weights"):
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)
                doc=failure_receipt()
                doc[flag]=True
                write(p,doc)
                with self.assertRaisesRegex(ValueError,"forged"):
                    verify(p)
    def test_gpu_failure_never_passes_safety_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            doc=failure_receipt()
            doc["synthetic_safety_gate_pass"]=True
            write(p,doc)
            with self.assertRaisesRegex(ValueError,"failed experiment"):
                verify(p)
    def test_fabricated_complete_without_real_steps_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            doc=failure_receipt()
            doc["status"]="COMPLETE_PRIVATE_FIRST_PARTY_AMBIGUITY_LORA"
            write(p,doc)
            with self.assertRaisesRegex(ValueError,"96-step"):
                verify(p)

if __name__=="__main__":
    unittest.main()
