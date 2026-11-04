"""Fail-closed H417 native drag receipt and source-integrity acceptance."""
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from h417.validate_receipt import verify

ROOT=Path(__file__).resolve().parents[1]

class H417ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=(ROOT/"h412/verified_receipt/h412-owner-private-duplicate-curriculum-lora/receipt.json").read_bytes()

    def reject(self,raw,sidecar=None):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/"receipt.json").write_bytes(raw)
            (path/"receipt.sha256").write_text(sha256(raw).hexdigest()
                                    if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                verify(path)

    def test_old_click_lora_is_not_native_drag_proof(self):
        self.reject(self.old)

    def test_owner_tamper_mismatched_hash_is_refused(self):
        self.reject(self.old,"0"*64)

    def test_fake_12k_claim_from_unrelated_receipt_refused(self):
        doc=json.loads(self.old)
        doc["schema"]="hala.h417.private-12k-native-drag.v1"
        doc["status"]="COMPLETE_PRIVATE_NATIVE_DRAG_14880_PASSES"
        doc["train_unique_images"]=11952
        doc["processed_training_examples"]=14880
        doc["completed_steps"]=3720
        self.reject(json.dumps(doc).encode())

    def test_external_independent_ood_claim_cannot_pass(self):
        doc=json.loads(self.old)
        doc["independent_external_ood"]=True
        self.reject(json.dumps(doc).encode())

    def test_strict_native_drag_result_scores_ignore_reversed_ordered(self):
        from h417.kaggle.h417_native_drag_lora import score_predicted
        task={"type":"drag","start_point":[145,300],
              "end_point":[755,300],"ordered":True,
              "coordinate_space":"normalized_1000"}
        swapped=dict(task,start_point=task["end_point"],
                          end_point=task["start_point"])
        self.assertTrue(score_predicted(json.dumps(task),task)["correct"])
        self.assertFalse(score_predicted(json.dumps(swapped),task)["correct"])
        unordered=dict(task,ordered=False)
        swapped_unordered=dict(unordered,start_point=unordered["end_point"],
                                    end_point=unordered["start_point"])
        self.assertTrue(score_predicted(json.dumps(swapped_unordered),unordered)["correct"])

if __name__=="__main__":
    unittest.main()
