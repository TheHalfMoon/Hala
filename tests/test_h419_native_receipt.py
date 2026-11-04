"""H419 private native-drag result receipt: fail-closed source/weights contract."""
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from h419 import validate_receipt as v
from h419.kaggle import h419_native_drag_continuation as m

ROOT=Path(__file__).resolve().parents[1]


class H419ReceiptSafety(unittest.TestCase):
    def prove_invalid(self,doc,sidecar=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            if isinstance(doc,bytes):
                raw=doc
            else:
                raw=json.dumps(doc,sort_keys=True).encode()
            (root/"receipt.json").write_bytes(raw)
            (root/"receipt.sha256").write_text(
                sha256(raw).hexdigest() if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                v.verify(root)

    def test_actual_h417_parent_receipt_cannot_pose_as_30k_continuation(self):
        root=ROOT/"h418/verified_receipt/h418-owner-original-30k-source/receipt.json"
        old=root.read_bytes()
        self.prove_invalid(old)

    def test_bad_sidecar_rejected_even_when_json_syntax_is_valid(self):
        fake={"status":"FAILED_PRIVATE_30000_NATIVE_DRAG_NO_RELEASE",
              "failure_class":"RuntimeError"}
        self.prove_invalid(fake,"0"*64)

    def test_genuine_error_is_never_reported_as_trained_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            state={"schema":v.SCHEMA,
                   "status":v.FAIL,
                   "failure_class":"RuntimeError",
                   "actual_optimizer_steps":200,
                   "model_release_qualified":False}
            raw=json.dumps(state).encode()
            (root/"receipt.json").write_bytes(raw)
            (root/"receipt.sha256").write_text(sha256(raw).hexdigest())
            evidence=v.verify(root)
            self.assertFalse(evidence["qualified"])
            self.assertEqual(evidence["status"],v.FAIL)
            self.assertNotIn("adapter_sha256",evidence)

    def test_fake_completed_large_model_cannot_pass_without_native_actions(self):
        fake={"schema":v.SCHEMA,"status":v.PASS,
              "completed_steps":9452,"train_unique_images":30976,
              "processed_training_examples":37808,
              "genuinely_independent_external_ood":False,
              "weights_public":False,"model_release_qualified":False}
        self.prove_invalid(fake)

    def test_forged_external_ood_and_rights_claims_are_rejected(self):
        doc={"schema":v.SCHEMA,"status":v.PASS,
             "genuinely_independent_external_ood":True,
             "independent_source_rights_signoff":True,
             "weights_public":True}
        self.prove_invalid(doc)

    def test_native_ordered_two_point_geometry_not_click_score(self):
        gold={"type":"drag","start_point":[105,310],
              "end_point":[910,600],"ordered":True,
              "coordinate_space":"normalized_1000"}
        self.assertTrue(m.score_predicted(json.dumps(gold),gold)["correct"])
        reversed=dict(gold,start_point=gold["end_point"],
                      end_point=gold["start_point"])
        self.assertFalse(m.score_predicted(json.dumps(reversed),gold)["correct"])
        unordered=dict(gold,ordered=False)
        swapped=dict(unordered,start_point=unordered["end_point"],
                     end_point=unordered["start_point"])
        self.assertTrue(m.score_predicted(json.dumps(swapped),unordered)["correct"])
        self.assertFalse(m.score_predicted('{"point_2d":[100,200]}',gold)["correct"])

if __name__=="__main__":
    unittest.main()
