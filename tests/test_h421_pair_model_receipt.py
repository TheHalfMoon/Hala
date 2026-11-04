"""H421 source-replayed actual GPU paired two-point DRAG receipt checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from h420 import counterfactual_score as scores
from h420 import validate_pair_model_receipt as v
from h419.kaggle import h419_native_drag_continuation as base


def save(root,doc):
    payload=json.dumps(doc,sort_keys=True).encode()
    (root/"receipt.json").write_bytes(payload)
    (root/"receipt.sha256").write_text(hashlib.sha256(payload).hexdigest())


class H421OriginalGPUReceiptTests(unittest.TestCase):
    def wrong(self,doc):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save(root,doc)
            with self.assertRaises(ValueError):
                v.verify(root,root/"labels.jsonl",
                         root/"source.json",root/"source.sha256")

    def test_forged_complete_with_no_genuine_native_predictions_rejected(self):
        self.wrong({"schema":v.SCHEMA,"status":v.RESEARCH_STATUS,
                    "h419_real_optimizer_steps":9452,
                    "model_training_executed":False,
                    "independent_external_ood":False,
                    "model_release_qualified":False})

    def test_failed_original_gpu_status_not_reported_as_model_qualified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save(root,{"schema":v.SCHEMA,"status":v.FAIL,
                       "failure_class":"RuntimeError",
                       "independent_external_ood":False,
                       "model_release_qualified":False})
            actual=v.verify(root,root/"absent.jsonl",
                            root/"absent_source",root/"absent_sha")
            self.assertFalse(actual["qualified"])
            self.assertNotIn("h419_actual_adapter_sha256",actual)
            self.assertEqual(actual["status"],v.FAIL)

    def test_fake_public_release_claim_never_valid(self):
        self.wrong({"schema":v.SCHEMA,"status":v.RESEARCH_STATUS,
                    "model_training_executed":True,
                    "independent_external_ood":True,
                    "model_release_qualified":True,
                    "weights_public":True})

    def test_actual_private_384_source_pair_oracle_replay_checks_false_scores(self):
        folder=os.environ.get("H421_OWNER_LABEL_ROOT")
        if not folder:
            self.skipTest("Owner PRIVATE source-label files not mounted in hosted CI")
        folder=Path(folder)
        pairs=scores.load_sources(
           folder/"h421_original_same_image_two_instructions.jsonl",
           folder/"h421_source_counterfactual_receipt.json",
           folder/"receipt.sha256")
        rows=[r for i in range(192) for r in (pairs[i][0],pairs[i][1])]
        original=[{"id":r["id"],"image_sha256":r["image_sha256"],
                   "raw_prediction":json.dumps(pairs[r["pair_id"]][0]["native_drag_target"])}
                  for r in rows]
        corrected=[{"id":r["id"],"image_sha256":r["image_sha256"],
                    "raw_prediction":json.dumps(r["native_drag_target"])}
                   for r in rows]
        prev=scores.calculate(rows,original)
        after=scores.calculate(rows,corrected)
        self.assertEqual(prev["both_actions_correct"],0)
        self.assertEqual(after["both_actions_correct"],192)
        fixture={
           "schema":v.SCHEMA,"status":v.RESEARCH_STATUS,
           "official_base":base.BASE,"official_revision":base.REV,
           "h417_parent_adapter_sha256":base.PARENT_ADAPTER_SHA,
           "source_pairs_jsonl_sha256":scores.SOURCE_JSONL_SHA,
           "source_pairs_receipt_sha256":scores.SOURCE_RECEIPT_SHA,
           "h419_real_optimizer_steps":base.H419_STEPS,
           "h419_real_gradient_examples":base.H419_TRAINING_PASSES,
           "paired_original_images":scores.IMAGE_PAIRS,
           "original_source_actions":scores.SAMPLE_N,
           "device_name":"Tesla T4",
           "model_training_executed":False,
           "independent_external_ood":False,
           "model_release_qualified":False,
           "weights_public":False,
           "h419_train_source_only_research_gate":True,
           "research_pair_improved":True,
           "h419_trained_adapter_sha256":"a"*64,
           "h419_trained_private_receipt_sha256":"b"*64,
           "h417_parent_pair_stats":prev,
           "h419_trained_pair_stats":after,
           "h417_parent_exact_predictions":original,
           "h419_trained_exact_predictions":corrected,
        }
        with tempfile.TemporaryDirectory() as tmp:
            receipt_root=Path(tmp)
            save(receipt_root,fixture)
            actual=v.verify(receipt_root,
                folder/"h421_original_same_image_two_instructions.jsonl",
                folder/"h421_source_counterfactual_receipt.json",
                folder/"receipt.sha256")
            self.assertTrue(actual["original_192_pairs_replayed"])
            self.assertTrue(actual["research_pair_improved"])
            self.assertFalse(actual["qualified"])
            # Deliberately falsify one reported pair metric. Byte receipt
            # sidecar still valid, but recomputed actual predictions MUST fail.
            fixture["h419_trained_pair_stats"]["both_actions_correct"]=191
            save(receipt_root,fixture)
            with self.assertRaisesRegex(ValueError,"forged"):
                v.verify(receipt_root,
                    folder/"h421_original_same_image_two_instructions.jsonl",
                    folder/"h421_source_counterfactual_receipt.json",
                    folder/"receipt.sha256")

if __name__=="__main__":
    unittest.main()
