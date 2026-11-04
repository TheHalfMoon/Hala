"""H421 original same-image two-task source DEV receipt (not external OOD)."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h420/verified_receipt/h421-original-counterfactual-dev"


class H421SameImageCounterfactualReceipt(unittest.TestCase):
    def test_192_true_original_jpeg_pairs_and_384_actions_isolated(self):
        raw=(SOURCE/"receipt.json").read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        self.assertEqual(digest,
            "02a4d17ead71fc06019a83669a04c9779430de5a7a48f800429ad2241d2ebaf4")
        self.assertEqual((SOURCE/"receipt.sha256").read_text("ascii").strip(),digest)
        original=json.loads(raw)
        self.assertEqual(original["schema"],
            "hala.h421.source-original-counterfactual-dev.v1")
        self.assertEqual(original["source_original_action_counts"],500554)
        self.assertEqual(original["owner_heldout_source_image_pairs"],192)
        self.assertEqual(original["native_two_instruction_eval_cases"],384)
        self.assertEqual(original["native_pairs_by_source_family"],
            {"slider":64,"slide_resize":64,"text_highlight":64})
        self.assertEqual(original["frozen_original_dev_rows_sha256"],
            "42d93219ec4083920462d4194b175154f1f12d9b3e7b36ffd1068d8f8a4d96b9")
        self.assertGreaterEqual(original["min_source_coord_target_ratio"],0.15)
        self.assertTrue(original["excluded_h417_and_h419_train_and_dev_source_images"])
        self.assertTrue(original["excluded_source_task_sha_of_h417_h419_models_and_dev"])
        self.assertTrue(original["source_target_different_instructions_and_coordinates"])
        self.assertTrue(original["original_jpeg_media_sha_replayed"])
        self.assertFalse(original["independent_external_ood"])
        self.assertFalse(original["model_training_executed"])
        self.assertFalse(original["model_evaluation_executed"])
        self.assertFalse(original["model_release_qualified"])

if __name__=="__main__":
    unittest.main()
