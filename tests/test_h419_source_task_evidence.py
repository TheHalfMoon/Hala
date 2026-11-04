"""H419 source-family disjoint evidence receipt provenance regression tests."""
import json
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"h419/verified_receipt/h419-owner-source-task-isolation"


class H419SourceTaskEvidenceTests(unittest.TestCase):
    def test_full_original_task_audit_500554_actions_has_zero_split_leakage(self):
        raw=(RECEIPT/"receipt.json").read_bytes()
        checksum=(RECEIPT/"receipt.sha256").read_text("ascii").strip()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),checksum)
        self.assertEqual(checksum,
          "fff2cc485424f5263780cd234dcb93d302b2f5cb421206875f49d8df7dd6be3b")
        data=json.loads(raw)
        self.assertEqual(data["schema"],
                         "hala.h419.private-original-source-task-isolation.v1")
        self.assertEqual(data["original_h60_source_actions_scanned"],500554)
        self.assertEqual(data["original_source_images_with_train_annotations"],47037)
        self.assertEqual(data["train_new_original_screen_sha"],30000)
        self.assertEqual(data["old_dev_original_screen_sha"],96)
        self.assertEqual(data["fresh_dev_original_screen_sha"],96)
        self.assertEqual(data["original_task_sha_counts"],
            {"h419_train_original":30057,
             "h417_old_dev_original":96,
             "h419_new_dev_original":96})
        self.assertEqual(data["raw_source_action_records_for_selected_images"],
            {"h419_train_original":294177,
             "h417_old_dev_original":1336,
             "h419_new_dev_original":995})
        self.assertEqual(len(data["overlaps"]),3)
        self.assertTrue(all(item=={
            "image_sha_overlap":0,
            "original_source_task_sha_overlap":0
        } for item in data["overlaps"].values()))
        self.assertFalse(data["h417_heldout_used_for_training"])
        self.assertTrue(data["source_only_author_visible_holdout"])
        self.assertFalse(data["independent_external_ood"])
        self.assertFalse(data["model_training_performed_by_this_audit"])
        self.assertFalse(data["model_release_qualified"])

if __name__=="__main__":
    unittest.main()
