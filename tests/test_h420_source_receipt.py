"""H420 actual owner-private complementary DRAG source receipt regression."""
import hashlib
import json
from pathlib import Path
import unittest

from h420 import curate_alternative_drags as h

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"h420/verified_receipt/h420-owner-authentic-second-drags"


class H420ActualOwnerSourceEvidence(unittest.TestCase):
    def test_actual_selected_29827_source_jpeg_bytes_all_proven(self):
        raw=(RECEIPT/"receipt.json").read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        self.assertEqual(digest,
            "ae32019e4ca81213cab3cc4e7135a2b09b68ee54e809dd0a553eacad14c51526")
        self.assertEqual((RECEIPT/"receipt.sha256").read_text("ascii").strip(),digest)
        state=json.loads(raw)
        self.assertEqual(state["schema"],
            "hala.h420.private-authentic-second-native-drags.v1")
        self.assertEqual(state["status"],
            "ORIGINAL_COMPLEMENTARY_SOURCE_BYTES_VERIFIED_MODEL_NOT_TRAINED")
        self.assertEqual(state["original_first_30000_source_sha256"],h.H418_ROWS_SHA)
        self.assertEqual(state["original_first_30000_source_receipt_sha256"],h.H418_RECEIPT_SHA)
        self.assertEqual(state["original_task_heldout_isolation_receipt_sha256"],
                         h.H419_SOURCE_TASK_AUDIT_SHA)
        self.assertEqual(state["original_source_second_actions_jsonl_sha256"],
            "58c88bf09c047973675ca6dd906eae8d6832005f5da338d492a67c8dd35b91ed")
        self.assertEqual(state["total_original_actions_scanned"],500554)
        self.assertEqual(state["new_original_distinct_complementary_native_actions"],29827)
        self.assertEqual(state["new_actions_with_different_instruction_text"],29827)
        self.assertEqual(state["source_images_without_native_alternative"],173)
        self.assertEqual(state["new_actions_geometric_ratio_at_least_0_08"],27769)
        self.assertEqual(state["original_source_second_actions_by_domain"],h.FROZEN_COUNT_BY_DOMAIN)
        self.assertTrue(state["all_selected_original_JPEG_bytes_SHA_verified"])
        self.assertTrue(state["all_actions_derived_from_original_train_only"])
        self.assertTrue(state["old_h417_and_new_h419_original_dev_tasks_quarantined"])
        self.assertFalse(state["model_training_executed"])
        self.assertFalse(state["external_independent_ood"])
        self.assertFalse(state["model_release_qualified"])
        self.assertFalse(state["weights_public"])

if __name__=="__main__":
    unittest.main()
