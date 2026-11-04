"""Actual OWNER source replay H422 original 16K same-JPEG paired TRAIN receipt."""
import hashlib
import json
from pathlib import Path
import unittest
from h422 import paired_source as p

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"h422/verified_receipt/h422-owner-original-16k-paired-actions"

class ActualH422PairedSourceEvidence(unittest.TestCase):
    def test_all_32000_source_JPEG_reads_and_two_actions_recorded_not_trained(self):
        raw=(RECEIPT/"receipt.json").read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        self.assertEqual(sha,"229097839788875059607e4934418c3c940ef32620c46083b38261aae061cf87")
        self.assertEqual((RECEIPT/"receipt.sha256").read_text("ascii").strip(),sha)
        receipt=json.loads(raw)
        self.assertEqual(receipt["schema"],
            "hala.h422.owner-private-original-16k-paired-native-drag-train.v1")
        self.assertEqual(receipt["status"],
            "SOURCE_VERIFIED_32000_PAIRED_NATIVE_DRAGS_MODEL_NOT_TRAINED")
        self.assertEqual(receipt["genuine_unique_original_JPEG_train"],16000)
        self.assertEqual(receipt["genuine_original_image_conditional_action_labels"],32000)
        self.assertEqual(receipt["genuine_paired_original_JPEG_media_reads"],32000)
        self.assertEqual(receipt["distinct_original_instructions_per_image"],2)
        self.assertEqual(receipt["original_train_by_domain"],p.PER_DOMAIN)
        self.assertEqual(receipt["original_training_action_rows_sha256"],
           "b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa")
        self.assertEqual(receipt["second_29827_original_action_rows_sha256"],
                         p.H420_SECOND_ROWS_SHA)
        self.assertEqual(receipt["untouched_H421_eval_384_source_rows_sha256"],
                         p.H421_UNSEEN_ROWS_SHA)
        self.assertTrue(receipt["untouched_H421_eval_192_original_screens_excluded"])
        self.assertTrue(receipt["untouched_H417_H419_dev_and_source_task_excluded"])
        self.assertTrue(receipt["real_original_media_SHA_replayed"])
        self.assertEqual(len(receipt["original_H60_nine_ZIP_sha256"]),9)
        self.assertFalse(receipt["model_training_executed"])
        self.assertFalse(receipt["independent_external_ood"])
        self.assertFalse(receipt["model_release_qualified"])
        self.assertFalse(receipt["weights_public"])

if __name__=="__main__":
    unittest.main()
