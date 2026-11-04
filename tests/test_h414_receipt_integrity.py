"""H414 refuses old, altered, or fabricated GPU receipts and data flags."""
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import unittest

from h414.validate_receipt import verify

ROOT=Path(__file__).resolve().parents[1]


class H414ReceiptIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h412=(ROOT/"h412/verified_receipt/"
                  "h412-owner-private-duplicate-curriculum-lora/receipt.json").read_bytes()

    def reject(self,raw,sidecar=None):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"receipt.json").write_bytes(raw)
            (p/"receipt.sha256").write_text(
                sha256(raw).hexdigest() if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                verify(p)

    def test_h412_192_step_receipt_cannot_claim_h414_thousands(self):
        self.reject(self.h412)

    def test_corrupt_original_sidecar_rejected(self):
        self.reject(self.h412,"0"*64)

    def test_relabeling_old_192_step_as_h414_is_not_real_training(self):
        data=json.loads(self.h412)
        data["schema"]="hala.h414.private-thousands-multisource-lora.v1"
        data["status"]="COMPLETE_PRIVATE_4768_EXAMPLE_MULTISOURCE_LORA"
        self.reject(json.dumps(data).encode())

    def test_false_release_ood_claim_cannot_pass(self):
        data=json.loads(self.h412)
        data["independent_external_ood"]=True
        self.reject(json.dumps(data).encode())

    def test_h414_private_train_inputs_and_no_eval_leakage(self):
        cfg=json.loads((ROOT/"h414/kaggle/kernel-metadata.json").read_text())
        self.assertIs(cfg["is_private"],True)
        self.assertEqual(cfg["dataset_sources"],
                         ["abdulazizshehri/hala-gui-training-library-private"])
        self.assertEqual(cfg["kernel_sources"],
                         ["abdulazizshehri/hala-h412-private-duplicate-curriculum-lora"])
        self.assertEqual(cfg["competition_sources"],[])

if __name__=="__main__":
    unittest.main()
