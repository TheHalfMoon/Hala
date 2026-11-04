"""H413 original native audit validation cannot be satisfied by old receipts."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from h413.validate_receipt import verify

class AuditEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=(Path(__file__).resolve().parents[1]/
              "h412/verified_receipt/h412-owner-private-duplicate-curriculum-lora/receipt.json")
        cls.h412=path.read_bytes()

    def reject(self,data,sidecar=None):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            (p/"receipt.json").write_bytes(data)
            (p/"receipt.sha256").write_text(sha256(data).hexdigest()
                                             if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                verify(root)

    def test_original_verified_new_family_receipt(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow required for visual-source replay")
        proof=verify()
        self.assertEqual(proof["receipt_sha256"],
            "1280b842e5b7b6729992f77bf44f20990363c4a90e7475cb935f0205f926b53b")
        self.assertFalse(proof["qualified"])
        self.assertEqual(proof["h409"]["all"]["correct"],85)
        self.assertEqual(proof["h412"]["all"]["correct"],92)
        self.assertEqual(proof["h412"]["duplicate"]["unsafe_clicks"],0)
        self.assertEqual(proof["h412"]["unique"]["correct"],28)
        self.assertIn("NOT external OOD",proof["source"])

    def test_h412_training_receipt_is_not_new_crossfamily_eval(self):
        self.reject(self.h412)

    def test_mutated_sha_fails(self):
        self.reject(self.h412,"f"*64)

    def test_falsely_renamed_training_receipt_still_fails(self):
        v=json.loads(self.h412)
        v["schema"]="hala.h413.private-original-new-family-eval.v1"
        v["status"]="COMPLETE_PRIVATE_AUTHOR_VISIBLE_NEW_FAMILY_EVAL_NO_RELEASE"
        v["training_performed"]=False
        v["model_release_qualified"]=False
        self.reject(json.dumps(v).encode())

    def test_wrong_external_ood_claim_fails(self):
        v=json.loads(self.h412)
        v["independent_external_ood"]=True
        self.reject(json.dumps(v).encode())

if __name__=="__main__":
    unittest.main()
