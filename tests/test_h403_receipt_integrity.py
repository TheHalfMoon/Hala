"""Private H403 GPU receipt integrity and no false Hala release approval."""
import json
import tempfile
import unittest
from pathlib import Path
from h403.validate_receipt import verify,ROOT

class H403OriginalReceiptTests(unittest.TestCase):
    def test_kaggle_real_result_has_correct_scope(self):
        r=verify()
        self.assertEqual(r["status"],"COMPLETED_PRIVATE_SYNTHETIC_ABSTENTION_LORA")
        self.assertEqual(r["parent"]["all"]["correct"],5)
        self.assertEqual(r["adapted"]["all"]["correct"],11)
        self.assertEqual(r["adapted"]["absent"]["false_clicks"],1)
    def test_modified_receipt_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/"receipt.json").write_bytes(
                (ROOT/"receipt.json").read_bytes()+b" ")
            (path/"receipt.sha256").write_bytes(
                (ROOT/"receipt.sha256").read_bytes())
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(path)
    def test_false_legal_ood_promotion_is_forbidden(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            receipt=json.loads((ROOT/"receipt.json").read_text("utf8"))
            receipt["independent_external_ood"]=True
            raw=json.dumps(receipt,sort_keys=True).encode()
            (path/"receipt.json").write_bytes(raw)
            (path/"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError,"illegal"):
                verify(path)
    def test_fake_bilingual_counts_are_forbidden(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            receipt=json.loads((ROOT/"receipt.json").read_text("utf8"))
            receipt["adapted_audit_generated"]["ar"]["correct"]=4
            raw=json.dumps(receipt,sort_keys=True).encode()
            (path/"receipt.json").write_bytes(raw)
            (path/"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError,"disagree"):
                verify(path)
if __name__=="__main__":
    unittest.main()
