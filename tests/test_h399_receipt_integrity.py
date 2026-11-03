"""H399 owner-Kaggle read-only receipt integrity; never loads model weights."""
import tempfile
import unittest
from pathlib import Path
from h399.validate_receipt import ROOT,verify

class H399ReceiptTests(unittest.TestCase):
    def test_source_bounded_private_completed_receipt(self):
        out=verify()
        self.assertEqual(out["optimizer_steps"],16)
        self.assertEqual(out["status"],
                         "COMPLETED_REAL_2B_SYNTHETIC_RESEARCH_LORA")
    def test_tampered_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=(ROOT/"receipt.json").read_bytes()
            (root/"receipt.json").write_bytes(original+b" ")
            (root/"receipt.sha256").write_bytes(
                (ROOT/"receipt.sha256").read_bytes())
            with self.assertRaisesRegex(ValueError,"SHA|digest"):
                verify(root)
if __name__=="__main__":
    unittest.main()
