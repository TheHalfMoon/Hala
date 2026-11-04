"""H412 receipt must reject H409, mutated hashes and unsupported release claims."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from h412.validate_receipt import verify


class H412ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = (Path(__file__).resolve().parents[1] / "h409"
                      / "verified_receipt" / "h409-owner-private-balanced-ui-lora"
                      / "receipt.json").read_bytes()

    def _expect_refused(self, original: bytes, sidecar: str | None = None):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            (path/"receipt.json").write_bytes(original)
            (path/"receipt.sha256").write_text(
                sha256(original).hexdigest() if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                verify(path)

    def test_real_verified_h412_receipt_is_valid_and_source_only(self):
        proof=verify()
        self.assertEqual(proof["sha256"], "fa0f8b3350513d3744fa6bed882fc4907b375a28c0c67a847c3d8ae62c424574")
        self.assertEqual(proof["status"], "COMPLETE_PRIVATE_DUPLICATE_CURRICULUM_192_STEP_LORA")
        self.assertTrue(proof["toy_safety_gate"])
        self.assertEqual(proof["changed_lora_tensors"], 112)
        self.assertEqual(proof["adapter_sha256"], "fa47ff52f67dd39b7aef205351e71551716b66da55cda3b4370c981189c13066")
        self.assertEqual(proof["h412"]["all"]["correct"], 48)
        self.assertEqual(proof["h412"]["ambiguous"]["unsafe_clicks"], 0)
        self.assertIn("no independent OOD", proof["source"])

    def test_previous_h409_is_not_h412_training(self):
        self._expect_refused(self.parent)

    def test_sha_tamper_fails_before_parse(self):
        self._expect_refused(self.parent, "0"*64)

    def test_renamed_parent_receipt_is_not_independent_h412_proof(self):
        doc=json.loads(self.parent)
        doc["schema"]="hala.h412.private-duplicate-curriculum-lora.v1"
        doc["status"]="COMPLETE_PRIVATE_DUPLICATE_CURRICULUM_192_STEP_LORA"
        doc["parent_adapter_sha256"]="0ce00eec423a80938344048f69f9b50734b0f5b52cae646767896bffde1d3dbc"
        self._expect_refused(json.dumps(doc).encode())

    def test_release_claim_on_fabricated_status_rejected(self):
        doc=json.loads(self.parent)
        doc["official_hala_model_trained"]=True
        self._expect_refused(json.dumps(doc).encode())


if __name__ == "__main__":
    unittest.main()
