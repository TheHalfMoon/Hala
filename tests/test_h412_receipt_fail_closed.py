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
