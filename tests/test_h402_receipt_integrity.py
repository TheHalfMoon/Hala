"""H402 research receipt integrity prevents fabricated independent OOD claims."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h402.validate_receipt import verify

def example():
    def values(n,correct,invalid=0,false_clicks=0,abstentions=0):
        return dict(n=n,correct=correct,invalid=invalid,
                    false_clicks=false_clicks,abstentions=abstentions)
    arm={"all":values(24,6,0,0,0),
         "en":values(12,3),
         "ar":values(12,3),
         "positive":values(12,6),
         "absent":values(12,0)}
    return {"schema":"hala.h402.private-author-negative-native-v1",
            "status":"COMPLETE_AUTHOR_SYNTHETIC_STRESS_ONLY",
            "base_model":"Qwen/Qwen3-VL-2B-Instruct",
            "base_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
            "adapter_sha256":"8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537",
            "evaluator_is_independent":False,"real_app_gui":False,
            "source_rights_approved":False,"externally_sealed_ood":False,
            "model_weights_published":False,"training_performed_in_this_run":False,
            "n":24,"device":"Tesla T4","base":arm,"adapted":arm}

def save(path,data):
    raw=json.dumps(data,sort_keys=True).encode()
    (path/"receipt.json").write_bytes(raw)
    (path/"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())

class H402ReceiptTests(unittest.TestCase):
    def test_can_validate_author_scoped_complete_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root,example())
            self.assertEqual(verify(root)["base"]["all"]["n"],24)
    def test_independent_blind_ood_claim_is_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=example()
            data["externally_sealed_ood"]=True
            save(root,data)
            with self.assertRaisesRegex(ValueError,"fabricated"):
                verify(root)
    def test_inconsistent_language_counts_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=example()
            data["adapted"]["ar"]["correct"]=12
            save(root,data)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                verify(root)
    def test_tampered_sha_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root,example())
            (root/"receipt.json").write_bytes((root/"receipt.json").read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(root)
if __name__=="__main__":
    unittest.main()
