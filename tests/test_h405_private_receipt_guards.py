"""No-ML H405 source-only safety receipt adversarial validation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h405.validate_receipt import ADAPTERS,verify

def fabricated():
    def group(n):
        return {"n":n,"correct":0,"invalid":n,"unsafe_clicks":0,
                "abstentions":0,"valid_points":0}
    metric={"all":group(72),"en":group(36),"ar":group(36),
            "unique":group(24),"absent":group(24),"ambiguous":group(24)}
    return {
        "schema":"hala.h405.private-invented-ambiguity-anti-click-v1",
        "status":"COMPLETE_FIRST_PARTY_SYNTHETIC_SAFETY_STRESS",
        "official_base":"Qwen/Qwen3-VL-2B-Instruct",
        "base_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
        "adapter_sha256":ADAPTERS,
        "original_only":True,"independent_ood":False,
        "original_gui_training_rights_approved":False,
        "official_hala_model_qualified":False,
        "new_training_performed":False,
        "weights_published":False,
        "pass_safety_gate":False,
        "n":72,"device_name":"Tesla T4",
        "original_cases_sha256":"a"*64,
        "h399":metric,"h403":metric}

def save(folder,record):
    data=json.dumps(record,sort_keys=True).encode()
    (folder/"receipt.json").write_bytes(data)
    (folder/"receipt.sha256").write_text(hashlib.sha256(data).hexdigest())

class H405ReceiptGuards(unittest.TestCase):
    def test_valid_zero_score_owner_visible_research_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            save(p,fabricated())
            r=verify(p)
            self.assertFalse(r["synthetic_safety_gate"])
            self.assertEqual(r["h403"]["all"]["invalid"],72)
    def test_forged_legal_or_hidden_ood_status_is_rejected(self):
        for key in ("independent_ood","original_gui_training_rights_approved",
                    "official_hala_model_qualified","weights_published"):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)
                doc=fabricated()
                doc[key]=True
                save(p,doc)
                with self.assertRaisesRegex(ValueError,"rights/real OOD"):
                    verify(p)
    def test_forged_native_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            doc=fabricated()
            doc["h403"]["all"]["correct"]=71
            save(p,doc)
            with self.assertRaisesRegex(ValueError,"impossible|inconsistent"):
                verify(p)
    def test_forged_synthetic_safety_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            doc=fabricated()
            doc["pass_safety_gate"]=True
            save(p,doc)
            with self.assertRaisesRegex(ValueError,"safety gate"):
                verify(p)
    def test_modified_receipt_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            save(p,fabricated())
            (p/"receipt.json").write_bytes((p/"receipt.json").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(p)

if __name__=="__main__":
    unittest.main()
