"""H408 source-only SHA and no fake UI/OOD release receipt controls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h408.validate_receipt import ADAPTERS,verify,ROOT

def failed():
    return {
        "schema":"hala.h408.private-first-party-ui-composition-transfer.v1",
        "status":"FAILED_NO_MODEL_QUALIFICATION",
        "official_base":"Qwen/Qwen3-VL-2B-Instruct",
        "base_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
        "parent_adapters":ADAPTERS,
        "source_family":"first-party newly authored GUI-like button/cards/sidebar",
        "external_ood":False,"third_party_gui_data":False,
        "original_gui_train_rights_approved":False,
        "new_training":False,"model_weights_published":False,
        "official_hala_model_qualified":False,
        "failure_class":"ValueError",
    }

def save(folder,doc):
    binary=json.dumps(doc,sort_keys=True).encode()
    (folder/"receipt.json").write_bytes(binary)
    (folder/"receipt.sha256").write_text(hashlib.sha256(binary).hexdigest())

class H408ReceiptSourceTests(unittest.TestCase):
    def test_real_owner_kaggle_v2_ui_transfer_cannot_hide_safety_regression(self):
        original=verify()
        self.assertEqual(original["status"],
                         "COMPLETE_FIRST_PARTY_UI_COMPOSITION_DIAGNOSTIC")
        self.assertEqual(original["h403"]["unique"]["correct"],11)
        self.assertEqual(original["h406"]["unique"]["correct"],4)
        self.assertEqual(original["h403"]["ambiguous"]["unsafe_clicks"],10)
        self.assertEqual(original["h406"]["ambiguous"]["unsafe_clicks"],3)
        self.assertEqual(original["h403"]["all"]["correct"],25)
        self.assertEqual(original["h406"]["all"]["correct"],25)
        self.assertEqual(original["sha256"],
                         "402dc7d129a2d21cdc201c13bc0696a2d34d1bce9d63da617ea5826fc25157ba")

    def test_failure_retains_research_only_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            save(p,failed())
            self.assertEqual(verify(p)["status"],"FAILED_NO_MODEL_QUALIFICATION")
    def test_sha_tampering_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            save(p,failed())
            (p/"receipt.json").write_bytes(
                (p/"receipt.json").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(p)
    def test_fake_original_gui_rights_ood_or_public_model_refused(self):
        for key in ("external_ood","third_party_gui_data",
                    "original_gui_train_rights_approved",
                    "new_training","model_weights_published",
                    "official_hala_model_qualified"):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)
                doc=failed()
                doc[key]=True
                save(p,doc)
                with self.assertRaisesRegex(ValueError,"cannot fabricate"):
                    verify(p)
    def test_cannot_claim_completed_evaluation_without_real_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            doc=failed()
            doc["status"]="COMPLETE_FIRST_PARTY_UI_COMPOSITION_DIAGNOSTIC"
            save(p,doc)
            with self.assertRaisesRegex(ValueError,"source count"):
                verify(p)

if __name__=="__main__":
    unittest.main()
