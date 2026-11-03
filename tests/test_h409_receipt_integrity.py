"""H409 immutable first-party LoRA research receipt fails closed."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from h409.validate_receipt import verify,PARENT_SHA

def failed():
    return {
        "schema":"hala.h409.private-balanced-two-family-lora.v1",
        "status":"FAILED_PRIVATE_SOURCE_ONLY_NO_RELEASE",
        "base":"Qwen/Qwen3-VL-2B-Instruct",
        "base_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
        "parent_adapter_sha256":PARENT_SHA,
        "source_families":["button_ui","color_grid"],
        "owner_first_party_generated_only":True,
        "reused_h405_h406_h408_audit":False,
        "independent_external_ood":False,
        "original_gui_train_rights_approved":False,
        "official_hala_model_trained":False,
        "weights_public":False,
        "optimizer_steps":0,
        "toy_safety_gate":False,
        "failure_class":"ValueError",
    }

def write(path,record):
    raw=json.dumps(record,sort_keys=True).encode()
    (path/"receipt.json").write_bytes(raw)
    (path/"receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())

class H409ReceiptIntegrityTests(unittest.TestCase):
    def test_original_owner_kaggle_128_step_success_and_false_clicks(self):
        verified=verify()
        self.assertEqual(verified["sha256"],
            "f36d27b737cd327860f0d479d27a7f28720ff19b77661c930c3d06fc49016d63")
        self.assertEqual(verified["status"],
            "COMPLETE_PRIVATE_TWO_FAMILY_128_STEP_LORA")
        self.assertEqual(verified["adapter_sha256"],
            "0ce00eec423a80938344048f69f9b50734b0f5b52cae646767896bffde1d3dbc")
        self.assertEqual(verified["changed_lora_tensors"],112)
        self.assertEqual(verified["h406"]["all"]["correct"],34)
        self.assertEqual(verified["h409"]["all"]["correct"],40)
        self.assertEqual(verified["h406"]["unique"]["correct"],7)
        self.assertEqual(verified["h409"]["unique"]["correct"],14)
        self.assertEqual(verified["h406"]["ambiguous"]["unsafe_clicks"],5)
        self.assertEqual(verified["h409"]["ambiguous"]["unsafe_clicks"],6)
        self.assertFalse(verified["toy_safety_gate"])


    def test_failed_research_does_not_masquerade_as_model_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root,failed())
            info=verify(root)
            self.assertEqual(info["status"],"FAILED_PRIVATE_SOURCE_ONLY_NO_RELEASE")
            self.assertFalse(info["toy_safety_gate"])

    def test_original_kaggle_json_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root,failed())
            (root/"receipt.json").write_bytes(
                (root/"receipt.json").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(root)

    def test_forged_independent_ood_rights_or_weight_publication_refused(self):
        for name in ("reused_h405_h406_h408_audit",
                     "independent_external_ood",
                     "original_gui_train_rights_approved",
                     "official_hala_model_trained","weights_public"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                doc=failed()
                doc[name]=True
                write(root,doc)
                with self.assertRaisesRegex(ValueError,"cannot invent"):
                    verify(root)

    def test_fabricated_complete_without_genuine_128_steps_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            doc=failed()
            doc["status"]="COMPLETE_PRIVATE_TWO_FAMILY_128_STEP_LORA"
            write(root,doc)
            with self.assertRaisesRegex(ValueError,"128-step"):
                verify(root)

    def test_failed_status_can_never_claim_safety_gate_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            doc=failed()
            doc["toy_safety_gate"]=True
            write(root,doc)
            with self.assertRaisesRegex(ValueError,"failed source-only"):
                verify(root)
if __name__=="__main__":unittest.main()
