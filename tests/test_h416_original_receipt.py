"""H416 original GPU receipt regression contract; no public model claims."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from h416.validate_receipt import verify

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"h416/verified_receipt/h416-private-absence-repair-lora"
EXPECTED="8db9eea782cd5e085db369eb483f27a0681c4cb0233497f53b99a663be733b0f"


class H416OriginalReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(DATA/"receipt.json").read_bytes()
        cls.state=json.loads(cls.raw)

    def reject(self,raw,sidecar=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"receipt.json").write_bytes(raw)
            (root/"receipt.sha256").write_text(
                sha256(raw).hexdigest() if sidecar is None else sidecar)
            with self.assertRaises(ValueError):
                verify(root)

    def test_owner_original_8864_gradient_train_and_model_byte_shas(self):
        self.assertEqual(sha256(self.raw).hexdigest(),EXPECTED)
        self.assertEqual((DATA/"receipt.sha256").read_text().strip(),EXPECTED)
        r=self.state
        self.assertEqual(r["schema"],"hala.h416.private-absence-repair-8864-lora.v1")
        self.assertEqual(r["status"],"COMPLETE_PRIVATE_8864_EXAMPLE_ABSENCE_REPAIR_LORA")
        self.assertEqual(r["optimizer_steps"],2216)
        self.assertEqual(r["processed_train_examples"],8864)
        self.assertEqual(r["changed_lora_tensors"],112)
        self.assertEqual(r["adapter_sha256"],
            "676dae18505281ef3921ae5a822a9a5d546cbed0e2fc0f0db400fb0788f74e93")
        self.assertEqual(r["device_name"],"Tesla T4")
        self.assertIs(r["weights_public"],False)
        self.assertIs(r["independent_external_ood"],False)

    def test_real_original_96_predictions_report_improvement_without_release(self):
        r=self.state
        self.assertEqual(len(r["parent_audit_exact"]),96)
        self.assertEqual(len(r["adapted_audit_exact"]),96)
        self.assertEqual(r["parent_audit"]["all"]["correct"],93)
        self.assertEqual(r["adapted_audit"]["all"]["correct"],96)
        self.assertEqual(r["parent_audit"]["absent"]["unsafe_clicks"],3)
        self.assertEqual(r["adapted_audit"]["absent"]["unsafe_clicks"],0)
        self.assertEqual(r["adapted_audit"]["duplicate"]["unsafe_clicks"],0)
        self.assertEqual(r["adapted_audit"]["unique"]["correct"],32)
        self.assertTrue(r["toy_safety_gate"])
        self.assertIs(r["official_hala_model_trained"],False)

    def test_old_h414_gpu_receipt_cannot_pose_as_8864_run(self):
        old=(ROOT/"h414/verified_receipt/h414-private-thousands-multisource-lora/receipt.json").read_bytes()
        self.reject(old)

    def test_bad_sidecar_is_a_hard_failure(self):
        self.reject(self.raw,"0"*64)

    def test_extra_training_claims_do_not_bypass_lineage(self):
        changed=dict(self.state)
        changed["base_revision"]="unapproved"
        self.reject(json.dumps(changed).encode())

    def test_fake_external_ood_attestation_is_rejected(self):
        changed=dict(self.state)
        changed["independent_external_ood"]=True
        self.reject(json.dumps(changed).encode())

    def test_duplicate_json_properties_rejected(self):
        changed=self.raw.decode("utf8").replace(
            '"schema": "hala.h416.private-absence-repair-8864-lora.v1"',
            '"schema": "hala.h416.private-absence-repair-8864-lora.v1", "schema":"bad"',
            1)
        self.reject(changed.encode())


if __name__=="__main__":
    unittest.main()
