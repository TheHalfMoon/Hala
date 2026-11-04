"""H422 audited REAL H417 Tesla T4 144 original DEV model error evidence."""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

from h422 import audit_h417_real_native_model_errors as audit

ROOT=Path(__file__).resolve().parents[1]
PROOF=ROOT/"h422/verified_receipt/h417-real-native-dev-error-audit"
PROOF_SHA="08b716216e0c41db5adfdeb6b290ebd55cb84856ac1e0d4ef5be7ad512ae958e"
EXPECTED={
    "sheet":(8,15,1),
    "slider":(15,9,0),
    "slide_resize":(1,23,0),
    "text_highlight":(5,19,0),
}


class H422RealH417T4ErrorAudit(unittest.TestCase):
    def test_actual_144_model_outputs_have_sha_bound_original_errors(self):
        raw=(PROOF/"receipt.json").read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),PROOF_SHA)
        self.assertEqual((PROOF/"receipt.sha256").read_text("ascii").strip(),PROOF_SHA)
        r=json.loads(raw)
        self.assertEqual(r["schema"],audit.RESEARCH_SCHEMA)
        self.assertEqual(r["source_h417_genuine_gpu_original_receipt_sha256"],
                         audit.REAL_H417_RECEIPT_SHA)
        self.assertEqual(r["original_H417_completed_optimizer_steps"],3720)
        self.assertEqual(r["original_H417_true_image_gradient_passes"],14880)
        self.assertTrue(r["original_H417_trained_native_model"])
        self.assertEqual(r["base"]["all"],{"n":144,"valid":0,"correct":0})
        self.assertEqual(r["genuinely_finetuned_H417"]["all"],
                         {"n":144,"valid":144,"correct":76})
        self.assertEqual(r["genuinely_finetuned_H417"]["by_source_family"]["original"],
                         {"n":96,"valid":96,"correct":29})
        self.assertEqual(r["genuinely_finetuned_H417"]["by_source_family"]["firstparty"],
                         {"n":48,"valid":48,"correct":47})
        self.assertEqual(r["genuinely_finetuned_H417"]["by_language"]["ar"],
                         {"n":24,"valid":24,"correct":24})
        by=r["original_drag_only_domain_errors"]
        for domain,(correct,coordinate_errors,ordered_errors) in EXPECTED.items():
            with self.subTest(domain=domain):
                self.assertEqual(by[domain]["n"],24)
                self.assertEqual(by[domain]["valid"],24)
                self.assertEqual(by[domain]["correct"],correct)
                self.assertEqual(by[domain]["coordinate_errors"],coordinate_errors)
                self.assertEqual(by[domain].get("ordered_errors",0),ordered_errors)
        self.assertEqual(r["evidence_only_priority_for_future_curriculum"],
            ["slide_resize","text_highlight","sheet","slider"])
        self.assertFalse(r["H419_trained_model_evaluated_by_this_report"])
        self.assertFalse(r["H421_unseen_source_counterfactual_dev_evaluated"])
        self.assertFalse(r["external_independent_ood"])
        self.assertFalse(r["model_release_qualified"])

    def test_replay_actual_original_private_H417_GPU_receipt_if_mounted(self):
        file=os.environ.get("H422_OWNER_REAL_H417_RECEIPT")
        if not file:
            self.skipTest("Real owner original private GPU receipt not mounted in CI")
        actual=audit.verify_real_h417(file)
        expected=json.loads((PROOF/"receipt.json").read_text("utf8"))
        self.assertEqual(actual,expected)
        self.assertEqual(actual["genuinely_finetuned_H417"]["errors"],
                         {"coordinates":66,"correct":76,"ordered_mismatch":2})

    def test_modified_trained_GPU_receipt_cannot_masquerade_as_real_model(self):
        file=os.environ.get("H422_OWNER_REAL_H417_RECEIPT")
        if not file:
            self.skipTest("Private owner original model receipt unavailable")
        with tempfile.TemporaryDirectory() as folder:
            fake=Path(folder)/"forged-h417-gpu-receipt.json"
            fake.write_bytes(Path(file).read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"unverified H417 GPU receipt"):
                audit.verify_real_h417(fake)

    def test_h422_original_model_error_audit_receipt_is_create_only(self):
        with tempfile.TemporaryDirectory() as folder:
            doc=json.loads((PROOF/"receipt.json").read_text("utf8"))
            self.assertEqual(audit.save(folder,doc),PROOF_SHA)
            with self.assertRaises(FileExistsError):
                audit.save(folder,doc)


if __name__=="__main__":
    unittest.main()
