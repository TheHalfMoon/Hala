"""H140 original unseen private inference requires pinning model and NO gold."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h140.blind_gpu_infer import (
    ADMISSION_SCHEMA,PREDICTIONS_SCHEMA,INPUT_MANIFEST_SHA,
    admit_v3_checkpoint,admit_redacted_source,
    OWNER_BLIND_SHA,TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA
)
from h140.create_blind_authorization import authorize

class H140OneTimeOriginalModelInferenceContracts(unittest.TestCase):
    @staticmethod
    def cert(sha):
        return {
            "schema":ADMISSION_SCHEMA,
            "status":"ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND",
            "train_manifest_sha256":TRAIN_MANIFEST_SHA,
            "dev_manifest_sha256":DEV_MANIFEST_SHA,
            "owner_original_manifest_sha256":OWNER_BLIND_SHA,
            "adapter_sha256":sha,
            "private_only":True,
            "neural_blind_calls_before_authorization":0,
        }

    def test_no_checkpoint_without_exact_owner_dev_unlock(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            adapter=root/"adapter";adapter.mkdir()
            weights=b"owner-generated fixture rank8 weights only"
            (adapter/"adapter_model.safetensors").write_bytes(weights)
            sha=hashlib.sha256(weights).hexdigest()
            cert=root/"cert.json"
            doc=self.cert(sha)
            cert.write_text(json.dumps(doc))
            self.assertEqual(admit_v3_checkpoint(adapter,cert),sha)
            doc["status"]="NOT_QUALIFIED_DO_NOT_TOUCH_BLIND"
            cert.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"DEV pass"):
                admit_v3_checkpoint(adapter,cert)
            doc["status"]="ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND"
            doc["adapter_sha256"]="0"*64
            cert.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"differs"):
                admit_v3_checkpoint(adapter,cert)
            with self.assertRaises((OSError,ValueError)):
                admit_redacted_source(root/"missing.zip")

    def test_gate_failure_refuses_creating_authorization_file(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"private-h140";root.mkdir()
            certificate=root/"authorize.json"
            with patch("h140.create_blind_authorization.evaluate",
                       return_value={"status":"NOT_QUALIFIED_DO_NOT_TOUCH_BLIND",
                                     "checks":{"diverse":False}}):
                with self.assertRaisesRegex(ValueError,"cannot access"):
                    authorize("v2","v3",certificate)
            self.assertFalse(certificate.exists())
        self.assertEqual(len(INPUT_MANIFEST_SHA),64)
        self.assertIn("model-predictions",PREDICTIONS_SCHEMA)

if __name__=="__main__":
    unittest.main()
