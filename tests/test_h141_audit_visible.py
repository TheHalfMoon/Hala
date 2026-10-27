"""No previously hidden target or old source screenshot enters visible goal."""
import json
import tempfile
import unittest
from pathlib import Path
from h141.audit_visible import audit,SCHEMA

class H141VisibleSemanticOwnerAuditTests(unittest.TestCase):
    def test_missing_original_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises((OSError,ValueError)):
                audit(folder)

    def test_training_eligibility_or_hidden_release_role_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"private-h141"/"react-original-visible-goal16"
            p.mkdir(parents=True)
            doc={
                "schema":SCHEMA,"status":"VISIBLE_ORIGINAL_COLUMN_DIAGNOSTIC_ONLY",
                "private":True,
                "training_eligible":True,
                "blind_eligible":False,
                "model_evaluation_eligible":False,
                "real_neural_calls":0,
                "target_role":"visible_destination_column_approach_not_release",
                "source_original_mit_sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
                "cases":[{} for _ in range(16)],
            }
            (p/"manifest.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"unqualified or oracle-leaking"):
                audit(folder)
            doc["training_eligible"]=False
            doc["target_role"]="hidden_exact_drop_point"
            (p/"manifest.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"unqualified or oracle-leaking"):
                audit(folder)
if __name__=="__main__":
    unittest.main()
