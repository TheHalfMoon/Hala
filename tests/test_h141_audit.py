"""H141 source audit must never turn oracle-hover images into model data."""
import unittest
from unittest.mock import patch
from h141.audit import audit,excluded_originals

class H141FailClosedSourceAuditTests(unittest.TestCase):
    def test_original_owner_sources_missing_fail_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                audit(folder)
            with self.assertRaises((OSError,ValueError)):
                excluded_originals(folder)

    def test_h141_diagnostic_never_trainable_or_blind_eligible(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as base:
            root=Path(base)
            folder=root/"private-h141"/"original-react-interactive-diagnostic8"
            folder.mkdir(parents=True)
            (folder/"manifest.json").write_text(json.dumps({
                "schema":"hala.h141.original-react-drag-reveal-private-source-qa.v1",
                "private":True,
                "status":"QUALIFIED_SOURCE_QA_NOT_MODEL_TRAINING",
                "training_eligible":True,
                "blind_eligible":False,"model_evaluation_eligible":False,
                "oracle_guided_drag_reference":True,
                "model_neural_calls":0,
                "source_app":"react-native-board",
                "source_git_sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
                "cases":[{} for _ in range(8)]}))
            with self.assertRaisesRegex(ValueError,"cannot be model eligible"):
                audit(root)
            doc=json.loads((folder/"manifest.json").read_text())
            doc["training_eligible"]=False
            doc["model_neural_calls"]=1
            (folder/"manifest.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"cannot be model eligible"):
                audit(root)

if __name__=="__main__":
    unittest.main()
