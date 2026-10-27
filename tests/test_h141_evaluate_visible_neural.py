"""Refuse leaked oracle, phantom predictions and non-H141 model evaluations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h141.evaluate_visible_neural import (
    load_predictions,admissible_point,FIELDS
)
from h141.visible_goal_source import EVAL_SCHEMA
from h141.gpu_visible_probe import (
    SCHEMA,STATUS,EVAL_INPUT_SHA,EVAL_MANIFEST_SHA,V2_ADAPTER_SHA,SOURCE_SHA
)
SOURCE_MANIFEST_SHA="20a472485cd04f3b6284c0a8d8e7ab01b5ff1f09154d8b5ab4479a313da8ad3b"

class H141TrueModelVisibleSourceEvaluationGate(unittest.TestCase):
    def setup_files(self,root):
        source=root/"private-h141"/"react-original-neural-probe16-v1"
        source.mkdir(parents=True)
        rows=[{"id":f"original-visible-goal-{i:03d}",
               "language":"en" if i%2==0 else "ar",
               "app":"react-native-board"}
              for i in range(16)]
        (source/"manifest.json").write_text(json.dumps({
            "schema":EVAL_SCHEMA,"cases":rows}),encoding="utf8")
        doc={
            "schema":SCHEMA,"status":STATUS,
            "original_source_role":"new_h141_neural_probe_not_h140_blind",
            "original_source_git_sha":SOURCE_SHA,
            "input_zip_sha256":EVAL_INPUT_SHA,
            "input_manifest_sha256":EVAL_MANIFEST_SHA,
            "adapter_sha256":V2_ADAPTER_SHA,
            "gold_boxes_seen_by_model":False,
            "training_eligible":False,
            "real_optimizer_steps":0,"sealed_h140_blind_calls":0,
            "model_calls":32,
            "predictions":[{
                **r,"start_point":[220,350],"start_output_valid":True,
                "visible_column_heading_point":[480,181],
                "visible_column_heading_output_valid":True}
                for r in rows],
        }
        file=root/"predictions.json"
        file.write_text(json.dumps(doc),encoding="utf8")
        return file,doc

    def test_only_complete_16_predictions_and_no_owner_gold_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            file,doc=self.setup_files(root)
            with patch("h141.evaluate_visible_neural.audit",return_value={
                    "source_manifest_sha256":SOURCE_MANIFEST_SHA}):
                results,manifest=load_predictions(file,root)
                self.assertEqual(len(results),16)
                self.assertEqual(len(manifest["cases"]),16)
                self.assertEqual(set(next(iter(results.values()))),FIELDS)
                doc["predictions"][0]["visible_heading_bbox"]={"x":1}
                file.write_text(json.dumps(doc),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"no oracle"):
                    load_predictions(file,root)
                doc["predictions"][0].pop("visible_heading_bbox")
                doc["sealed_h140_blind_calls"]=1
                file.write_text(json.dumps(doc),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"gold-free"):
                    load_predictions(file,root)
                doc["sealed_h140_blind_calls"]=0
                doc["predictions"][0]["start_point"]=[1000,100]
                file.write_text(json.dumps(doc),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"validity mismatch"):
                    load_predictions(file,root)

    def test_no_nonfinite_or_offscreen_points(self):
        self.assertTrue(admissible_point([959,719]))
        for point in ([960,719],[-2,340],[float("nan"),200],
                      [None,100],None,{"x":20},["12","20"]):
            self.assertFalse(admissible_point(point))

if __name__=="__main__":
    unittest.main()
