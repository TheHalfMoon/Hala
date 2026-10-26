"""H140 one-shot original browser benchmark must reject oracle and replay errors."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h140.evaluate_one_shot_browser_blind import (
    SCHEMA,INPUT_SHA,load_predictions,score_original_blind,valid_point
)
from h140.export_blind_redacted import OWNER_BLIND_SHA

class H140NoLeakedBlindNativeBrowserTests(unittest.TestCase):
    def fake_pairs(self):
        result={}
        for app in ("ood-vanilla-complete","ood-second-drag"):
            result[app]=[(None,{"id":f"blind-{app}-{i:03d}",
                                "language":"en" if i%2==0 else "ar"})
                         for i in range(16)]
        return result

    def fake_model_predictions(self,sha):
        rows=[]
        for app,cases in self.fake_pairs().items():
            for _,case in cases:
                rows.append({"id":case["id"],"app":app,
                             "language":case["language"],
                             "start_point":[200,300],
                             "end_point":[500,300]})
        return {
            "schema":SCHEMA,
            "owner_original_manifest_sha256":OWNER_BLIND_SHA,
            "redacted_manifest_sha256":INPUT_SHA,
            "adapter_sha256":sha,
            "model_evaluation_role":"one_shot_blind_inference_only",
            "trained_on_blind":False,
            "model_calls":64,"predictions":rows,
        }

    def test_exact_32_model_frames_without_oracle(self):
        sha="7"*64
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"predictions.json"
            doc=self.fake_model_predictions(sha)
            file.write_text(json.dumps(doc),encoding="utf8")
            with patch("h140.evaluate_one_shot_browser_blind.source_pairing",
                       return_value=self.fake_pairs()):
                found=load_predictions(file,sha,folder)
                self.assertEqual(len(found),32)
                doc["predictions"][0]["start_bbox"]={"x":10}
                file.write_text(json.dumps(doc),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"oracle keys"):
                    load_predictions(file,sha,folder)
                doc["predictions"][0].pop("start_bbox")
                doc["predictions"][0]["end_point"]=[960,300]
                file.write_text(json.dumps(doc),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"on-screen"):
                    load_predictions(file,sha,folder)

    def test_dev_no_go_refuses_consuming_blind(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("h140.evaluate_one_shot_browser_blind.evaluate",
                       return_value={"status":"NOT_QUALIFIED_DO_NOT_TOUCH_BLIND"}):
                with self.assertRaisesRegex(ValueError,"failed DEV"):
                    score_original_blind(folder,"fake","v2","v3")
            self.assertFalse((Path(folder)/"frozen-source-v1"/"blind_test"/
                              "ONE_SHOT_EVALUATION_ALREADY_STARTED.json").exists())

    def test_point_validation_prevents_outside_browser_gestures(self):
        self.assertTrue(valid_point([959,719]))
        self.assertFalse(valid_point([-1,20]))
        self.assertFalse(valid_point([1000,20]))
        self.assertFalse(valid_point([None,20]))
        self.assertFalse(valid_point([float("nan"),20]))
        self.assertFalse(valid_point(None))

if __name__=="__main__":
    unittest.main()
