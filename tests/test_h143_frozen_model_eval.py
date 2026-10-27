"""No H143 owner oracle, previous BLIND, model retuning or replay drift."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h143.gpu_probe import (
    SCHEMA,IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,ADAPTER_MODEL_SHA,
    OWNER_SOURCE_MANIFEST_SHA,admit_images,admit_adapter
)
from h143.owner_neural_drag_eval import predictions,valid_point,PREDICTIONS_FIELDS
from h143.freeze_eval import audit
from h143.collect_shifted_source import TRANSFORMS

class H143FrozenIndependentVisionContracts(unittest.TestCase):
    def test_private_free_t4_only_and_pinned_new_media(self):
        path=Path(__file__).parent.parent/"h143"/"kaggle_probe"/"kernel-metadata.json"
        doc=json.loads(path.read_text("utf8"))
        self.assertTrue(doc["is_private"])
        self.assertTrue(doc["enable_gpu"])
        self.assertEqual(doc["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(doc["dataset_sources"],[
            "abdulazizshehri/hala-h143-private-shifted-layout-eval-v1"])
        self.assertEqual(doc["model_sources"],[])
        self.assertEqual(doc["kernel_sources"],[])
        for sha in (IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,
                    ADAPTER_MODEL_SHA,OWNER_SOURCE_MANIFEST_SHA):
            self.assertEqual(len(sha),64)
        self.assertEqual(len(TRANSFORMS),4)

    def test_absent_owner_sources_and_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises((OSError,ValueError)):
                audit(root)
            with self.assertRaises((OSError,ValueError)):
                admit_images(root/"unknown.zip")
            with self.assertRaises((OSError,ValueError)):
                admit_adapter(root/"unknown.zip",root/"adapter")

    def test_model_predictions_must_contain_exact_16_and_no_oracle(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            owner=root/"private-h143"/"shifted-eval16-v1"
            owner.mkdir(parents=True)
            rows=[{
                "id":f"h143-new-shifted-{i:03d}",
                "app":"react-native-board",
                "language":"ar" if i%2 else "en",
                "destination_status":"done" if i%4 in (2,3) else "doing",
                "layout":tuple(TRANSFORMS)[i//4],
                "image_sha256":f"{i+1:064x}",
            } for i in range(16)]
            (owner/"manifest.json").write_text(
                json.dumps({"cases":rows}),encoding="utf8")
            data={
                "schema":SCHEMA,
                "status":"COMPLETE_REAL_MODEL_SHIFTED_LAYOUT_32_INFERENCE_CALLS",
                "original_new_source_manifest_sha256":OWNER_SOURCE_MANIFEST_SHA,
                "input_manifest_sha256":IMAGE_MANIFEST_SHA,
                "model_checkpoint_sha256":ADAPTER_MODEL_SHA,
                "gpu_calls":32,"optimizer_steps":0,
                "owner_gold_boxes_available_to_model":False,
                "h140_sealed_blind_calls":0,
                "flagship_Hala_35B_updated":False,
                "predictions":[{
                    "id":r["id"],"app":r["app"],"language":r["language"],
                    "destination_status":r["destination_status"],
                    "layout":r["layout"],"start_point":[210,250],
                    "start_output_valid":True,
                    "visible_column_heading_point":[490,210],
                    "visible_column_heading_output_valid":True,
                } for r in rows]
            }
            file=root/"model.json";file.write_text(json.dumps(data))
            with patch("h143.owner_neural_drag_eval.audit",return_value={
                "manifest_sha256":OWNER_SOURCE_MANIFEST_SHA}):
                pred,cohort=predictions(file,root)
                self.assertEqual(len(pred),16)
                self.assertEqual(len(cohort["cases"]),16)
                self.assertEqual(set(next(iter(pred.values()))),PREDICTIONS_FIELDS)
                data["predictions"][0]["end_bbox"]={"x":1}
                file.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,"must not contain owner oracle"):
                    predictions(file,root)
                data["predictions"][0].pop("end_bbox")
                data["h140_sealed_blind_calls"]=1
                file.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,"not proven"):
                    predictions(file,root)

    def test_nonfinite_and_oob_original_mouse_points_not_actionable(self):
        self.assertTrue(valid_point([959,719]))
        for p in (None,[960,200],[-1,200],["480","200"],
                  [float("nan"),220],[None,100],{"x":200}):
            self.assertFalse(valid_point(p))
if __name__=="__main__":
    unittest.main()
