"""CPU-only H140 neural receipt and strict original-DEV anti-collapse tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h140.train_gpu import (
    STEPS,GRAD_ACCUM,TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA,
    EXPECTED_APP_SET,summarize_native,private_splits,
)
from h140.qualification import SCHEMA


class H140NativeGUIResearchGrounderContract(unittest.TestCase):
    @staticmethod
    def cases():
        sizes={"vanilla-native-board":22,
               "react-native-board":22,
               "react-dndkit-board":20}
        rows=[]
        for app,n in sizes.items():
            for i in range(n):
                x=40+(i%16)*26
                rows.append({"id":f"{app}-{i}","app":app,
                             "language":"ar" if i%2 else "en",
                             "start_bbox":{"x":x,"y":160,"width":20,"height":20},
                             "end_bbox":{"x":x+40,"y":360,"width":20,"height":20}})
        return {"cases":rows}

    @staticmethod
    def predictions(doc,collapsed=False):
        out=[]
        for row in doc["cases"]:
            for role in ("start","end"):
                rect=row[role+"_bbox"]
                point=[42,44] if collapsed else [
                    rect["x"]+rect["width"]/2,rect["y"]+rect["height"]/2]
                hit=(rect["x"]<=point[0]<=rect["x"]+rect["width"] and
                     rect["y"]<=point[1]<=rect["y"]+rect["height"])
                out.append({"case_id":row["id"],"role":role,
                            "raw":"CPU-only fixture never model output",
                            "point":point,"hit":bool(hit)})
        return out

    def test_balanced_one_epoch_all_512_source_roles(self):
        self.assertEqual((STEPS,GRAD_ACCUM),(128,4))
        self.assertEqual(STEPS*GRAD_ACCUM,512)
        self.assertEqual(len(TRAIN_MANIFEST_SHA),64)
        self.assertEqual(len(DEV_MANIFEST_SHA),64)
        self.assertEqual(len(EXPECTED_APP_SET),3)

    def test_diverse_all_three_apps_and_both_roles_pass(self):
        doc=self.cases()
        result=summarize_native(self.predictions(doc),doc)
        self.assertTrue(result["all_apps_and_roles_diverse"])
        self.assertEqual(result["point_hits"]["__all__"],
                         {"start":64,"end":64,"both":64,"cases":64})
        for roles in result["point_diversity_by_app"].values():
            self.assertGreaterEqual(roles["start"]["unique_points"],8)
            self.assertGreaterEqual(roles["end"]["unique_points"],8)

    def test_fixed_native_points_fail_even_if_model_text_is_valid(self):
        doc=self.cases()
        result=summarize_native(self.predictions(doc,collapsed=True),doc)
        self.assertFalse(result["all_apps_and_roles_diverse"])
        for roles in result["point_diversity_by_app"].values():
            self.assertFalse(roles["start"]["passed"])
            self.assertFalse(roles["end"]["passed"])
        with self.assertRaises(ValueError):
            summarize_native(self.predictions(doc)[:-1],doc)

    def test_training_mount_refuses_even_single_blind_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            blind=root/"sealed-model-blind"
            blind.mkdir()
            (blind/"manifest.json").write_text(json.dumps({
                "schema":SCHEMA,"split":"blind_test","cases":[]}))
            with patch("h140.train_gpu.DATA_ROOT",root):
                with self.assertRaisesRegex(ValueError,"refuse BLIND mount"):
                    private_splits()


if __name__=="__main__":
    unittest.main()
