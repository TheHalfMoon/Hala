"""H137 frozen three real external library evaluation safeguards."""
import json
import unittest
from pathlib import Path
from h137.source import IDS,FAMILIES,VENDOR_SOURCES,SCHEMA,center
from h137.gpu_eval import (FROZEN_SOURCE_MANIFEST_SHA,FROZEN_ADAPTER_SHA)
from h137.owner_replay import verify,ARMS

class H137Contracts(unittest.TestCase):
    def test_real_mit_vendor_package_and_balanced_36_scenes(self):
        self.assertEqual(len(IDS),36)
        self.assertEqual(len(FAMILIES),3)
        self.assertEqual([sum(i%3==j for i in IDS) for j in range(3)],[12]*3)
        self.assertEqual([sum(i%3==j and i%2==1 for i in IDS) for j in range(3)],[6]*3)
        self.assertEqual({x[0] for x in VENDOR_SOURCES.values()},
                         {"nouislider","sortablejs","split.js"})
        self.assertIn("interaction-libraries",SCHEMA)
        for sha in (FROZEN_SOURCE_MANIFEST_SHA,FROZEN_ADAPTER_SHA):
            self.assertEqual(len(sha),64)

    def test_actual_vendor_interactions_not_self_simulated(self):
        owner=Path(__file__).resolve().parents[1]
        js=(owner/"h137"/"app.js").read_text("utf8")
        self.assertIn("noUiSlider.create",js)
        self.assertIn("new Sortable",js)
        self.assertIn("Split([",js)
        self.assertIn('workspace.dataset.success',js)
        self.assertNotIn("source_id",js)
        self.assertNotIn("oracle",js)

    def test_raw_model_native_points_and_holdout_guards(self):
        cases=[]
        raw=[]
        for seed in IDS:
            case={"case_id":f"h137-{seed}","lang":"ar" if seed%2 else "en",
                  "family":FAMILIES[seed%3],"image_sha256":f"sha-{seed}",
                  "start_bbox":{"x":95,"y":95,"width":10,"height":10},
                  "end_bbox":{"x":195,"y":195,"width":10,"height":10}}
            cases.append(case)
        for arm in ARMS:
            for case in cases:
                for role,p,grid in (("start",[100,100],[104,139]),
                                    ("end",[200,200],[208,278])):
                    raw.append({"arm":arm,"case_id":case["case_id"],
                        "lang":case["lang"],"family":case["family"],
                        "image_sha256":case["image_sha256"],
                        "role":role,"point":p,"raw":json.dumps({"point_2d":grid}),
                        "hit":True})
        report={"schema":"hala.h137.private-real-third-party-library-first-blind-eval.v1",
            "status":"H137_PRIVATE_THREE_REAL_MIT_INTERACTION_LIBRARIES_144_INFERENCE",
            "actual_neural_calls":144,"actual_optimizer_steps":0,
            "source_sha256":FROZEN_SOURCE_MANIFEST_SHA,
            "frozen_adapter_sha256":FROZEN_ADAPTER_SHA,
            "training_source_attached":False,"paid_compute_used":False,
            "benchmark_is_independent_production_apps":False}
        self.assertEqual(len(verify({"cases":cases},report,raw)),144)
        bad=dict(report,actual_optimizer_steps=1)
        with self.assertRaises(ValueError):verify({"cases":cases},bad,raw)
        badraw=[dict(i) for i in raw]
        badraw[0]["point"]=[600,400]
        with self.assertRaises(ValueError):verify({"cases":cases},report,badraw)

if __name__=="__main__":unittest.main()
