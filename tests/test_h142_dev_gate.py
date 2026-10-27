"""CPU-only precommitted H142 DEV gate cannot promote weak heading LoRA."""
import json
import tempfile
import unittest
from pathlib import Path
from h142.dev_gate import qualify

class H142VisibleSourceDevGateTests(unittest.TestCase):
    def make(self):
        whole={"cases":16,"start_hit":16,"visible_column_heading_hit":12,
               "both_hit":12}
        before={"all":{"cases":16,"start_hit":16,
                       "visible_column_heading_hit":0}}
        after={"all":whole,
               "doing":{"visible_column_heading_hit":6},
               "done":{"visible_column_heading_hit":6},
               "ar":{"visible_column_heading_hit":6},
               "en":{"visible_column_heading_hit":6}}
        from h142.train_gpu import TRAIN_SHA,DEV_SHA
        return {"schema":"hala.h142.fresh-original-visible-column-2b-adapter.v1",
                "status":"COMPLETED_64_REAL_FREE_T4_TRAIN_STEPS",
                "steps":64,"accum":3,
                "train_manifest_sha256":TRAIN_SHA,
                "dev_manifest_sha256":DEV_SHA,
                "source_train_original":64,"source_dev_original":16,
                "start_examples_seen":64,
                "visible_heading_examples_seen":128,
                "paid_compute_used":False,
                "h140_blind_model_calls":0,
                "h141_consumed_probe_train_or_dev_used":False,
                "flagship_Hala_35B_updated":False,
                "v2_checkpoint_before":before,
                "h142_checkpoint_after":after}
    def test_new_original_visible_goal_gain_qualifies_only_one_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"receipt.json"
            d=self.make()
            f.write_text(json.dumps(d))
            result=qualify(f)
            self.assertEqual(result["status"],"QUALIFIED_ONLY_FOR_NEW_INDEPENDENT_ACTION_COHORT")
            self.assertFalse(result["independent_generalization_proven"])
            self.assertEqual(result["h140_blind_neural_calls"],0)
    def test_no_go_if_no_real_heading_gain_or_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"receipt.json"
            d=self.make()
            d["h142_checkpoint_after"]["all"]["visible_column_heading_hit"]=8
            d["h142_checkpoint_after"]["all"]["both_hit"]=8
            f.write_text(json.dumps(d))
            self.assertEqual(qualify(f)["status"],"NO_GO_DO_NOT_TOUCH_H140_BLIND")
            d["h142_checkpoint_after"]["all"]["visible_column_heading_hit"]=12
            d["h142_checkpoint_after"]["all"]["both_hit"]=12
            d["h140_blind_model_calls"]=1
            f.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"not qualified"):
                qualify(f)
if __name__=="__main__":
    unittest.main()
