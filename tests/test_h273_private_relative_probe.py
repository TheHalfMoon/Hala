"""H273 private scientific A/B research guard tests (invented numbers only)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from h273.private_relative_probe import ARMS,ALLOWED_SCENES,aggregate
from h273.build_private_payload import (
    KERNEL_ID,SCRIPT_NAME,META_NAME,LMFE_SHA,make,
)

def synthetic_row(scene,arm,hit=False,success=False):
    return {"scene":scene,"arm":arm,
       "completed_real_pretrained_generations":1,
       "token_level_filter_generate_calls":1,
       "output_valid_native_after_preregistered_frame_conversion":True,
       "kind":"click","native_browser_input_dispatched":True,
       "model_chosen_point_hits_correct_first_button":hit,
       "firstparty_inventory_task_success":success}


def four_rows():
    return [synthetic_row(scene,arm)
      for scene in ALLOWED_SCENES for arm in ARMS]


class H273SourceBoundABContracts(unittest.TestCase):
    def test_aggregate_has_causal_same_image_and_unscored_firstparty_limits(self):
        rows=four_rows()
        rows[1]["model_chosen_point_hits_correct_first_button"]=True
        rows[1]["firstparty_inventory_task_success"]=True
        out=aggregate(rows,["a","a","b","b"],{"free_t4":True,"t4_count":2})
        self.assertEqual(out["relative_first_plus_hits"],1)
        self.assertEqual(out["pixel_first_plus_hits"],0)
        self.assertEqual(out["relative_native_task_success"],1)
        self.assertFalse(out["independent_blind_ood"])
        self.assertFalse(out["official_train_dev_ood_admitted"])
        self.assertFalse(out["model_learned_generalizable_skill_proven"])
        self.assertFalse(out["raw_normalized_text_or_points_persisted"])
        self.assertFalse(out["public_weights_or_adapters_persisted"])
        self.assertNotIn('"gold":',json.dumps(out))

    def test_fail_closed_missing_real_generation_or_unpaired_images(self):
        rows=four_rows()
        with self.assertRaises(ValueError):
            aggregate(rows,["a","different","b","b"],{"free_t4":True})
        with self.assertRaises(ValueError):
            aggregate(rows,["a","a","b","b"],{"free_t4":False})
        rows[0]["completed_real_pretrained_generations"]=0
        with self.assertRaises(ValueError):
            aggregate(rows,["a","a","b","b"],{"free_t4":True})
        rows=four_rows()
        rows[0]["firstparty_inventory_task_success"]=True
        with self.assertRaises(ValueError):
            aggregate(rows,["a","a","b","b"],{"free_t4":True})

    def test_payload_one_private_code_only_no_uploaded_images_or_model(self):
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as scratch:
            path=Path(scratch)/"fresh"
            status=make(root/"h273"/"private_relative_probe.py",
                        root/"h273"/"relative_native_bridge.py",path)
            self.assertTrue(status["private"])
            self.assertEqual(status["model_or_private_media_sources"],0)
            self.assertEqual({p.name for p in path.iterdir()},
                             {SCRIPT_NAME,META_NAME})
            meta=json.loads((path/META_NAME).read_text("utf-8"))
            script=(path/SCRIPT_NAME).read_text("utf-8")
            compile(script,"h273_private_script","exec")
            self.assertEqual(meta["id"],KERNEL_ID)
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"],[])
            self.assertEqual(meta["model_sources"],[])
            self.assertIn("normalize_strict_action",script)
            self.assertIn("build_normalized_token_filter",script)
            self.assertIn(LMFE_SHA,script)
            self.assertIn("/tmp/hala-h273-relative-point",script)
            self.assertNotIn("HF_HOME='/kaggle/working",script)
            self.assertEqual(script.count("from __future__ import annotations"),1)
            with self.assertRaises(ValueError):
                make(root/"h273"/"private_relative_probe.py",
                     root/"h273"/"relative_native_bridge.py",path)

if __name__=="__main__":
    unittest.main()
