"""H270 CPU synthetic-only contracts; no GPU, weights or user private data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from h270.coordinate_weighted_lora import (
    TRAIN_SCENES,EVAL_SCENES,N_STEPS,COORDINATE_WEIGHT,OUTPUT,
    measured_aggregate,
)
from h270.build_private_payload import (
    make,KERNEL_ID,SCRIPT_NAME,META_NAME,
)


def train_steps():
    return [{"changed_tensors":112,"coordinate_tokens":2,
             "gradient_norm":1.0,"loss":2.0} for _ in range(N_STEPS)]


def eval_rows(succeed=False):
    return [
        {"scene":scene[0],"real_generation_complete":True,
         "model_time_grammar_called":True,
         "strict_native_json":True,"kind":"click",
         "browser_action_dispatched":True,
         "firstparty_task_success":succeed}
        for scene in EVAL_SCENES
    ]


class H270SourceBoundTests(unittest.TestCase):
    def test_eight_distinct_training_modes_two_nonidentical_holdouts(self):
        self.assertEqual(len(TRAIN_SCENES),N_STEPS)
        self.assertEqual(len(set(TRAIN_SCENES)),N_STEPS)
        self.assertEqual(len(EVAL_SCENES),2)
        self.assertEqual(len(set(EVAL_SCENES)),2)
        self.assertFalse(set(TRAIN_SCENES).intersection(EVAL_SCENES))
        self.assertEqual(sum(x[1]=="en" for x in TRAIN_SCENES),6)
        self.assertEqual(sum(x[1]=="ar" for x in TRAIN_SCENES),2)
        self.assertTrue(all(x[1]=="ar" for x in EVAL_SCENES))
        self.assertEqual(COORDINATE_WEIGHT,8.0)
        self.assertEqual(OUTPUT,"hala-h270-private-weighted-firstparty-aggregate.json")

    def test_aggregate_says_firstparty_only_even_with_success(self):
        agg=measured_aggregate(train_steps(),eval_rows(False),
                               eval_rows(True),
                               {"free_t4_verified":True,"t4_count":2})
        self.assertEqual(agg["real_optimizer_steps"],8)
        self.assertEqual(agg["paired_firstparty_delta"],2)
        self.assertEqual(agg["paired_firstparty_base_successes"],0)
        self.assertEqual(agg["paired_firstparty_adapted_successes"],2)
        self.assertFalse(agg["independent_blind_ood_proven"])
        self.assertFalse(agg["model_generalized_improvement_proven"])
        self.assertFalse(agg["official_train_dev_ood_admitted"])
        self.assertFalse(agg["model_weights_published"])
        self.assertFalse(agg["raw_predictions_images_coords_labels_saved_to_output"])
        self.assertTrue(agg["eval_exact_pngs_identical_before_after"])
        self.assertNotIn("image_sha256",json.dumps(agg))
        self.assertNotIn("coordinates",json.dumps(agg))

    def test_fail_closed_on_fake_optimizer_or_wrong_heldout(self):
        invalid_steps=[
            train_steps()[:7],
            [{"changed_tensors":0,"coordinate_tokens":2,
              "gradient_norm":1.0,"loss":2.0}] +train_steps()[1:],
            [{"changed_tensors":2,"coordinate_tokens":0,
              "gradient_norm":1.0,"loss":2.0}] +train_steps()[1:],
            [{"changed_tensors":2,"coordinate_tokens":1,
              "gradient_norm":float("nan"),"loss":2.0}]+train_steps()[1:],
        ]
        for steps in invalid_steps:
            with self.subTest(n=len(steps)),self.assertRaises(ValueError):
                measured_aggregate(steps,eval_rows(),eval_rows(),
                                   {"free_t4_verified":True})
        bad=eval_rows()
        bad[1]["model_time_grammar_called"]=False
        with self.assertRaises(ValueError):
            measured_aggregate(train_steps(),eval_rows(),bad,
                               {"free_t4_verified":True})
        bad=eval_rows()
        bad[0]["scene"]="h173_sealed"
        with self.assertRaises(ValueError):
            measured_aggregate(train_steps(),bad,eval_rows(),
                               {"free_t4_verified":True})
        bad=eval_rows()
        bad[0]["firstparty_task_success"]=True
        bad[0]["browser_action_dispatched"]=False
        with self.assertRaises(ValueError):
            measured_aggregate(train_steps(),bad,eval_rows(),
                               {"free_t4_verified":True})

    def test_private_script_no_media_or_weights_uploaded(self):
        root=Path(__file__).resolve().parents[1]
        source=root/"h270"/"coordinate_weighted_lora.py"
        with TemporaryDirectory() as td:
            path=Path(td)/"fresh"
            proof=make(source,path)
            self.assertTrue(proof["private"])
            self.assertEqual(proof["model_or_media_uploads"],0)
            self.assertEqual({x.name for x in path.iterdir()},
                             {SCRIPT_NAME,META_NAME})
            meta=json.loads((path/META_NAME).read_text("utf-8"))
            data=(path/SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(meta["id"],KERNEL_ID)
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"],[])
            self.assertEqual(meta["model_sources"],[])
            self.assertIn('torchao',data)
            self.assertIn("coordinate_weighted",data)
            self.assertIn("init_scale=1.0",data)
            owner_trainer=(root/"training"/"scripts"/"train_adapter.py").read_text("utf-8")
            self.assertIn("init_scale=1.0",owner_trainer)
            self.assertIn("shutil.rmtree(root",data)
            with self.assertRaises(ValueError):
                make(source,path)


if __name__=="__main__":
    unittest.main()
