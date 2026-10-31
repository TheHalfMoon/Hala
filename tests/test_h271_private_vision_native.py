"""H271 CPU invented-contract checks; no private media or GPU ever used."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from h271.vision_lora_native import (
    TRAIN_SCENES,EVAL_SCENES,N_STEPS,COORDINATE_WEIGHT,
    OUTPUT,measured_aggregate,
)
from h271.build_private_payload import make,KERNEL_ID,SCRIPT_NAME,META_NAME
from training.scripts.train_adapter import (
    VISION_LORA_TARGET_PATTERN,qualify_vision_target_names,
)


def eight_steps():
    return [{"changed_tensors":16,"coordinate_tokens":2,
             "gradient_norm":1.2,"loss":2.7} for _ in range(N_STEPS)]


def rows(success=False,hit=False):
    return [
        {"scene":scene[0],"real_generation_complete":True,
         "model_time_grammar_called":True,
         "strict_native_json":True,"kind":"click",
         "browser_action_dispatched":True,
         "model_chosen_click_hits_first_plus":hit,
         "firstparty_task_success":success}
        for scene in EVAL_SCENES
    ]


class H271VisionOnlyContracts(unittest.TestCase):
    def test_use_proven_eight_last_four_vision_targets(self):
        names=[f"model.visual.blocks.{i}.attn.{proj}"
               for i in (20,21,22,23) for proj in ("qkv","proj")]
        self.assertEqual(qualify_vision_target_names(names),tuple(sorted(names)))
        import re
        self.assertTrue(all(re.fullmatch(VISION_LORA_TARGET_PATTERN,n)
                            for n in names))
        with self.assertRaises(ValueError):
            qualify_vision_target_names(names[:7])
        self.assertEqual(len(TRAIN_SCENES),N_STEPS)
        self.assertEqual(COORDINATE_WEIGHT,8.)
        self.assertFalse(set(TRAIN_SCENES)&set(EVAL_SCENES))

    def test_aggregate_keeps_vision_training_and_ood_gates_separate(self):
        result=measured_aggregate(eight_steps(),rows(False,False),
            rows(True,True),{"free_t4_verified":True,"t4_count":2})
        self.assertEqual(result["adapter_scope"],"vision_last_four_blocks_only")
        self.assertEqual(result["trainable_vision_adapter_tensors"],16)
        self.assertEqual(result["trainable_language_adapter_tensors"],0)
        self.assertEqual(result["paired_base_first_plus_hits"],0)
        self.assertEqual(result["paired_adapted_first_plus_hits"],2)
        self.assertEqual(result["paired_firstparty_delta"],2)
        self.assertFalse(result["independent_blind_ood_proven"])
        self.assertFalse(result["official_train_dev_ood_admitted"])
        self.assertFalse(result["model_generalized_improvement_proven"])
        self.assertFalse(result["model_weights_published"])
        self.assertFalse(result["raw_predictions_images_coords_labels_saved_to_output"])
        payload=json.dumps(result)
        self.assertNotIn("image_sha256",payload)
        self.assertNotIn("coordinates",payload)
        self.assertNotIn('"point"',payload)

    def test_rejects_fake_first_button_hit_or_success(self):
        bad=rows(True,False)
        with self.assertRaises(ValueError):
            measured_aggregate(eight_steps(),bad,rows(False,False),
                               {"free_t4_verified":True})
        bad=rows(False,False)
        bad[0]["model_chosen_click_hits_first_plus"]=None
        with self.assertRaises(ValueError):
            measured_aggregate(eight_steps(),rows(False),bad,
                               {"free_t4_verified":True})
        with self.assertRaises(ValueError):
            measured_aggregate(eight_steps()[:7],rows(),rows(),
                               {"free_t4_verified":True})

    def test_only_one_private_script_no_media_weights_and_source_isolation(self):
        root=Path(__file__).resolve().parents[1]
        source=root/"h271"/"vision_lora_native.py"
        with TemporaryDirectory() as scratch:
            output=Path(scratch)/"one"
            receipt=make(source,output)
            self.assertTrue(receipt["private"])
            self.assertEqual(receipt["weights_or_media_attached"],0)
            self.assertEqual({p.name for p in output.iterdir()},
                             {SCRIPT_NAME,META_NAME})
            meta=json.loads((output/META_NAME).read_text("utf-8"))
            code=(output/SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(meta["id"],KERNEL_ID)
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["dataset_sources"],[])
            self.assertEqual(meta["model_sources"],[])
            self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
            self.assertIn("qualify_vision_target_names",code)
            self.assertIn("VISION_LORA_TARGET_PATTERN",code)
            self.assertIn("exact 16 original H96 visual adapter tensors",code)
            self.assertIn("model_chosen_click_hits_first_plus",code)
            self.assertIn("nonlocal model_point",code)
            self.assertIn("shutil.rmtree(root",code)
            self.assertNotIn("HF_HOME='/kaggle/working",code)
            with self.assertRaises(ValueError):
                make(source,output)


if __name__=="__main__":
    unittest.main()
