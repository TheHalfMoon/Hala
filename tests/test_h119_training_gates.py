"""H119 isolated optimizer and 1024-point contracts; CPU-only guards."""
import unittest
from pathlib import Path
from h106.research_train import config as h106_config
from h119.research_train import config as h119_config
from h119.training_plan import checkpoint_due,STEPS,PAIRS,get_plan
from training.preflight import TrainingConfig
from training.scripts.train_adapter import train,H119_FULL512_SCOPE_NONCE
from unittest.mock import patch

class H119TrainingGates(unittest.TestCase):
    def test_legacy_h106_stays_256_and_h119_only_1024(self):
        old=h106_config(Path("."))
        new=h119_config(Path("."))
        self.assertEqual(old["max_steps"],256)
        self.assertEqual(new["max_steps"],1024)
        self.assertEqual(old["lora_rank"],new["lora_rank"])
        self.assertEqual(old["coordinate_weight"],new["coordinate_weight"])
        self.assertEqual(TrainingConfig.from_dict(new).max_steps,STEPS)
        self.assertIn("h119",H119_FULL512_SCOPE_NONCE)

    def test_exact_only_sixteen_checkpoints(self):
        self.assertEqual([i for i in range(1,1025) if checkpoint_due(i)],
                         list(range(64,1025,64)))
        for bad in (0,-1,1025,True):
            with self.assertRaises(ValueError):checkpoint_due(bad)
        with self.assertRaises(ValueError):checkpoint_due(64,256)

    def test_source_plan_wrong_size_fails_without_reading_original(self):
        for n in (0,256,512,1023,1025):
            with self.assertRaises(ValueError):get_plan([],Path("."),n)
        self.assertEqual(PAIRS,512)

    def test_h119_cannot_mix_prior_weights_or_change_train_scope(self):
        for opts in (
            dict(joint_lora=False),
            dict(joint_lora=True,point_sft_path="legacy"),
            dict(joint_lora=True,conditioned_delta_sft_path="legacy"),
            dict(joint_lora=True,init_adapter="legacy"),
            dict(joint_lora=True,resume=True),
            dict(joint_lora=True,sampling_strategy=None),
            dict(joint_lora=True,vision_lora=True),
        ):
            with self.subTest(opts=opts):
                with self.assertRaises(ValueError):
                    train("not-a-config","not-source","no-manifest","no-baseline",
                          "no-output",robust_anchor_sft_path="private-source",**opts)
        with self.assertRaises(ValueError):
            train("not-a-config","not-source","no-manifest","no-baseline","no-output",
                  joint_lora=True,robust_anchor_sft_path="private-source",
                  sampling_strategy="hala.h106.joint-vision-language-h105-source-prefix256.v1",
                  stop_after_step=23)

if __name__=="__main__":unittest.main()
