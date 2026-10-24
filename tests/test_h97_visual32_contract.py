"""H97 source-bound visual-only 32-step CPU regression tests."""
import inspect
import unittest
from pathlib import Path
from h97.research_train import STEPS,training_config
from h96.research_probe import vision_pilot_config
from training.scripts.train_adapter import VISION_LORA_SCOPE_NONCE,qualify_vision_target_names,train

class H97VisualStudyTests(unittest.TestCase):
    def test_pinned_32_step_coordinate_weighted_schedule(self):
        repo=Path(__file__).resolve().parents[1]
        original=vision_pilot_config(repo)
        modified=training_config(repo)
        self.assertEqual(STEPS,32)
        self.assertEqual(modified["max_steps"],32)
        self.assertEqual({k:v for k,v in modified.items() if k!="max_steps"},
                         {k:v for k,v in original.items() if k!="max_steps"})
        self.assertEqual(modified["coordinate_weight"],8.0)
    def test_default_language_stays_unchanged(self):
        self.assertEqual(inspect.signature(train).parameters["vision_lora"].default,False)
        self.assertIn("vision-last-four",VISION_LORA_SCOPE_NONCE)
    def test_exact_eight_visual_and_no_language_module(self):
        layers=[f"model.visual.blocks.{i}.attn.{name}"
                for i in range(20,24) for name in ("qkv","proj")]
        qualified=qualify_vision_target_names(layers+["model.language_model.layers.0.self_attn.q_proj"])
        self.assertEqual(set(qualified),set(layers))
        self.assertEqual(len(qualified),8)

if __name__=="__main__":
    unittest.main()
