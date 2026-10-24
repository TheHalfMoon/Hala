"""H106 opt-in pinned joint visual+language source sampler contracts."""
import re
import unittest
from training.scripts.train_adapter import (
    JOINT_LORA_TARGET_PATTERN,LORA_TARGET_PATTERN,VISION_LORA_TARGET_PATTERN,
    qualify_joint_target_names,qualify_vision_target_names)
from h106.training_plan import STRATEGY,STEPS,H105_QUALIFIED_PLAN_SHA

def names():
    visual=[f"model.visual.blocks.{i}.attn.{x}"
            for i in range(20,24) for x in ("qkv","proj")]
    language=[f"model.language_model.layers.{i}.self_attn.{x}"
              for i in range(18) for x in ("q_proj","v_proj")]
    return visual+language


class H106JointContracts(unittest.TestCase):
    def test_exact_source_bound_vision_and_language_modules(self):
        joint,visual,lang=qualify_joint_target_names(names())
        self.assertEqual(len(visual),8)
        self.assertEqual(len(lang),36)
        self.assertEqual(len(joint),44)
        self.assertTrue(all(re.fullmatch(JOINT_LORA_TARGET_PATTERN,x) for x in joint))
    def test_missing_visual_or_language_scope_refused(self):
        base=names()
        with self.assertRaises(ValueError):
            qualify_joint_target_names(base[1:])
        with self.assertRaises(ValueError):
            qualify_joint_target_names(base[:8])
    def test_unknown_visual_blocks_excluded(self):
        base=names()+["model.visual.blocks.19.attn.qkv",
                      "model.language_model.layers.1.mlp.down_proj"]
        joint,_,_=qualify_joint_target_names(base)
        self.assertEqual(len(joint),44)
        self.assertNotIn("model.visual.blocks.19.attn.qkv",joint)
    def test_h99_vision_only_pattern_unchanged(self):
        self.assertEqual(
          qualify_vision_target_names(names()),
          tuple(sorted(names()[:8])))
        self.assertTrue(re.fullmatch(VISION_LORA_TARGET_PATTERN,names()[0]))
        self.assertTrue(re.fullmatch(LORA_TARGET_PATTERN,names()[8]))
        self.assertFalse(re.fullmatch(VISION_LORA_TARGET_PATTERN,names()[8]))
    def test_pinned_original_h105_trained_prefix_identity(self):
        self.assertEqual(STEPS,256)
        self.assertEqual(len(H105_QUALIFIED_PLAN_SHA),64)
        self.assertIn("h106.",STRATEGY)

if __name__=="__main__":unittest.main()
