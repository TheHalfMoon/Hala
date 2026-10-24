"""H96 conservative vision LoRA source and private provider admission tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from h96.research_probe import require_private_vision_research,vision_pilot_config
from training.scripts.train_adapter import (
    LORA_TARGET_PATTERN,VISION_LORA_TARGET_PATTERN,
    qualify_vision_target_names,VISION_LORA_SCOPE_NONCE)

ENV={"KAGGLE_KERNEL_RUN_TYPE":"Batch",
     "HALA_NONCOMMERCIAL_RESEARCH":"1",
     "HALA_REMOTE_TRAINING_ALLOWED":"1"}
VISUAL=tuple(
    f"model.visual.blocks.{i}.attn.{layer}"
    for i in range(20,24) for layer in ("qkv","proj"))

class IsolatedVisionLoRATests(unittest.TestCase):
    def test_qualified_late_visual_targets_exactly_eight(self):
        all_names=list(VISUAL)+["model.language_model.layers.0.self_attn.q_proj",
                                "model.visual.blocks.19.attn.qkv"]
        self.assertEqual(set(qualify_vision_target_names(all_names)),set(VISUAL))
    def test_missing_or_duplicated_vision_projection_refused(self):
        for bad in (list(VISUAL)[:-1], list(VISUAL)+[VISUAL[0]],
                    [name for name in VISUAL if ".20." not in name]):
            with self.subTest(count=len(bad)),self.assertRaises(ValueError):
                qualify_vision_target_names(bad)
    def test_older_language_lora_scope_is_preserved(self):
        self.assertEqual(LORA_TARGET_PATTERN,
                         r".*language_model.*\.(q_proj|v_proj)$")
        self.assertIn("visual",VISION_LORA_TARGET_PATTERN)
        self.assertIn("h96",VISION_LORA_SCOPE_NONCE)
    def test_real_h94_config_changes_only_one_step_and_audit(self):
        repo=Path(__file__).resolve().parents[1]
        from h94.research_train import research_configuration
        original=research_configuration(repo)
        new=vision_pilot_config(repo)
        self.assertEqual(new["max_steps"],1)
        self.assertEqual(new["audit_records"],4)
        self.assertEqual(
            {k:v for k,v in original.items() if k not in ("max_steps","audit_records")},
            {k:v for k,v in new.items() if k not in ("max_steps","audit_records")})
    def test_require_real_noncommercial_kaggle(self):
        require_private_vision_research(env=ENV,operating_system="Linux",
            scratch="/kaggle/tmp/h96",output="/kaggle/working/h96-adapter")
        for changes in ({"HALA_NONCOMMERCIAL_RESEARCH":"0"},
                        {"HALA_REMOTE_TRAINING_ALLOWED":""},
                        {"KAGGLE_KERNEL_RUN_TYPE":""}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                require_private_vision_research(env={**ENV,**changes},
                  operating_system="Linux",
                  scratch="/kaggle/tmp/h96",output="/kaggle/working/h96-adapter")
    def test_no_windows_or_non_kaggle_or_overwrite(self):
        for system,scratch,output in (
            ("Windows","/kaggle/tmp/h96","/kaggle/working/h96"),
            ("Linux","/tmp/h96","/kaggle/working/h96"),
            ("Linux","/kaggle/tmp/h96","/tmp/h96")):
            with self.subTest(system=system),self.assertRaises(ValueError):
                require_private_vision_research(env=ENV,operating_system=system,
                                               scratch=scratch,output=output)
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                require_private_vision_research(env=ENV,operating_system="Linux",
                    scratch="/kaggle/tmp/h96",output=td)

if __name__=="__main__":
    unittest.main()
