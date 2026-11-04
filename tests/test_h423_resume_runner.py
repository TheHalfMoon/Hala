"""H423 genuine source-checkpoint rescue runner integrity tests.

All mock safetensors are synthetic TEST FIXTURES, never trained model
evidence. The runner must reject real unqualified provider logs/weights.
"""
import json
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from h423.qualify_h419_checkpoint import qualify,store
from h423.kaggle import h423_h419_checkpoint_resume as g
from test_h423_checkpoint_qualification import fake_checkpoint


class H423SegmentedNativeRescueSource(unittest.TestCase):
    def verified_source(self,folder):
        checkpoint,original_log=fake_checkpoint(folder)
        proof=qualify(checkpoint,original_log,"7")
        blob=Path(folder)/"manifest"
        store(blob,proof)
        folder=Path(folder)
        copies={
            g.PARTIAL_ADAPTER_FILE:folder/g.PARTIAL_ADAPTER_FILE,
            g.PARTIAL_CONFIG_FILE:folder/g.PARTIAL_CONFIG_FILE,
            g.PARTIAL_GPU_LOG:folder/g.PARTIAL_GPU_LOG,
            g.PARTIAL_EVIDENCE:folder/g.PARTIAL_EVIDENCE,
            g.PARTIAL_EVIDENCE_SHA:folder/g.PARTIAL_EVIDENCE_SHA,
        }
        shutil.copyfile(checkpoint/"adapter_model.safetensors",copies[g.PARTIAL_ADAPTER_FILE])
        shutil.copyfile(checkpoint/"adapter_config.json",copies[g.PARTIAL_CONFIG_FILE])
        shutil.copyfile(original_log,copies[g.PARTIAL_GPU_LOG])
        shutil.copyfile(blob/"receipt.json",copies[g.PARTIAL_EVIDENCE])
        shutil.copyfile(blob/"receipt.sha256",copies[g.PARTIAL_EVIDENCE_SHA])
        return copies

    def test_resume_runner_plan_preserves_original_frozen_shuffled_order(self):
        self.assertEqual(g.H419_STEPS,9452)
        self.assertEqual(g.GRAD_ACCUM,4)
        self.assertEqual(g.H423_T4_SEGMENT_STEPS,1500)
        self.assertEqual(g.H419_TRAINING_PASSES,37808)
        self.assertEqual(g.H419_UNIQUE_IMAGES,30976)
        self.assertNotEqual(g.OUT.name,"h419-owner-private-native-drag")
        self.assertIn("h423",g.OUT.name)
        self.assertFalse(g.H423_STEP_STATUS.endswith("RELEASE"))
        self.assertIn("RESET_ADAMW",g.H423_STEP_STATUS)

    def test_true_partial_1500_step_checkpoint_qualified_and_copied_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.verified_source(tmp)
            out=Path(tmp)/"separate-output"
            with patch.object(g,"only_input_file",side_effect=lambda name:paths[name]),\
                 patch.object(g,"OUT",out):
                receipt,sha,actual=g.exact_real_original_checkpoint()
            self.assertEqual(receipt["original_model_checkpoint_step"],1500)
            self.assertEqual(receipt["verified_original_source_gradient_examples"],6000)
            self.assertEqual(sha,sha256(paths[g.PARTIAL_EVIDENCE].read_bytes()).hexdigest())
            self.assertEqual(g.sha_file(actual/"adapter_model.safetensors"),
                             receipt["checkpoint_adapter_model_sha256"])
            self.assertTrue((actual/"adapter_config.json").is_file())

    def test_substituted_original_lora_bytes_refused_even_with_valid_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.verified_source(tmp)
            original=paths[g.PARTIAL_ADAPTER_FILE].read_bytes()
            paths[g.PARTIAL_ADAPTER_FILE].write_bytes(original[:-1]+b"!")
            with patch.object(g,"only_input_file",side_effect=lambda name:paths[name]),\
                 patch.object(g,"OUT",Path(tmp)/"separate-output"):
                with self.assertRaisesRegex(ValueError,"bytes/config/GPU progress"):
                    g.exact_real_original_checkpoint()

    def test_unverified_provider_log_wrong_sha_refused_before_GPU_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.verified_source(tmp)
            with paths[g.PARTIAL_GPU_LOG].open("a") as f:
                f.write("\nH419_REAL_ORIGINAL_30000_NATIVE_GRADIENT 1950 / 9452 "
                        "image_conditioned_samples_seen 7800 loss 0.001\n")
            with patch.object(g,"only_input_file",side_effect=lambda name:paths[name]),\
                 patch.object(g,"OUT",Path(tmp)/"separate-output"):
                with self.assertRaisesRegex(ValueError,"bytes/config/GPU progress"):
                    g.exact_real_original_checkpoint()

    def test_does_not_falsely_qualify_complete_original_training_or_ood(self):
        code=Path(g.__file__).read_text(encoding="utf8")
        self.assertIn('"uninterrupted_9452_step_optimizer_history":False',code)
        self.assertIn('"optimizer_moments_restored_from_H419":False',code)
        self.assertIn('"AdamW_optimizer_reset_on_resume":True',code)
        self.assertIn('"independent_external_ood":False',code)
        self.assertIn('"model_release_qualified":False',code)
        self.assertIn('for absolute_step in range(begin,end):',code)
        self.assertIn('order[absolute_step*GRAD_ACCUM+micro]',code)
        self.assertNotIn("source_only_research_gate=True",code)

if __name__=="__main__":
    unittest.main()
