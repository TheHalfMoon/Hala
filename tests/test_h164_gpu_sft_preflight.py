"""H164 future FREE-T4 SFT fail-closed source/model and no-eval-leak guards."""
import inspect
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from h164 import train_gpu as gpu
from h164.native_action_contract import parse_action

class H164GpuSafetyTests(unittest.TestCase):
    def test_exact_train_budget_is_all_384_new_scenes(self):
        self.assertEqual((gpu.STEPS,gpu.ACCUM,gpu.STEPS*gpu.ACCUM),(192,2,384))
        self.assertEqual(gpu.SEED,164)
        self.assertEqual(len(gpu.PARENT_LORA_SHA),64)
    def test_unqualified_source_fails_before_GPU(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"wrong.zip"
            with ZipFile(f,"w") as z:z.writestr("manifest.json","{}")
            with self.assertRaisesRegex(ValueError,"unqualified"):
                gpu.admit(f,"TRAIN")
            with self.assertRaisesRegex(ValueError,"unqualified"):
                gpu.admit(f,"DEV")
    def test_all_gold_only_in_training(self):
        src=inspect.getsource(gpu.admit)
        self.assertIn('if split=="TRAIN":',src)
        self.assertIn("DEV reference leaked",src)
        self.assertIn("supervision(row)",src)
        self.assertIn("DEV",src)
    def test_loss_masks_visual_prompt(self):
        src=inspect.getsource(gpu.assemble)
        self.assertIn('labels[:,:begin]=-100',src)
        self.assertIn("attention_mask",src)
        self.assertIn("supervision(row)",src)
    def test_model_does_not_read_ood_for_training(self):
        src=inspect.getsource(gpu.run)
        self.assertNotIn("ood_zip",src)
        self.assertIn("torch.cuda.is_available()",src)
        self.assertIn('torch.isfinite(loss)',src)
        self.assertIn('optim.step()',src)
        self.assertIn("source_train_zip_sha256",src)
    def test_new_native_typed_action_contract(self):
        self.assertEqual(parse_action('{"actions":[{"kind":"type","point":[50,40],"text":"Ana"}]}')["kind"],"type")
        self.assertFalse(parse_action('{"actions":[{"kind":"click","point":[1,2],"selector":"#button"}]}')["valid"])
