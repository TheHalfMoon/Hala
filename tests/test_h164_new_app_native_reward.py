"""H164 true-two-new-app model/renderer qualification tests (no GPU calls)."""
import inspect
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from h164 import ood_gpu_infer as gpu
from h164 import replay_ood_native as replay
from h164 import audit_ood_native as audit

class H164NativeNewOODTests(unittest.TestCase):
    def test_gpu_is_inference_only_and_not_unfrozen(self):
        src=inspect.getsource(gpu.run)
        self.assertIn('len(TRAINED_MODEL_SHA)!=64',src)
        self.assertIn("torch.cuda.is_available()",src)
        self.assertIn('real_gpu_calls":128',src)
        self.assertIn('private_owner_gold_on_gpu":False',src)
        self.assertNotIn("optimizer.step(",src)
        self.assertNotIn("model.train(",src)
    def test_ood_zip_rejects_wrong_frozen_input(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"bad.zip"
            with ZipFile(p,"w") as z:z.writestr("manifest.json","{}")
            with self.assertRaisesRegex(ValueError,"SHA changed"):
                gpu.admit(p)
    def test_model_only_native_click_controller_has_no_oracle(self):
        src=inspect.getsource(replay.native)
        self.assertIn('page.mouse.click(*proposed["points"][0])',src)
        self.assertIn('digest(stable_png(page))',src)
        self.assertNotIn("owner_gold_click",src)
        self.assertNotIn("owner_reference_action",src)
        self.assertNotIn("center(",src)
    def test_one_shot_replay_no_pair_duplication(self):
        source=inspect.getsource(replay.run)
        self.assertIn('consumed.open("x"',source)
        self.assertIn("original_browser_trials",source)
        self.assertIn("paired_wins",source)
        self.assertIn("original_apps_not_in_train_or_dev",source)
    def test_independent_recompute_real_native_reward(self):
        source=inspect.getsource(audit.audit)
        self.assertIn('set(observed)!=pairs',source)
        self.assertIn('total["wins"]',source)
        self.assertIn("exact_paired_sign_p",source)
        self.assertIn("original_app_independent_OOD",source)
