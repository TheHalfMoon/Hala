"""Guard H426 bounded synthetic model evaluation before any GPU or weights."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h426.synthetic_inference_eval import (
    MAX_DEV_INDICES, validate_readonly_eval, evaluate_already_loaded_model,
)

ROOT=Path(__file__).resolve().parents[1]
CORRECT_ENV = {
    "HALA_H426_SYNTHETIC_READONLY_EVAL": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


class H426ReadOnlyGPUPreflight(unittest.TestCase):
    def test_bounded_explicit_optin_and_full_offline_flags(self):
        with patch.dict(os.environ,CORRECT_ENV,clear=True):
            for n in (1,4,32):
                self.assertEqual(validate_readonly_eval(n),n)
            for n in (0,-1,33,MAX_DEV_INDICES+1,True,1.1,"4",None):
                with self.subTest(n=n), self.assertRaises(ValueError):
                    validate_readonly_eval(n)

    def test_missing_readonly_optin_rejected(self):
        for key in CORRECT_ENV:
            env=dict(CORRECT_ENV)
            del env[key]
            with patch.dict(os.environ,env,clear=True):
                with self.subTest(missing=key),self.assertRaises(ValueError):
                    validate_readonly_eval(1)

    def test_no_cuda_environment_rejected_before_model_access(self):
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with patch.dict(os.environ,{
                **CORRECT_ENV,"KAGGLE_KERNEL_RUN_TYPE":"Interactive"},clear=True):
            with patch.dict(sys.modules,{"torch":fake_torch}):
                with self.assertRaisesRegex(ValueError,"owner FREE Kaggle T4"):
                    evaluate_already_loaded_model(None,None,1)

    def test_never_launches_or_saves_a_model_or_original_source(self):
        source=(ROOT/"h426/synthetic_inference_eval.py").read_text("utf8")
        self.assertIn('generate("dev", index, domain)',source)
        self.assertIn('score_dev_pair(index,domain,predictions)',source)
        self.assertIn("torch.inference_mode()",source)
        self.assertIn('"HALA_H426_SYNTHETIC_READONLY_EVAL"',source)
        self.assertIn('"HF_HUB_OFFLINE"',source)
        self.assertIn('"TRANSFORMERS_OFFLINE"',source)
        self.assertIn("model_or_processor_sha_authenticated_here",source)
        for forbidden in ("model.save_pretrained(", "optimizer.step(",
                          "torch.optim.", "model.train(", "kaggle kernels push",
                          "requests.get(", "subprocess.run("):
            self.assertNotIn(forbidden,source)


if __name__=="__main__":
    unittest.main()
