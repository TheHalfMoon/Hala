"""Guard H426 bounded synthetic model evaluation before any GPU or weights."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h426.synthetic_inference_eval import (
    MAX_DEV_INDICES, validate_readonly_eval, evaluate_already_loaded_model,
    compare_readonly_synthetic_reports,
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

    def test_two_sequential_readonly_models_are_paired_by_synthetic_image(self):
        def make_report(which):
            rows=[]
            for i in range(6):
                rows.append({
                    "image_sha256": hashlib.sha256(
                        f"invented-h426-matched-image-{i}".encode("ascii")
                    ).hexdigest(),
                    "domain": ("slider", "slide_resize", "text_highlight")[i%3],
                    "source_family": "invented-ui-family-3",
                    "languages": ["en","ar"],
                    "correct": [bool(which),True],
                })
            return {
                "schema": "hala.h426.invented-dev-readonly-model-eval.v1",
                "synthetic_dev_author_visible":True,
                "trained_or_modified_model_here":False,
                "model_weights_saved_or_published":False,
                "independent_external_ood":False,
                "release_qualified":False,
                "synthetic_per_image_binary_outcomes":rows,
            }
        baseline=make_report(False)
        candidate=make_report(True)
        result=compare_readonly_synthetic_reports(baseline,candidate)
        self.assertEqual(result["image_units"],6)
        self.assertEqual(result["baseline_correct_actions"],6)
        self.assertEqual(result["candidate_correct_actions"],12)
        self.assertEqual(result["candidate_only_both_correct"],6)
        self.assertEqual(result["action_accuracy_difference_candidate_minus_baseline"],0.5)
        self.assertEqual(result["evaluation_source"],
                         "invented_author_visible_not_external_ood")
        self.assertFalse(result["model_predictions_verified_here"])
        self.assertFalse(result["independent_blind_external_ood_attested_here"])
        self.assertFalse(result["model_or_release_qualified"])
        self.assertNotIn("raw_prediction",json.dumps(result))
        baseline["synthetic_per_image_binary_outcomes"].reverse()
        self.assertEqual(
            compare_readonly_synthetic_reports(baseline,candidate),result
        )

    def test_reject_swapped_extra_duplicate_and_unverified_source(self):
        def trial():
            image=lambda i:hashlib.sha256(
                ("h426-private-synthetic-"+str(i)).encode("ascii")
            ).hexdigest()
            return {
                "schema":"hala.h426.invented-dev-readonly-model-eval.v1",
                "synthetic_dev_author_visible":True,
                "trained_or_modified_model_here":False,
                "model_weights_saved_or_published":False,
                "independent_external_ood":False,
                "release_qualified":False,
                "synthetic_per_image_binary_outcomes":[
                    {"image_sha256":image(i),"domain":"slider",
                     "source_family":"invented-ui-family-3",
                     "languages":["en","ar"],"correct":[True,False]}
                    for i in range(2)],
            }
        correct=trial()
        invalid=[]
        r=trial()
        r["synthetic_per_image_binary_outcomes"][0]["source_family"]="wrong-family"
        invalid.append(r)
        r=trial()
        r["synthetic_per_image_binary_outcomes"][0]["image_sha256"]="0"*64
        invalid.append(r)
        r=trial()
        r["synthetic_per_image_binary_outcomes"][0]["correct"]=[True,1]
        invalid.append(r)
        r=trial()
        r["synthetic_per_image_binary_outcomes"][0]["languages"]=["ar","en"]
        invalid.append(r)
        r=trial()
        r["synthetic_per_image_binary_outcomes"][1]=copy.deepcopy(
            r["synthetic_per_image_binary_outcomes"][0])
        invalid.append(r)
        r=trial()
        r["independent_external_ood"]=True
        invalid.append(r)
        r=trial()
        r["synthetic_per_image_binary_outcomes"].append(
            copy.deepcopy(r["synthetic_per_image_binary_outcomes"][0]))
        invalid.append(r)
        for wrong in invalid:
            with self.subTest(wrong=str(wrong)[:140]),self.assertRaises(ValueError):
                compare_readonly_synthetic_reports(correct,wrong)

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
