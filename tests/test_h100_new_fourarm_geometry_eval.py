"""H100 independent sample/H99 source-checkpoint/strict scorer guards."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from h100 import eval as h100
from h61.official_eval import parse_action
from h99.research_train import STEPS as H99_STEPS
from h99.sampler import STRATEGY as H99_STRATEGY

class H100FourArmFreshSourceTests(unittest.TestCase):
    def test_disjoint_from_three_prior_official_public_eval_samples(self):
        self.assertEqual((h100.START_INDEX,h100.PER_DOMAIN,h100.FETCH_COUNT),(12,4,16))
        self.assertEqual(h100.ARMS,("base","h94_language","h97_vision","h99_balanced_vision"))
        self.assertEqual(len(h100.DOMAINS),4)
        self.assertEqual(H99_STEPS,128)
    def test_real_h99_final_bytes_are_immutable_pinned(self):
        self.assertEqual(h100.H99_KERNEL,"hala-h99-private-balanced-visual-128")
        self.assertEqual(h100.H99_ADAPTER_SHA,
            "701109c5fbf61df8a101ae6c23093e6b1e35224b701a4983c5f4f0784d915eb1")
        self.assertEqual(h100.H99_CHECKPOINT_SHA,
            "d15c8bf63b1ebb48897c597c5502b30457e962671d5c5251e8bce5c2cf51a613")
        self.assertEqual(h100.H99_PLAN_SHA,
            "76ea4d7105fd2e3cce02d9e009e3df6c8fce20e69e877495a2320b6ae86c7c63")
        self.assertIn("h99",H99_STRATEGY)
    def test_no_owner_private_checkpoint_denied(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                h100.admit_h99_balanced_vision(Path(td),Path(td))
    def test_original_strict_json_score_not_modified(self):
        action='{"type":"drag","start_point":[30,40],"end_point":[60,80],"coordinate_space":"source_image_pixels","ordered":true}'
        self.assertIsNotNone(parse_action(action,1024,640))
        fenced="```json\n"+action+"\n```"
        self.assertIsNone(parse_action(fenced,1024,640))
        self.assertIsNotNone(parse_action(h100.normalize_fenced_json(fenced),1024,640))
    def test_private_noncommercial_runtime_only(self):
        correct={"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"}
        h100.require_owner_private_runtime(env=correct,system="Linux",
            work="/kaggle/tmp/h100",output="/kaggle/working/h100")
        for env,system,work,output in (
            ({}, "Linux","/kaggle/tmp/h100","/kaggle/working/h100"),
            (correct,"Windows","/kaggle/tmp/h100","/kaggle/working/h100"),
            (correct,"Linux","/tmp/h100","/kaggle/working/h100"),
            (correct,"Linux","/kaggle/tmp/h100","/tmp/h100")):
            with self.subTest(env=env,system=system),self.assertRaises(ValueError):
                h100.require_owner_private_runtime(env=env,system=system,work=work,output=output)

if __name__=="__main__":
    unittest.main()
