"""H98 CPU-only pre-registration checks. No GPU/evaluation conclusions."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from h98 import eval as h98
from h61.official_eval import parse_action
from h95.eval import H94_ADAPTER_SHA

class ThreeArmNewSourceEvaluationTests(unittest.TestCase):
    def test_exact_disjoint_new_indices_and_arms(self):
        self.assertEqual((h98.START_INDEX,h98.PER_DOMAIN,h98.FETCH_COUNT),(8,4,12))
        self.assertEqual(h98.ARMS,("base","h94_language","h97_vision"))
        self.assertEqual(len(h98.DOMAINS),4)
    def test_pinned_exact_real_vision_and_language_sha(self):
        self.assertEqual(h98.H97_ADAPTER_SHA,
           "5dbc350c56dcdbbcec5feda97a300d925e48e7e6b11d5bdb51e3bdcd8632fedb")
        self.assertEqual(h98.H97_CHECKPOINT_SHA,
           "16eb940c91e7e4fc8599019a722819e8867ad9eabc60571336969fa5ebf101b1")
        self.assertEqual(H94_ADAPTER_SHA,
           "d73cc9a302782bf7eca08f434a2050a9c3f212368c25afe65014f71319eb673c")
        self.assertEqual(h98.STEPS,32)
    def test_requires_private_noncommercial_provider(self):
        correct={"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"}
        h98.require_owner_private_runtime(env=correct,system="Linux",
            work="/kaggle/tmp/h98",output="/kaggle/working/h98")
        for env,system,work,output in (
            ({},"Linux","/kaggle/tmp/h98","/kaggle/working/h98"),
            ({"KAGGLE_KERNEL_RUN_TYPE":"Batch"},"Linux","/kaggle/tmp/h98","/kaggle/working/h98"),
            (correct,"Windows","/kaggle/tmp/h98","/kaggle/working/h98"),
            (correct,"Linux","/home/h98","/kaggle/working/h98"),
            (correct,"Linux","/kaggle/tmp/h98","/home/h98")):
            with self.subTest(env=env,system=system),self.assertRaises(ValueError):
                h98.require_owner_private_runtime(env=env,system=system,work=work,output=output)
    def test_private_h97_checkpoint_required(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                h98.admit_h97_vision(Path(tmp),Path(tmp))
    def test_original_strict_scorer_not_relaxed(self):
        correct='{"type":"drag","start_point":[10,20],"end_point":[30,40],"coordinate_space":"source_image_pixels","ordered":true}'
        self.assertIsNotNone(parse_action(correct,1024,640))
        self.assertIsNone(parse_action("```json\n"+correct+"\n```",1024,640))
        self.assertIsNotNone(parse_action(h98.normalize_fenced_json(
            "```json\n"+correct+"\n```"),1024,640))

if __name__=="__main__":
    unittest.main()
