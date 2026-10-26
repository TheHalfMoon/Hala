"""V3 free Kaggle 160-step job consumes only source-qualified TRAIN+DEV."""
import json
import unittest
from pathlib import Path
from h140.train_kaggle_v3.kaggle_private_train import SOURCE_SHA, source_mounts

class H140V3PrivateTrainingKernelContracts(unittest.TestCase):
    def test_free_t4_private_and_exact_mounts(self):
        base=Path(__file__).parent.parent/"h140"/"train_kaggle_v3"
        meta=json.loads((base/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"],[
            "abdulazizshehri/hala-h140-private-train256-original",
            "abdulazizshehri/hala-h140-private-dev64-original"])
        self.assertEqual(meta["kernel_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(len(SOURCE_SHA),40)
        from h140.train_gpu_v3 import STEPS,GRAD_ACCUM
        self.assertEqual(STEPS*GRAD_ACCUM,640)
        self.assertEqual(STEPS,160)
        self.assertTrue(callable(source_mounts))

if __name__=="__main__":
    unittest.main()
