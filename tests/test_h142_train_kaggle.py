"""H142 original private GPU mount and free-T4-only training safeguards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h142.train_gpu import find_input_sources,TRAIN_SHA,DEV_SHA
from h142.train_kaggle.kaggle_private_train import find_adapter,SOURCE_SHA

class H142KagglePrivateTrainingAdmission(unittest.TestCase):
    def test_kernel_is_private_exact_new_original_data_only(self):
        path=Path(__file__).parent.parent/"h142"/"train_kaggle"/"kernel-metadata.json"
        meta=json.loads(path.read_text("utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"],
                         ["abdulazizshehri/hala-h142-private-visible-heading-sft-v1"])
        self.assertEqual(meta["kernel_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(len(SOURCE_SHA),40)
        self.assertEqual(len(TRAIN_SHA),64)
        self.assertEqual(len(DEV_SHA),64)

    def test_missing_real_train_dev_or_checkpoint_refuses_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch("h142.train_gpu.DATA_ROOT",root):
                with self.assertRaisesRegex(ValueError,"exactly original TRAIN64"):
                    find_input_sources()
            with self.assertRaisesRegex(ValueError,"exactly one"):
                find_adapter(root)
            (root/"h140-private-original-v2-rank8-lora").mkdir()
            with self.assertRaisesRegex(ValueError,"weights missing"):
                find_adapter(root)
if __name__=="__main__":
    unittest.main()
