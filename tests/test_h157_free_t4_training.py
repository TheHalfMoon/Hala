"""H157 true free-T4 training admission and retention schedule (CPU-only)."""
import inspect
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from h157 import train_gpu


class H157RealSpatialTrainingTests(unittest.TestCase):
    def test_spatial_schedule_uses_only_280_new_train_examples(self):
        tasks=("click-test","click-button","enter-text",
               "enter-text-dynamic","choose-list")
        cases=[{"task":task} for task,n in zip(tasks,(64,64,64,64,24))
               for _ in range(n)]
        order=train_gpu.train_schedule(cases)
        self.assertEqual(len(order),304)
        self.assertEqual(Counter(cases[i]["task"] for i in order),{
            "click-test":64,"click-button":64,
            "enter-text":64,"enter-text-dynamic":64,"choose-list":48})
        self.assertEqual(len(set(order)),280)
        self.assertEqual(train_gpu.STEPS,152)
        self.assertEqual(train_gpu.GRAD_ACCUM,2)
        self.assertEqual(order,train_gpu.train_schedule(cases))

    def test_verified_model_and_dataset_provenance_pin(self):
        for constant in (train_gpu.SOURCE_SHA,train_gpu.SOURCE_ZIP_SHA,
                         train_gpu.START_ADAPTER_SHA,
                         train_gpu.START_ADAPTER_ZIP_SHA):
            self.assertEqual(len(constant),64)
            self.assertTrue(all(c in "0123456789abcdef" for c in constant))
        self.assertNotEqual(train_gpu.SOURCE_SHA,train_gpu.START_ADAPTER_SHA)
        self.assertEqual(train_gpu.MODEL_FILES,
                         {"README.md","adapter_config.json",
                          "adapter_model.safetensors"})

    def test_bad_source_rejected_before_any_gpu_import(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                train_gpu.read_source(Path(folder)/"no-source.zip")
        self.assertNotIn("H140",inspect.getsource(train_gpu.train_schedule))


if __name__=="__main__":
    unittest.main()
