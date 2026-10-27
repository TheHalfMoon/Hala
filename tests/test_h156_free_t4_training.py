"""H156 CPU-only free-T4 research protocol and model/source separation."""
import json
import unittest

from h156 import train_gpu


class H156TrainGPUPreflight(unittest.TestCase):
    def test_frozen_admission_and_bounded_steps(self):
        self.assertEqual(len(train_gpu.SOURCE_SHA),64)
        self.assertEqual(train_gpu.STEPS,24)
        self.assertEqual(train_gpu.GRAD_ACCUM,2)
        self.assertEqual(len(train_gpu.ADAPTER_SHA),64)
        self.assertEqual(len(train_gpu.ADAPTER_ZIP_SHA),64)

    def test_visible_typed_action_prompt_and_expected_json(self):
        row={
            "instruction":"Select Ghana from the list and click Submit.",
            "training_supervision":{"actions":[
                {"kind":"select","point":[135,205],"text":"Ghana"},
                {"kind":"submit","point":[145,250]}]}
        }
        prompt=train_gpu.prompt(row)
        self.assertIn("960x720",prompt)
        self.assertIn("absolute pixels",prompt)
        self.assertIn("No DOM selectors",prompt)
        data=json.loads(train_gpu.label(row))
        self.assertEqual(data["actions"][0]["point"],[135,205])
        self.assertEqual(data["actions"][0]["text"],"Ghana")


if __name__=="__main__":
    unittest.main()
