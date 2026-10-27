"""H153 frozen external 2B source plus paired genuine neural controller gates."""
import json
import unittest
from pathlib import Path
from collections import Counter
from h153.collect_shifted_source import source_case,OFFSET,CASES,TRANSFORMS
from h153.owner_neural_controller_ab import valid_point
from h153.gpu_probe import (
    IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,
    OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA
)
class H153NewPairedNeuralEvaluationTests(unittest.TestCase):
    def test_new_original_holdout_and_language_goal_balance(self):
        self.assertEqual(OFFSET,184000)
        self.assertEqual(CASES,16)
        rows=[source_case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in rows),
                         {("ar","doing"):4,("en","doing"):4,
                          ("ar","done"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in rows}),4)
        with self.assertRaises(ValueError):
            source_case(174000)
    def test_admission_unchanged_model_no_paid_gpu_or_old_blind(self):
        path=Path(__file__).parent.parent/"h153"/"kaggle_probe"/"kernel-metadata.json"
        data=json.loads(path.read_text("utf8"))
        self.assertTrue(data["is_private"])
        self.assertEqual(data["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(data["dataset_sources"],[
            "abdulazizshehri/hala-h153-private-neural-paired-gui-v1"])
        self.assertEqual(data["kernel_sources"],[])
        for value in (IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,
                      OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA):
            self.assertEqual(len(value),64)
    def test_nonfinite_or_outside_viewport_model_point_rejected(self):
        self.assertTrue(valid_point([450,200]))
        for x in (None,[-1,250],[1000,100],
                  [float("nan"),250],["450","200"],{"x":1}):
            self.assertFalse(valid_point(x))
if __name__=="__main__":
    unittest.main()
