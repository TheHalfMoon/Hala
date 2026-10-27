"""H151 frozen source model inference gate, no hidden oracle."""
import json
import unittest
from pathlib import Path
from h151.gpu_probe import (
    IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,
    OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA
)
from h151.owner_neural_drag_eval import valid_point
class H151FrozenModelInputChecks(unittest.TestCase):
    def test_only_owner_private_free_t4(self):
        meta=json.loads((Path(__file__).parent.parent/
                         "h151"/"kaggle_probe"/"kernel-metadata.json").read_text())
        self.assertTrue(meta["is_private"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"],[
            "abdulazizshehri/hala-h151-private-frozen-model-visual-drop-v1"])
        self.assertEqual(meta["kernel_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        for sha in (IMAGE_ARCHIVE_SHA,IMAGE_MANIFEST_SHA,
                    OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA):
            self.assertEqual(len(sha),64)
    def test_nonfinite_or_out_of_original_viewport_model_points_rejected(self):
        self.assertTrue(valid_point([400,300]))
        for point in (None,[960,100],[-2,200],
                      [float("nan"),150],["480","180"],{"x":10}):
            self.assertFalse(valid_point(point))
if __name__=="__main__":
    unittest.main()
