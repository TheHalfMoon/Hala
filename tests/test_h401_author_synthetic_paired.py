"""Private H401 paired synthetic click evaluation safety, no model import."""
import importlib.util
import json
import unittest
from pathlib import Path
from h401.kaggle.h401_native_paired import (
    BASE,REV,ADAPTER_SHA,parse,hit)

class H401NativeParsingTests(unittest.TestCase):
    def test_only_exact_finite_normalized_point_json_is_allowed(self):
        self.assertEqual(parse('{"point_2d":[143,714]}'),[143.,714.])
        self.assertIsNone(parse('{"point_2d":[1,2],"confidence":1}'))
        self.assertIsNone(parse('{"point_2d":["1",2]}'))
        self.assertIsNone(parse('{"point_2d":[NaN,2]}'))
        self.assertIsNone(parse('{"point_2d":[1001,2]}'))
        self.assertIsNone(parse('extra_text {"point_2d":[1,2]}'))
        self.assertIsNone(parse('{"point_2d":[1]}'))
    def test_hit_requires_native_target_rectangle(self):
        target=[500.,500.]
        self.assertTrue(hit([500.,500.],target))
        self.assertFalse(hit(None,target))
        self.assertFalse(hit([800.,500.],target))
    def test_immutable_private_kaggle_adapter_source(self):
        path=Path(__file__).resolve().parents[1]/"h401"/"kaggle"/"kernel-metadata.json"
        metadata=json.loads(path.read_text("utf8"))
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["dataset_sources"],[])
        self.assertEqual(metadata["kernel_sources"],
                         ["abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora"])
        self.assertEqual(metadata["model_sources"],[])
        self.assertEqual(BASE,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(REV),40)
        self.assertEqual(len(ADAPTER_SHA),64)

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Optional first-party renderer unavailable")
class H401InventedInputsTests(unittest.TestCase):
    def test_new_author_exposed_sources_do_not_match_h399(self):
        from h401.kaggle.h401_native_paired import sample,fingerprint
        current=[sample(i,4901,"unseen") for i in range(18)]
        earlier=[*[sample(i,3981) for i in range(24)],
                 *[sample(i,3982) for i in range(6)],
                 *[sample(i,3983,"unseen") for i in range(6)]]
        self.assertEqual(len(set(map(fingerprint,current))),18)
        self.assertFalse(set(map(fingerprint,current)) & set(map(fingerprint,earlier)))
        self.assertEqual(sum(x[3]==0 for x in current),9)
        self.assertEqual(sum(x[3]==1 for x in current),9)
if __name__=="__main__":
    unittest.main()
