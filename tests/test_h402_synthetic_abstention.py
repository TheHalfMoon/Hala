"""H402 developer-invented abstention and native-action scoring guards."""
import importlib.util
import json
import unittest
from pathlib import Path
from h402.kaggle.h402_synthetic_negative_native import parse,scoring

class H402ScoringTests(unittest.TestCase):
    def test_exact_no_target_abstention_only(self):
        self.assertEqual(parse('{"abstain":true}'),("abstain",None))
        self.assertTrue(scoring("abstain",None,None)["correct"])
        self.assertFalse(scoring("point",[500.,500.],None)["correct"])
        self.assertTrue(scoring("point",[500.,500.],None)["false_click"])
    def test_malformed_answers_never_count_as_correct(self):
        for raw in ('not-json','{"abstain":false}', '{"point_2d":[9999,50]}',
                    '{"point_2d":["200",400]}','{"point_2d":[NaN,0]}',
                    '{"point_2d":[100,20],"abstain":true}'):
            self.assertEqual(parse(raw)[0],"invalid")
    def test_existing_target_requires_right_rectangle(self):
        target=[500.,500.,14]
        self.assertTrue(scoring("point",[500.,500.],target)["correct"])
        self.assertFalse(scoring("point",[100.,100.],target)["correct"])
        self.assertFalse(scoring("abstain",None,target)["correct"])
    def test_kaggle_is_private_with_only_pretrained_adapter_kernel(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/"h402"/"kaggle"/"kernel-metadata.json").read_text("utf8"))
        self.assertIs(cfg["is_private"],True)
        self.assertEqual(cfg["dataset_sources"],[])
        self.assertEqual(cfg["kernel_sources"],
            ["abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora"])
        self.assertEqual(cfg["model_sources"],[])

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Original rectangle generation uses optional PIL")
class H402InventedSourceTests(unittest.TestCase):
    def test_positive_and_absent_linguistic_groups_are_balanced(self):
        from h402.kaggle.h402_synthetic_negative_native import sample,checksum
        items=[sample(i) for i in range(24)]
        self.assertEqual(sum(row[4] for row in items),12)
        for absent in (True,False):
            subset=[row for row in items if row[4]==absent]
            self.assertEqual(len(subset),12)
            self.assertEqual(sum(row[3]==0 for row in subset),6)
            self.assertEqual(sum(row[3]==1 for row in subset),6)
            for image,_,target,_,_ in subset:
                self.assertEqual(image.size,(224,224))
                self.assertIs(target is None,absent)
        ids={checksum(row[0].tobytes()+row[1].encode("utf8")) for row in items}
        self.assertEqual(len(ids),24)
if __name__=="__main__":
    unittest.main()
