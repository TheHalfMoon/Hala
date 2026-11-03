"""H403 original bilingual absent-target source and policy contracts."""
import importlib.util
import json
import unittest
from pathlib import Path
from h403.kaggle.h403_safe_abstention_lora import (
    MODEL,REV,PARENT_ADAPTER_SHA,parse,STEPS,TRAIN_N,DEV_N,AUDIT_N)

class H403StrictResponseTests(unittest.TestCase):
    def test_strict_point_and_refusal_only(self):
        self.assertEqual(parse('{"abstain":true}'),("abstain",None))
        self.assertEqual(parse('{"point_2d":[123,999]}'),("point",[123.,999.]))
        for invalid in ('{"abstain":false}','{"point_2d":[1001,0]}',
                        '{"point_2d":[0,NaN]}','{"point_2d":["5",2]}',
                        '{"point_2d":[1,2],"extra":1}','not-json'):
            self.assertEqual(parse(invalid)[0],"invalid")
    def test_private_source_pinned_and_not_official_hala(self):
        self.assertEqual(MODEL,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(REV,"89644892e4d85e24eaac8bacfd4f463576704203")
        self.assertEqual(len(PARENT_ADAPTER_SHA),64)
        self.assertEqual((STEPS,TRAIN_N,DEV_N,AUDIT_N),(48,48,24,24))
        metadata=json.loads((Path(__file__).resolve().parents[1]/
             "h403"/"kaggle"/"kernel-metadata.json").read_text(encoding="utf8"))
        self.assertIs(metadata["is_private"],True)
        self.assertIs(metadata["enable_gpu"],True)
        self.assertEqual(metadata["dataset_sources"],[])
        self.assertEqual(metadata["model_sources"],[])
        self.assertEqual(metadata["kernel_sources"],
            ["abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora"])

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Optional first-party rendered synthetic images use Pillow")
class H403OriginalSourceTests(unittest.TestCase):
    def test_three_splits_balanced_no_heldout_reuse(self):
        from h403.kaggle.h403_safe_abstention_lora import (
            make_case,validate_groups)
        train=[make_case(i,40311) for i in range(TRAIN_N)]
        dev=[make_case(i,40321) for i in range(DEV_N)]
        audit=[make_case(i,40331,"shifted") for i in range(AUDIT_N)]
        keys=validate_groups(train,dev,audit)
        self.assertEqual(set(keys),{"train","dev","audit_author_visible"})
        for rows in (train,dev,audit):
            self.assertEqual(sum(r[4] for r in rows),len(rows)//2)
            self.assertEqual(sum(r[3] for r in rows),len(rows)//2)
            for image,prompt,gold,lang,missing,target in rows:
                self.assertEqual(image.size,(224,224))
                self.assertIn('{"abstain":true}',prompt)
                if missing:self.assertEqual(gold,'{"abstain":true}')
                else:self.assertEqual(json.loads(gold),{"point_2d":target})

if __name__=="__main__":
    unittest.main()
