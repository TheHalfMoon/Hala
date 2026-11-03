"""H409 original-source two-family balanced LoRA safety preflight."""
import importlib.util
import json
import unittest
from pathlib import Path

from h409.kaggle.h409_balanced_two_family_lora import (
    BASE,REV,TRAIN_N,DEV_N,AUDIT_N,STEPS,parse,pick_family_kind)

class H409NoGPUProtocolTests(unittest.TestCase):
    def test_exact_kaggle_owner_private_only_h406_adapter(self):
        data=json.loads((Path(__file__).resolve().parents[1]/
           "h409"/"kaggle"/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(data["is_private"])
        self.assertTrue(data["enable_gpu"])
        self.assertEqual(data["dataset_sources"],[])
        self.assertEqual(data["model_sources"],[])
        self.assertEqual(data["kernel_sources"],[
            "abdulazizshehri/hala-h406-private-invented-ambiguity-lora"])
        self.assertEqual((BASE,REV,TRAIN_N,DEV_N,AUDIT_N,STEPS),(
            "Qwen/Qwen3-VL-2B-Instruct",
            "89644892e4d85e24eaac8bacfd4f463576704203",
            128,48,48,128))
    def test_each_family_training_oversamples_unique_and_retains_refusal(self):
        from collections import Counter
        seen=Counter(pick_family_kind(i,"train")[:2] for i in range(TRAIN_N))
        for family in ("button_ui","color_grid"):
            self.assertEqual(seen[(family,"unique")],32)
            self.assertEqual(seen[(family,"absent")],16)
            self.assertEqual(seen[(family,"ambiguous")],16)
    def test_h409_strict_json_rejects_dup_and_untrusted_number_forms(self):
        self.assertEqual(parse('{"abstain":true}'),("abstain",None))
        self.assertEqual(parse('{"point_2d":[0,1000]}'),("point",[0.,1000.]))
        for raw in ('{"abstain":false}','{"point_2d":[NaN,0]}',
                    '{"point_2d":[Infinity,1]}',
                    '{"point_2d":[true,500]}',
                    '{"point_2d":[1,1001]}',
                    '{"abstain":true,"abstain":true}',
                    '{"point_2d":[0,100],"\\u0070oint_2d":[500,500]}',
                    '{"point_2d":[0,100],"abstain":true}',"not JSON"):
            with self.subTest(raw=raw):
                self.assertEqual(parse(raw)[0],"invalid")

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Pillow optional first-party image generator")
class H409OriginalSourceTests(unittest.TestCase):
    def test_all_splits_first_party_disjoint_balanced_and_action_gold(self):
        from h409.kaggle.h409_balanced_two_family_lora import compose,freeze
        sets=[
            [compose(i,40911,"train") for i in range(TRAIN_N)],
            [compose(i,40921,"dev") for i in range(DEV_N)],
            [compose(i,40931,"audit") for i in range(AUDIT_N)]
        ]
        hashes=freeze(*sets)
        self.assertEqual(set(hashes),{"train","dev","audit"})
        self.assertEqual(len(set(hashes.values())),3)
        for rows in sets:
            for im,prompt,gold,family,kind,lang,boxes in rows:
                self.assertIn(family,("button_ui","color_grid"))
                self.assertIn('{"abstain":true}',prompt)
                self.assertEqual(len(boxes),{"unique":1,"absent":0,"ambiguous":2}[kind])
                self.assertEqual(im.size, (320,240) if family=="button_ui"
                                               else (224,224))
                if kind=="unique":
                    self.assertEqual(set(json.loads(gold)),{"point_2d"})
                else:
                    self.assertEqual(json.loads(gold),{"abstain":True})
    def test_h405_h406_h408_audit_images_are_not_training(self):
        from hashlib import sha256
        from h409.kaggle.h409_balanced_two_family_lora import compose
        from h408.kaggle.h408_private_gui_composition_eval import invent as old408
        from h405.kaggle.h405_synthetic_ambiguity_eval import invent as old405
        from h406.kaggle.h406_ambiguity_refusal_lora import generate as old406
        new=[compose(i,40911,"train") for i in range(TRAIN_N)]
        old=[
            *[old408(i) for i in range(36)],
            *[old405(i) for i in range(72)],
            *[old406(i,40631,"audit") for i in range(48)]]
        original=lambda row:sha256(row[0].tobytes()).hexdigest()
        self.assertFalse({original(x) for x in new}&
                         {original(x) for x in old})
        text=lambda row:sha256(
            row[0].tobytes()+row[1].encode("utf8")).hexdigest()
        self.assertFalse({text(x) for x in new}&{text(x) for x in old})

if __name__=="__main__":unittest.main()
