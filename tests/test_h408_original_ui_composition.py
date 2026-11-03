"""H408 entirely original UI-composition source/strict action safety tests."""
import importlib.util
import json
import unittest
from pathlib import Path
from h408.kaggle.h408_private_gui_composition_eval import (
    N,WIDTH,HEIGHT,LABELS,PARENTS,BASE,REV,parse,score)

class H408SourceOnlyPolicyTests(unittest.TestCase):
    def test_absent_or_ambiguous_target_never_authorizes_click(self):
        for regime in ("absent","ambiguous"):
            self.assertEqual(score("abstain",None,[],regime),(True,False))
            self.assertEqual(score("point",[500,500],[],regime),(False,True))
            self.assertEqual(score("invalid",None,[],regime),(False,False))
    def test_unique_button_requires_point_inside_original_visible_box(self):
        box=[(180,130,272,164)]
        point=[226/WIDTH*1000,147/HEIGHT*1000]
        self.assertEqual(score("point",point,box,"unique"),(True,False))
        self.assertEqual(score("point",[0,0],box,"unique"),(False,False))
        self.assertEqual(score("abstain",None,box,"unique"),(False,False))
    def test_strict_json_rejects_bad_numbers_duplicates_and_extra_keys(self):
        self.assertEqual(parse('{"point_2d":[500,500]}'),("point",[500.,500.]))
        self.assertEqual(parse('{"abstain":true}'),("abstain",None))
        for bad in ('not JSON','{"point_2d":[NaN,2]}',
                    '{"point_2d":[1001,2]}','{"point_2d":[true,2]}',
                    '{"point_2d":[1,2],"abstain":true}',
                    '{"abstain":false}'):
            self.assertEqual(parse(bad)[0],"invalid")
    def test_duplicate_json_keys_even_escaped_aliases_are_invalid(self):
        for raw in (
            '{"abstain":true,"abstain":true}',
            '{"abstain":true,"\\u0061bstain":true}',
            '{"point_2d":[100,100],"point_2d":[200,200]}',
            '{"point_2d":[100,100],"\\u0070oint_2d":[200,200]}',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse(raw)[0],"invalid")

    def test_metadata_requires_private_owner_only_prior_adapters(self):
        path=Path(__file__).resolve().parents[1]/"h408"/"kaggle"/"kernel-metadata.json"
        meta=json.loads(path.read_text(encoding="utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["dataset_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(meta["kernel_sources"],[
            "abdulazizshehri/hala-h403-private-synthetic-safe-abstention-lora",
            "abdulazizshehri/hala-h406-private-invented-ambiguity-lora"])
        self.assertEqual(set(PARENTS),{"h403","h406"})
        self.assertEqual((N,WIDTH,HEIGHT,BASE,REV),(
            36,448,336,"Qwen/Qwen3-VL-2B-Instruct",
            "89644892e4d85e24eaac8bacfd4f463576704203"))

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Pillow needed for originally authored UI mock images")
class H408FirstPartyImagesTests(unittest.TestCase):
    def test_composed_ui_groups_balanced_bilingual_and_source_distinct(self):
        from h408.kaggle.h408_private_gui_composition_eval import invent,verify_sources
        cases=[invent(i) for i in range(N)]
        self.assertEqual(len(verify_sources(cases)),64)
        for regime in ("unique","absent","ambiguous"):
            rows=[x for x in cases if x[3]==regime]
            self.assertEqual(len(rows),12)
            self.assertEqual(sum(x[4]==0 for x in rows),6)
            self.assertEqual(sum(x[4]==1 for x in rows),6)
            for im,task,boxes,kind,lang,label,gold in rows:
                self.assertEqual(im.size,(WIDTH,HEIGHT))
                self.assertIn(label,LABELS)
                self.assertIn(label,task)
                self.assertEqual(len(boxes),{"unique":1,"absent":0,
                                               "ambiguous":2}[kind])
                if kind=="unique":self.assertEqual(set(gold),{"point_2d"})
                else:self.assertEqual(gold,{"abstain":True})
    def test_composed_ui_differs_from_all_prior_colored_square_images(self):
        from hashlib import sha256
        from h408.kaggle.h408_private_gui_composition_eval import invent
        from h405.kaggle.h405_synthetic_ambiguity_eval import invent as old405
        from h406.kaggle.h406_ambiguity_refusal_lora import generate as old406
        now=[invent(i)[0] for i in range(N)]
        old=[*[old405(i)[0] for i in range(72)],
             *[old406(i,40611,"train")[0] for i in range(96)]]
        f=lambda im:sha256(im.tobytes()).digest()
        self.assertEqual(len(set(map(f,now))),N)
        self.assertFalse(set(map(f,now)) & set(map(f,old)))

if __name__=="__main__":
    unittest.main()
