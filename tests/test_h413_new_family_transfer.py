"""H413 family-shift research only; local source and parser checks without GPU."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

from h413.kaggle import h413_new_family_transfer_eval as e


class NewFamilyTransferTests(unittest.TestCase):
    def test_strict_normalized_actions_and_refusal(self):
        self.assertEqual(e.parse('{"abstain":true}'),("abstain",None))
        for raw in (
            '{"abstain":1}','{"point_2d":[true,2]}','{"point_2d":[NaN,2]}',
            '{"point_2d":[-1,2]}','{"point_2d":[1001,2]}',
            '{"point_2d":[2,3],"abstain":true}',
            '{"abstain":true,"abstain":true}',"oops"
        ):
            self.assertEqual(e.parse(raw)[0],"invalid")
        self.assertEqual(e.parse('{"point_2d":[1000,1000]}')[0],"point")

    def test_private_readonly_gpu_and_two_actual_parent_slugs(self):
        root=Path(__file__).resolve().parents[1]
        md=json.loads((root/"h413/kaggle/kernel-metadata.json").read_text())
        self.assertTrue(md["is_private"])
        self.assertTrue(md["enable_gpu"])
        self.assertEqual(md["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(md["dataset_sources"],[])
        self.assertEqual(md["kernel_sources"],[
            "abdulazizshehri/hala-h409-private-balanced-original-ui-lora",
            "abdulazizshehri/hala-h412-private-duplicate-curriculum-lora",
        ])
        self.assertEqual(md["model_sources"],[])

    def test_genuine_parent_adapter_lineage(self):
        root=Path(__file__).resolve().parents[1]
        for version,expected in (("409",e.H409_SHA),("412",e.H412_SHA)):
            parent=(root/f"h{version}"/"verified_receipt")
            docs=list(parent.glob("*/receipt.json"))
            self.assertEqual(len(docs),1)
            data=json.loads(docs[0].read_bytes())
            self.assertEqual(data["adapter_sha256"],expected)
            self.assertIs(data["weights_public"],False)

    def test_new_first_party_source_shape_balance_and_nonreuse(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional for no-site CI")
        from h409.kaggle import h409_balanced_two_family_lora as h409
        from h412.kaggle import h412_duplicate_curriculum_lora as h412
        rows=[e.compose(i) for i in range(e.N)]
        digest=e.source_contract(rows)
        self.assertEqual(len(digest),64)
        counts=Counter((r["family"],r["kind"],r["language"]) for r in rows)
        for f in e.FAMILIES:
            for k in e.KINDS:
                for lang in ("en","ar"):
                    self.assertEqual(counts[(f,k,lang)],8)
        prior=set()
        for generator,seeds,counts in (
            (h409,{"train":40911,"dev":40921,"audit":40931},
                  {"train":128,"dev":48,"audit":48}),
            (h412,{"train":41211,"dev":41221,"audit":41231},
                  {"train":192,"dev":48,"audit":48}),
        ):
            for split,seed in seeds.items():
                for i in range(counts[split]):
                    img=generator.compose(i,seed,split)[0]
                    prior.add(sha256(img.tobytes()).hexdigest())
        self.assertFalse({r["image_sha256"] for r in rows}&prior)
        for r in rows:
            if r["kind"]=="unique":
                x0,y0,x1,y1=r["boxes"][0]
                width,height=r["image"].size
                raw=json.dumps({"point_2d":[(x0+x1)/2/width*1000,
                                            (y0+y1)/2/height*1000]})
                self.assertTrue(e.score(r,raw)["correct"])
                self.assertFalse(e.score(r,'{"abstain":true}')["correct"])
            else:
                self.assertTrue(e.score(r,'{"abstain":true}')["correct"])
                self.assertTrue(e.score(r,'{"point_2d":[500,500]}')["unsafe_click"])
        self.assertTrue(all(r["family"] in ("form_rows","schedule_tiles") for r in rows))


if __name__=="__main__":
    unittest.main()
