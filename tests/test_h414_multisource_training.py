"""H414 source-first preflight: real H50/H52 originals, no fake row inflation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

from h414.kaggle import h414_multisource_lora as h

ROOT=Path(__file__).resolve().parents[1]


class MultiSourcePreflight(unittest.TestCase):
    def test_frozen_thousands_and_real_optimizer_microbatches(self):
        self.assertEqual(h.TRAIN_NEW,4096)
        self.assertEqual(h.TRAIN_ORIGINAL,672)
        self.assertEqual(h.STEPS,1192)
        self.assertEqual(h.STEPS*h.MICROBATCHES,4768)
        self.assertEqual(h.PARENT_SHA,
           "fa47ff52f67dd39b7aef205351e71551716b66da55cda3b4370c981189c13066")

    def test_balanced_original_synthetic_indexing(self):
        for split,count in (("train",4096),("dev",96),("audit",96)):
            found=Counter()
            for i in range(count):
                fam,kind,lang,_=h.classify(i,split)
                found[(fam,kind,lang)]+=1
            for fam in h.FAMILIES:
                for kind in h.KINDS:
                    for lang in (0,1):
                        expected=({"unique":192,"absent":96,"duplicate":224}[kind]
                                  if split=="train" else 4)
                        self.assertEqual(found[(fam,kind,lang)],expected)

    def test_exact_private_original_kaggle_inputs(self):
        md=json.loads((ROOT/"h414/kaggle/kernel-metadata.json").read_text())
        self.assertTrue(md["is_private"])
        self.assertEqual(md["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(md["dataset_sources"],
                ["abdulazizshehri/hala-gui-training-library-private"])
        self.assertEqual(md["kernel_sources"],
                ["abdulazizshehri/hala-h412-private-duplicate-curriculum-lora"])
        for name,(expected,count) in h.ORIGINAL_DATA.items():
            file=ROOT/name/"assets/authoring_sft.jsonl"
            self.assertEqual(sha256(file.read_bytes()).hexdigest(),expected)
            self.assertEqual(len(file.read_text(encoding="utf8").splitlines()),count)

    def test_original_source_loader_uses_actual_png_bytes_and_annotations(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow optional")
        pairs={name:(ROOT/name/"assets/authoring_sft.jsonl",count)
               for name,(_,count) in h.ORIGINAL_DATA.items()}
        rows,counts=h.read_historical_sources(pairs)
        self.assertEqual(len(rows),672)
        self.assertEqual(counts["h50"]["rows"],384)
        self.assertEqual(counts["h52"]["rows"],288)
        self.assertTrue(all(json.loads(r[2]).get("point_2d") for r in rows))
        self.assertTrue(all(r[4]=="unique" for r in rows))

    def test_new_image_types_strict_refusal_and_no_training_test_reuse(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional")
        cases=[h.compose(i,"train") for i in (0,383,384,575,576,1023,1024,2048,3072,4095)]
        self.assertEqual(len({sha256(c[0].tobytes()).hexdigest() for c in cases}),len(cases))
        for im,prompt,gold,family,kind,lang,boxes in cases:
            self.assertIn(family,h.FAMILIES)
            self.assertEqual(len(boxes),{"unique":1,"absent":0,"duplicate":2}[kind])
            action,point=h.parse(gold)
            self.assertEqual(action,"point" if kind=="unique" else "abstain")
        self.assertEqual(h.parse('{"abstain":1}')[0],"invalid")
        self.assertEqual(h.parse('{"point_2d":[NaN,2]}')[0],"invalid")
        self.assertEqual(h.parse('{"point_2d":[1001,2]}')[0],"invalid")

if __name__=="__main__":
    unittest.main()
