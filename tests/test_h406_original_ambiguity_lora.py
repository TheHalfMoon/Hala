"""H406 private first-party anti-ambiguity LoRA source and output guards."""
import importlib.util
import json
import unittest
from pathlib import Path
from h406.kaggle.h406_ambiguity_refusal_lora import (
    BASE,REV,STEPS,TRAIN_N,DEV_N,AUDIT_N,safe_parse)

class H406StaticSafetyTests(unittest.TestCase):
    def test_only_exact_abstain_or_bounded_point_accepted(self):
        self.assertEqual(safe_parse('{"abstain":true}'),("abstain",None))
        self.assertEqual(safe_parse('{"point_2d":[100,999]}'),
                         ("point",[100.,999.]))
        for text in ('{"abstain":false}','{"point_2d":[-1,500]}',
                     '{"point_2d":[NaN,0]}','{"point_2d":["2",3]}',
                     '{"point_2d":[1,2],"extra":true}','not-json'):
            self.assertEqual(safe_parse(text)[0],"invalid")
    def test_private_lora_metadata_and_frozen_compute_contract(self):
        meta=json.loads((Path(__file__).resolve().parents[1]/
            "h406"/"kaggle"/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["dataset_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(meta["kernel_sources"],
            ["abdulazizshehri/hala-h403-private-synthetic-safe-abstention-lora"])
        self.assertEqual((BASE,REV,STEPS,TRAIN_N,DEV_N,AUDIT_N),
            ("Qwen/Qwen3-VL-2B-Instruct",
             "89644892e4d85e24eaac8bacfd4f463576704203",
             96,96,48,48))

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                     "Optional Pillow required to draw invented images")
class H406FirstPartyInputTests(unittest.TestCase):
    def test_research_splits_are_first_party_bilingual_cardinality_checked(self):
        from h406.kaggle.h406_ambiguity_refusal_lora import (
            generate,validate_sources)
        train=[generate(i,40611,"train") for i in range(TRAIN_N)]
        dev=[generate(i,40621,"dev") for i in range(DEV_N)]
        audit=[generate(i,40631,"audit") for i in range(AUDIT_N)]
        hashes=validate_sources(train,dev,audit)
        self.assertEqual(set(hashes),{"train","dev","audit_author_visible"})
        for rows in (train,dev,audit):
            for x in rows:
                self.assertEqual(x[0].size,(224,224))
                self.assertTrue('{"abstain":true}' in x[1])
                if x[3]=="unique":
                    self.assertIn("point_2d",json.loads(x[2]))
                else:
                    self.assertEqual(json.loads(x[2]),{"abstain":True})
    def test_no_reuse_of_h405_exposed_evaluation_pixel_images(self):
        from hashlib import sha256
        from h406.kaggle.h406_ambiguity_refusal_lora import generate
        from h405.kaggle.h405_synthetic_ambiguity_eval import invent
        tested=[invent(i) for i in range(72)]
        datasets=[
            *[generate(i,40611,"train") for i in range(TRAIN_N)],
            *[generate(i,40621,"dev") for i in range(DEV_N)],
            *[generate(i,40631,"audit") for i in range(AUDIT_N)]]
        imgsha=lambda x:sha256(x[0].tobytes()).hexdigest()
        textsha=lambda x:sha256(
            x[0].tobytes()+x[1].encode("utf8")).hexdigest()
        self.assertFalse(set(map(imgsha,tested)) & set(map(imgsha,datasets)))
        self.assertFalse(set(map(textsha,tested)) & set(map(textsha,datasets)))

if __name__=="__main__":
    unittest.main()
