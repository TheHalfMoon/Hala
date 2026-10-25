"""H120 deterministic 4-domain next64 selection and paired official H61 scorer."""
import hashlib
import json
import unittest
from h120.protocol import (START,END,ARMS,freeze_domain,freeze_all,
                           exact_mcnemar,score_paired)
from h61.official_eval import DOMAINS


def original_prefixes():
    result={}
    for domain in DOMAINS:
        result[domain]=[
          {"id":domain+"-"+str(i),
           "domain":domain,
           "image_sha256":hashlib.sha256((domain+"/"+str(i)).encode()).hexdigest(),
           "image_width":1024,"image_height":640,
           "task":{"start_bbox":[100,100,110,110],
                   "end_bbox":[900,500,910,510],"ordered":True}}
          for i in range(END)]
    return result


def action(good):
    return json.dumps({"type":"drag","coordinate_space":"source_image_pixels",
                       "ordered":True,"start_point":[105,105],
                       "end_point":[905,505] if good else [450,350]})

class ProtocolTest(unittest.TestCase):
    def test_next_original64_independent_of_h118(self):
        source=original_prefixes()
        new=freeze_all(source)
        self.assertEqual((START,END),(92,108))
        self.assertEqual(len(new),64)
        self.assertEqual(len({row["image_sha256"] for row in new}),64)
        for domain in DOMAINS:
            self.assertEqual([x["id"] for x in freeze_domain(source[domain])],
                [domain+"-"+str(i) for i in range(START,END)])
            prev={x["image_sha256"] for x in source[domain][76:92]}
            now={x["image_sha256"] for x in freeze_domain(source[domain])}
            self.assertFalse(prev&now)

    def test_incomplete_or_overlapping_original_eval_rejected(self):
        source=original_prefixes()
        with self.assertRaises(ValueError):
            freeze_all({name:rows[:-1] for name,rows in source.items()})
        with self.assertRaises(ValueError):
            freeze_all({**source,"slider":source["slider"][:-1]})
        altered=[dict(x) for x in source["slider"]]
        altered[92]["image_sha256"]=altered[76]["image_sha256"]
        with self.assertRaises(ValueError):
            freeze_domain(altered)
        with self.assertRaises(ValueError):
            freeze_all({**source,"slider":source["sheet"]})

    def test_exact_paired_mcnemar(self):
        self.assertEqual(exact_mcnemar([False]*64,[False]*64)
                         ["two_sided_mcnemar_exact_p"],1)
        p=exact_mcnemar([True]*6,[False]*6)
        self.assertEqual(p["candidate_only_wins"],6)
        self.assertAlmostEqual(p["two_sided_mcnemar_exact_p"],.03125)
        with self.assertRaises(ValueError):
            exact_mcnemar([1],[False])

    def test_same_real_H61_strict_scorer_for_three_arms(self):
        scenes=freeze_all(original_prefixes())
        ids=[x["id"] for x in scenes]
        observations={
          ARMS[0]:{id:action(False) for id in ids},
          ARMS[1]:{id:action(i==0) for i,id in enumerate(ids)},
          ARMS[2]:{id:action(i<6) for i,id in enumerate(ids)},
        }
        scored=score_paired(scenes,observations)
        self.assertEqual(scored["source_scenes"],64)
        self.assertEqual(scored["per_arm"][ARMS[0]]["task_success"],0)
        self.assertEqual(scored["per_arm"][ARMS[1]]["task_success"],1)
        self.assertEqual(scored["per_arm"][ARMS[2]]["task_success"],6)
        self.assertEqual(scored["h119_vs_h114"]["candidate_only_wins"],6)
        self.assertAlmostEqual(scored["h119_vs_h114"]
                               ["two_sided_mcnemar_exact_p"],.03125)
        self.assertFalse(scored["model_GUI_improved_claimed"])
        bad={**observations,ARMS[0]:{ids[0]:action(False)}}
        with self.assertRaises(ValueError):
            score_paired(scenes,bad)


if __name__=="__main__":
    unittest.main()
