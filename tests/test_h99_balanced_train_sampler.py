"""H99 CPU: sampler diversity, data integrity, deterministic identifier binding."""
import hashlib
import json
import random
import unittest
from copy import deepcopy

from h99.sampler import STRATEGY, build_balanced_plan

def fake(index, lang=None, family=None, dx=None, dy=None, *, image=None):
    lang = lang or ("ar" if index % 7 == 0 else "en")
    family = family or ("h61_f" if index % 3 == 0 else "h60_f")
    dx = dx if dx is not None else ((index % 8)-3)*40 + 1
    dy = dy if dy is not None else ((index % 5)-2)*60 + 1
    x1,y1=500,310
    completion={"type":"drag","start_point":[x1,y1],
                "end_point":[x1+dx,y1+dy],
                "coordinate_space":"source_image_pixels","ordered":True}
    return {"id":f"train-{index:05d}","split":"train","language":lang,
            "family":family,"capability":"drag",
            "image_sha256": image or hashlib.sha256(f"image-{index}".encode()).hexdigest(),
            "completion":json.dumps(completion,sort_keys=True)}

class H99BalancedSamplerTests(unittest.TestCase):
    def test_balances_real_language_family_groups(self):
        records=[fake(i) for i in range(320)]
        original=deepcopy(records)
        plan=build_balanced_plan(records,64)
        self.assertEqual(len(plan.indices),64)
        self.assertEqual(len(set(plan.indices)),64)
        self.assertEqual(records,original)
        self.assertEqual(plan.summary["unique_source_images_selected"],64)
        self.assertEqual(set(plan.summary["selected_languages"]),{"ar","en"})
        self.assertEqual(set(plan.summary["selected_families"]),{"h60_f","h61_f"})
        self.assertEqual(plan.summary["schema"],STRATEGY)
        self.assertEqual(plan.summary["coverage_fraction"],64/320)
        self.assertTrue(all(i < len(records) for i in plan.indices))
        self.assertGreaterEqual(len(plan.summary["selected_geometry"]),6)
    def test_reproducible_across_original_index_permutation(self):
        records=[fake(i) for i in range(320)]
        plan=build_balanced_plan(records,32)
        reordered=records[:]
        random.Random(7).shuffle(reordered)
        other=build_balanced_plan(reordered,32)
        self.assertEqual(plan.sha256,other.sha256)
        self.assertEqual([records[i]["id"] for i in plan.indices],
                         [reordered[i]["id"] for i in other.indices])
        self.assertNotEqual(plan.indices,other.indices)
    def test_bounded_seed_changes_plan(self):
        records=[fake(i) for i in range(320)]
        self.assertNotEqual(
            build_balanced_plan(records,32,seed="a").sha256,
            build_balanced_plan(records,32,seed="b").sha256)
    def test_skip_source_degenerate_supervision(self):
        records=[fake(i) for i in range(100)]
        records[4]=fake(4,dx=0,dy=0)
        plan=build_balanced_plan(records,32)
        self.assertNotIn(4,plan.indices)
        self.assertEqual(plan.summary["source_degenerate_targets_skipped"],1)
    def test_repeated_screenshot_not_selected(self):
        records=[fake(i,image=hashlib.sha256(f"image-{i//2}".encode()).hexdigest())
                 for i in range(250)]
        plan=build_balanced_plan(records,60)
        selected=[records[i]["image_sha256"] for i in plan.indices]
        self.assertEqual(len(set(selected)),60)
    def test_reject_eval_source(self):
        rows=[fake(i) for i in range(20)]
        rows[0]["split"]="eval"
        with self.assertRaises(ValueError):
            build_balanced_plan(rows,8)
    def test_reject_unbound_completion(self):
        rows=[fake(i) for i in range(20)]
        rows[0]["completion"]=json.dumps({"type":"click","start_point":[2,3]})
        with self.assertRaises(ValueError):
            build_balanced_plan(rows,8)
    def test_reject_duplicate_record_identity(self):
        rows=[fake(i) for i in range(30)]
        rows[1]["id"]=rows[0]["id"]
        with self.assertRaises(ValueError):
            build_balanced_plan(rows,12)
    def test_reject_insufficient_nondegenerate_source(self):
        rows=[fake(i,dx=0,dy=0) for i in range(40)]
        with self.assertRaises(ValueError):
            build_balanced_plan(rows,10)
    def test_reject_too_many_strata_for_prefix(self):
        rows=[fake(i,lang="ar" if i%2==0 else "en",family=f"family_{i}")
              for i in range(15)]
        with self.assertRaises(ValueError):
            build_balanced_plan(rows,8)
    def test_bounded_schedule_and_private_digest(self):
        rows=[fake(i) for i in range(180)]
        for n in (0,129,True,200):
            with self.subTest(steps=n),self.assertRaises(ValueError):
                build_balanced_plan(rows,n)
        p=build_balanced_plan(rows,64)
        self.assertEqual(len(p.sha256),64)
        self.assertNotIn(rows[0]["id"],str(p.summary))
        self.assertTrue(p.summary["private_raw_ids_in_summary"] is False)

if __name__=="__main__":
    unittest.main()
