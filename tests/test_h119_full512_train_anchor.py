"""H119 source-only anchor robust curriculum contracts; no GPU or EVAL labels."""
import hashlib
import json
import unittest
from h114.schedule import QualifiedSchedule, ORIGINAL_H113_DERIVED_SHA
from h119.curriculum import PAIRS, SCALES, derive_schedule, shift_anchor
from h61.official_eval import strict_json


def source_schedule():
    rows=[]
    for i in range(PAIRS):
        origin="synthetic-TRAIN-only-%03d"%i
        sha=hashlib.sha256(origin.encode()).hexdigest()
        intent="Drag this original training slider to the requested endpoint."
        shared=dict(original_source_id=origin, image_sha256=sha,
                    image_path="train/"+origin+".png", language="ar" if i%4==0 else "en",
                    family=("slider","sheet","slide_resize","text_highlight","other")[i%5],
                    split="train", original_source_split="train",
                    source_bound_validation_passed=True, no_eval_or_test_source=True,
                    source_image_width=1024, source_image_height=640)
        st=[200+i%150,220+i%100]
        en=[700+i%70,650+i%70]
        rows.append(dict(shared,id=origin+"::start",endpoint_role="start",
                         prompt=intent+"\nLocate ONLY the START endpoint of the requested drag action. Return native JSON.",
                         completion=json.dumps({"role":"start","point":st,"coordinate_space":"normalized_0_1000"})))
        rows.append(dict(shared,id=origin+"::end",endpoint_role="end",
                         prompt=intent+"\nLocate ONLY the END endpoint of the requested drag action. Return native JSON.",
                         completion=json.dumps({"role":"end","point":en,"coordinate_space":"normalized_0_1000"})))
    return QualifiedSchedule(tuple(rows),{
        "source_split":"TRAIN_ONLY",
        "original_h113_point_jsonl_sha256":ORIGINAL_H113_DERIVED_SHA,
        "pilot_selected_original_screenshots":PAIRS,
        "selected_dual_point_optimizer_steps":2*PAIRS})


class H119SourceOnlyTest(unittest.TestCase):
    def test_full_512_genuine_pair_geometry_and_digest_determinism(self):
        parent=source_schedule()
        first=derive_schedule(parent)
        second=derive_schedule(parent)
        self.assertEqual(first.summary["derived_private_train_sha256"],
                         second.summary["derived_private_train_sha256"])
        self.assertEqual(len(first.selected),1024)
        self.assertEqual(first.summary["original_source_images"],512)
        self.assertEqual(sum(first.summary["anchor_magnitudes"].values()),512)
        self.assertGreater(first.summary["anchor_magnitudes"]["0"],0)
        self.assertGreater(sum(first.summary["anchor_magnitudes"][str(v)] for v in SCALES[1:]),0)
        self.assertTrue(first.summary["no_eval_or_test_data"])
        for i in range(PAIRS):
            old_start=parent.selected[i*2]
            old_end=parent.selected[i*2+1]
            new_start=first.selected[i*2]
            new_end=first.selected[i*2+1]
            self.assertEqual(old_start,new_start)
            self.assertEqual(old_end["image_sha256"],new_end["image_sha256"])
            self.assertNotIn("bbox",new_end["prompt"])
            actual_original_end=strict_json(old_end["completion"])["point"]
            anchor,_=shift_anchor(strict_json(old_start["completion"])["point"],old_start["image_sha256"])
            signed=strict_json(new_end["completion"])["delta"]
            self.assertEqual([x+y for x,y in zip(anchor,signed)],actual_original_end)
        self.assertEqual(parent.selected[0]["endpoint_role"],"start")

    def test_fail_closed_eval_source_or_missing_images(self):
        parent=source_schedule()
        corrupt=list(parent.selected)
        corrupt[0]=dict(corrupt[0],split="eval")
        with self.assertRaises(ValueError):
            derive_schedule(QualifiedSchedule(tuple(corrupt),parent.summary))
        with self.assertRaises(ValueError):
            derive_schedule(QualifiedSchedule(parent.selected[:-2],parent.summary))
        with self.assertRaises(ValueError):
            derive_schedule(QualifiedSchedule(parent.selected,
                dict(parent.summary,original_h113_point_jsonl_sha256="bad")))

    def test_anchor_only_from_original_train_point_and_digest(self):
        original=[430,570]
        sha=hashlib.sha256(b"TRAIN").hexdigest()
        self.assertEqual(shift_anchor(original,sha),shift_anchor(original,sha))
        self.assertEqual(original,[430,570])
        with self.assertRaises(ValueError):
            shift_anchor([1100,5],sha)
        with self.assertRaises(ValueError):
            shift_anchor(original,"x"*64)


if __name__=="__main__":
    unittest.main()
