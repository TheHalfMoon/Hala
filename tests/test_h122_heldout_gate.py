"""H122 preregistered heldout-TRAIN END threshold and neural SHA evidence."""
import hashlib
import unittest
from h122.grid import end_completion
from h122.heldout_gate import score_heldout,HOLDOUT,FROZEN_SPLIT_SHA
from h122.schedule import Schedule

def fixtures(correct=12,repeat=False):
    families=("slider","sheet","slide_resize","text_highlight","firstparty")
    scenes=[]
    attempts=[]
    for i in range(HOLDOUT):
        true_x,true_y=50+(11*i)%700,50+(7*i)%400
        predicted=[true_x,true_y] if i<correct else [true_x+50,true_y+10]
        if repeat:
            predicted=[5,5]
        prompt="Locate destination of original drag on unchanged screenshot."
        response=end_completion(predicted)
        scene={"id":f"train-scene-{i}","split":"train",
               "source_split":"TRAIN_ONLY_HELDOUT",
               "family":families[i%len(families)],
               "image_sha256":hashlib.sha256(f"image:{i}".encode()).hexdigest(),
               "prompt":prompt,"image_width":1000,"image_height":600,
               "end_bbox":[true_x-2,true_y-2,true_x+2,true_y+2]}
        evidence={"id":scene["id"],"image_sha256":scene["image_sha256"],
                  "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                  "raw_output":response,
                  "output_sha256":hashlib.sha256(response.encode()).hexdigest()}
        scenes.append(scene)
        attempts.append(evidence)
    heldout=tuple({"original_source_id":c["id"],
                   "image_sha256":c["image_sha256"],
                   "prompt":c["prompt"],"split":"train",
                   "original_source_split":"train"} for c in scenes)
    schedule=Schedule(
        train=tuple({"id":f"training-only-{i}"} for i in range(448)),
        heldout=heldout,
        train_original_indices=tuple(range(448)),
        heldout_original_indices=tuple(range(448,512)),
        summary={"schedule_sha256":FROZEN_SPLIT_SHA,
                 "source_split":"TRAIN_ONLY"})
    return scenes,attempts,schedule

class H122HeldoutGateTest(unittest.TestCase):
    def test_original_train_64_preregistered_gate(self):
        cases,raw,schedule=fixtures()
        result=score_heldout(cases,raw,schedule)
        self.assertTrue(result["holdout_gate_passed"])
        self.assertEqual(result["strict_END_hits"],12)
        self.assertEqual(result["strict_valid_END_grid_and_point"],64)
        self.assertEqual(result["unique_valid_END_points"],64)
        self.assertEqual(len(result["original_train_families"]),5)
        self.assertFalse(result["GUI_skill_improvement_claimed"])

    def test_no_improvement_for_repeated_constant_output(self):
        cases,raw,schedule=fixtures(repeat=True)
        result=score_heldout(cases,raw,schedule)
        self.assertFalse(result["holdout_gate_passed"])
        self.assertEqual(result["unique_valid_END_points"],1)
        self.assertEqual(result["strict_END_hits"],0)
        self.assertEqual(result["modal_point_fraction"],1.0)

    def test_raw_neural_hash_cannot_be_faked(self):
        cases,raw,schedule=fixtures()
        changed=[dict(x) for x in raw]
        changed[0]["raw_output"]="tampered"
        with self.assertRaises(ValueError):
            score_heldout(cases,changed,schedule)

    def test_no_eval_or_changed_split(self):
        cases,raw,schedule=fixtures()
        cases[0]["split"]="eval"
        with self.assertRaises(ValueError):
            score_heldout(cases,raw,schedule)
        cases,raw,schedule=fixtures()
        with self.assertRaises(ValueError):
            score_heldout(cases,raw,schedule,split_sha="0"*64)

    def test_partial_predictions_never_silently_drop(self):
        cases,raw,schedule=fixtures()
        with self.assertRaises(ValueError):
            score_heldout(cases,raw[:-1],schedule)
        raw[0]["id"]="incorrect"
        with self.assertRaises(ValueError):
            score_heldout(cases,raw,schedule)

if __name__=="__main__":
    unittest.main()
