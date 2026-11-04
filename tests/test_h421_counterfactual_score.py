"""Original same-image TWO instruction native DRAG scorer tests."""
import json
import unittest

from h420 import counterfactual_score as c


def native(end):
    return {"type":"drag","start_point":[100,200],
            "end_point":[end,200],"ordered":True,
            "coordinate_space":"normalized_1000"}


def rows_and_predictions():
    rows=[];pred=[]
    domains=("slider","slide_resize","text_highlight")
    for i in range(192):
        domain=domains[i//64]
        image=f"true-original-jpeg-sha-{i:03d}"
        for j,end in ((0,200),(1,800)):
            rid=f"original-source-{i:03d}-task-{j}"
            row={"pair_id":i,"alternative_index":j,
                 "id":rid,"image_sha256":image,
                 "source_domain":domain,"intent":f"Drag task {i} to {end}",
                 "native_drag_target":native(end),
                 "same_image_pair_native_target_ratio":0.36}
            rows.append(row)
            pred.append({"id":rid,"image_sha256":image,
                         "raw_prediction":json.dumps(native(end))})
    return rows,pred


class H421PairMetricTests(unittest.TestCase):
    def test_all_both_correct_when_model_follows_each_original_prompt(self):
        cases,outputs=rows_and_predictions()
        stats=c.calculate(cases,outputs)
        self.assertEqual(stats["image_pairs"],192)
        self.assertEqual(stats["native_instructions"],384)
        self.assertEqual(stats["both_actions_correct"],192)
        self.assertEqual(stats["single_action_correct"],384)
        self.assertEqual(stats["one_action_correct"],0)
        self.assertEqual(stats["wrong_instruction_source_target"],0)
        self.assertEqual(stats["collapsed_same_raw_action"],0)
        self.assertEqual(stats["by_domain"]["slider"]["both"],64)
        self.assertFalse(stats["independent_external_ood"])
        self.assertFalse(stats["model_release_qualified"])

    def test_constant_one_action_collapses_and_fails_pair_metric(self):
        cases,outputs=rows_and_predictions()
        for item in outputs:
            item["raw_prediction"]=json.dumps(native(200))
        stats=c.calculate(cases,outputs)
        self.assertEqual(stats["single_action_correct"],192)
        self.assertEqual(stats["both_actions_correct"],0)
        self.assertEqual(stats["one_action_correct"],192)
        self.assertEqual(stats["collapsed_same_raw_action"],192)
        self.assertEqual(stats["wrong_instruction_source_target"],192)

    def test_same_source_image_reused_between_test_pairs_rejected(self):
        cases,outputs=rows_and_predictions()
        cases[2]["image_sha256"]=cases[0]["image_sha256"]
        cases[3]["image_sha256"]=cases[1]["image_sha256"]
        with self.assertRaisesRegex(ValueError,"repeated original screenshot"):
            c.calculate(cases,outputs)

    def test_misaligned_prediction_image_sha_rejected(self):
        cases,outputs=rows_and_predictions()
        outputs[1]["image_sha256"]="other-original-sha"
        with self.assertRaisesRegex(ValueError,"source-image pairing drift"):
            c.calculate(cases,outputs)

    def test_identical_target_for_two_source_instructions_rejected(self):
        cases,outputs=rows_and_predictions()
        cases[1]["native_drag_target"]=cases[0]["native_drag_target"]
        with self.assertRaisesRegex(ValueError,"not instruction-contrastive"):
            c.calculate(cases,outputs)

    def test_missing_model_action_or_duplicate_output_rejected(self):
        cases,outputs=rows_and_predictions()
        with self.assertRaises(ValueError):
            c.calculate(cases,outputs[:-1])
        outputs[1]["id"]=outputs[0]["id"]
        with self.assertRaisesRegex(ValueError,"repeated"):
            c.calculate(cases,outputs)

if __name__=="__main__":
    unittest.main()
