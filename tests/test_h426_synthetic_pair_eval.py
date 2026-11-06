"""H426 invented DEV evaluator uses exact H425 action scorer, never a model.

All predictions here are deliberately forged from generated first-party gold.
"""
import copy
import json
import unittest
from pathlib import Path

from h426.synthetic_drag_curriculum import DOMAINS, generate
from h426.synthetic_pair_eval import score_dev_pair, compare_synthetic_dev

ROOT=Path(__file__).resolve().parents[1]


def predictions(domain, index=7, variant="correct"):
    case=generate("dev",index,domain)
    out=[]
    for row in case["actions"]:
        target=copy.deepcopy(row["target"])
        if variant=="wrong_end" and row["language"]=="en":
            point=target["end_point"]
            target["end_point"]=[0 if point[0]>500 else 1000,
                                 0 if point[1]>500 else 1000]
        elif variant=="wrong_start" and row["language"]=="en":
            point=target["start_point"]
            target["start_point"]=[0 if point[0]>500 else 1000,
                                   0 if point[1]>500 else 1000]
        elif variant=="both_wrong" and row["language"]=="en":
            for key in ("start_point","end_point"):
                point=target[key]
                target[key]=[0 if point[0]>500 else 1000,
                             0 if point[1]>500 else 1000]
        text=(json.dumps(target,separators=(",",":"))
              if variant!="invalid" or row["language"]=="ar" else "not json")
        out.append({"id":row["id"],"image_sha256":case["image_sha256"],
                    "raw_prediction":text})
    return out


class SyntheticH426PairedMetricTests(unittest.TestCase):
    def test_original_h425_evaluator_accepts_exact_generated_gold(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                proof=score_dev_pair(7,domain,predictions(domain))
                self.assertEqual(proof["actions"],2)
                self.assertEqual(proof["correct_actions"],2)
                self.assertTrue(proof["both_actions_correct"])
                self.assertEqual(set(proof["per_language"]),{"en","ar"})
                self.assertTrue(all(x["start_near"] and x["end_near"]
                                    for x in proof["per_language"].values()))
                self.assertFalse(proof["new_model_inference_executed"])
                self.assertFalse(proof["independent_external_ood"])

    def test_real_h425_scorer_detects_single_endpoint_errors(self):
        for domain in DOMAINS:
            for variant, field in (("wrong_start","start_near"),
                                   ("wrong_end","end_near"),
                                   ("both_wrong","correct")):
                with self.subTest(domain=domain,variant=variant):
                    proof=score_dev_pair(7,domain,predictions(domain,variant=variant))
                    en=proof["per_language"]["en"]
                    ar=proof["per_language"]["ar"]
                    self.assertFalse(en[field])
                    self.assertTrue(ar["correct"])
                    self.assertFalse(proof["both_actions_correct"])
                    self.assertEqual(proof["correct_actions"],1)

    def test_invalid_prediction_is_not_silently_scored_as_valid(self):
        proof=score_dev_pair(7,"slider",predictions("slider",variant="invalid"))
        self.assertEqual(proof["correct_actions"],1)
        self.assertFalse(proof["per_language"]["en"]["valid"])
        self.assertEqual(proof["per_language"]["en"]["error"],"schema")

    def test_pairwise_comparison_reuses_identical_image_without_inference(self):
        for domain in DOMAINS:
            out=compare_synthetic_dev(7,domain,
                predictions(domain,variant="wrong_end"),predictions(domain))
            self.assertEqual(out["baseline_correct"],1)
            self.assertEqual(out["candidate_correct"],2)
            self.assertFalse(out["baseline_both_correct"])
            self.assertTrue(out["candidate_both_correct"])
            self.assertEqual(len(out["source_image_sha256"]),64)
            self.assertFalse(out["trained_or_queried_model_here"])
            self.assertFalse(out["model_release_qualified"])
            self.assertTrue(out["per_language"]["en"]["new_correct"])

    def test_specific_end_only_error_separate_from_other_geometry_failures(self):
        proof=compare_synthetic_dev(7,"slide_resize",predictions("slide_resize"),
                    predictions("slide_resize",variant="wrong_end"))
        self.assertTrue(proof["per_language"]["en"]["end_only_error"])
        self.assertFalse(proof["per_language"]["en"]["start_only_error"])
        self.assertFalse(proof["per_language"]["en"]["both_points_error"])
        self.assertTrue(proof["per_language"]["ar"]["new_correct"])

    def test_wrong_sha_id_duplicate_or_missing_pair_refuses(self):
        correct=predictions("slider")
        variants=[
            [correct[0]],
            [correct[0],correct[0]],
            [dict(correct[0],image_sha256="0"*64),correct[1]],
            [dict(correct[0],id="wrong-source"),correct[1]],
            [dict(correct[0],raw_prediction=1),correct[1]],
            [dict(correct[0],raw_prediction="x"*8193),correct[1]],
            [dict(correct[0],unknown_field=1),correct[1]],
        ]
        for case in variants:
            with self.subTest(case=str(case)[:50]),self.assertRaises(ValueError):
                score_dev_pair(7,"slider",case)

    def test_generated_dev_only_and_no_raw_model_output_in_report(self):
        report=score_dev_pair(2,"text_highlight",predictions("text_highlight",2))
        self.assertEqual(report["split"],"dev")
        self.assertNotIn("raw_prediction",str(report))
        self.assertNotIn("start_point",str(report))
        self.assertFalse(report["model_release_qualified"])
        with self.assertRaises(ValueError):
            score_dev_pair(-1,"slider",predictions("slider"))

    def test_only_consumes_supplied_predictions_and_has_no_gpu_calls(self):
        source=(ROOT/"h426/synthetic_pair_eval.py").read_text("utf8")
        for forbidden in ("model.generate(", "optimizer.step(", "cuda:0",
                          "kaggle kernels push", "model.save_pretrained("):
            self.assertNotIn(forbidden,source)
        self.assertIn("h425.score_predicted(",source)
        self.assertIn('generate("dev",',source)


if __name__=="__main__":
    unittest.main()
