"""Image-cluster matched H426 research statistics, no model inference.

Every outcome here is fabricated; the tests prove only that paired math is
correct and cannot be presented as external OOD or newly trained Qwen accuracy.
"""
import copy
import hashlib
import json
import unittest

from h426.paired_study_statistics import (
    _mcnemar_two_sided, summarize_matched_images,
    summarize_synthetic_dev_pairs,
)
from h426.synthetic_drag_curriculum import generate


def rows(n=12):
    out=[]
    for index in range(n):
        out.append({
            "image_sha256": hashlib.sha256(
                f"h426-invented-stat-image-{index}".encode("utf8")
            ).hexdigest(),
            "domain": ("slider","slide_resize","text_highlight")[index%3],
            "source_family": "invented-ui-family-" + str(index%5),
            "languages": ["en","ar"] if index%2 else ["en","en"],
            "baseline_correct": [False, False],
            "candidate_correct": [True, True],
        })
    return out


class MatchedImageStudyTests(unittest.TestCase):
    def test_all_twelve_two_action_improvements_count_as_twelve_images(self):
        result=summarize_matched_images(rows())
        self.assertEqual(result["image_units"],12)
        self.assertEqual(result["paired_action_units"],24)
        self.assertEqual(result["baseline_correct_actions"],0)
        self.assertEqual(result["candidate_correct_actions"],24)
        self.assertEqual(result["candidate_both_correct_images"],12)
        self.assertEqual(result["candidate_only_both_correct"],12)
        self.assertEqual(result["action_accuracy_difference_candidate_minus_baseline"],1)
        self.assertAlmostEqual(result["exact_two_sided_mcnemar_p_both_correct"],
                               2/2**12,places=9)
        self.assertEqual(result["image_cluster_bootstrap_95pct_difference"],[1.,1.])
        self.assertEqual(sum(x["images"] for x in result["by_domain"].values()),12)
        self.assertFalse(result["model_predictions_verified_here"])
        self.assertFalse(result["independent_blind_external_ood_attested_here"])
        self.assertFalse(result["copyright_train_rights_verified_here"])
        self.assertFalse(result["model_or_release_qualified"])

    def test_exact_paired_discordance_two_sided(self):
        self.assertEqual(_mcnemar_two_sided(0,0),1.)
        self.assertEqual(_mcnemar_two_sided(1,0),1.)
        self.assertEqual(_mcnemar_two_sided(10,0),2/2**10)
        self.assertAlmostEqual(
            _mcnemar_two_sided(2,10),
            2*(1+12+66)/(2**12),
        )
        self.assertEqual(_mcnemar_two_sided(5,5),1.)
        with self.assertRaises(ValueError):
            _mcnemar_two_sided(-1,0)

    def test_reordering_does_not_change_cluster_bootstrap(self):
        first=rows(19)
        for i in (3,4,7,11):
            first[i]["candidate_correct"]=[False,False]
        for i in (3,4,9,14):
            first[i]["baseline_correct"]=[True,True]
        a=summarize_matched_images(first)
        b=summarize_matched_images(list(reversed(first)))
        self.assertEqual(a,b)
        self.assertEqual(a["paired_image_improved"],13)
        self.assertEqual(a["paired_image_regressed"],2)
        self.assertEqual(a["paired_image_unchanged"],4)
        self.assertLessEqual(a["image_cluster_bootstrap_95pct_difference"][0],
                             a["action_accuracy_difference_candidate_minus_baseline"])
        self.assertGreaterEqual(a["image_cluster_bootstrap_95pct_difference"][1],
                                a["action_accuracy_difference_candidate_minus_baseline"])

    def test_two_tasks_on_one_image_never_masquerade_as_two_images(self):
        sample=rows(2)
        sample[0]["baseline_correct"]=[False,True]
        sample[0]["candidate_correct"]=[True,False]
        sample[1]["baseline_correct"]=[True,True]
        sample[1]["candidate_correct"]=[False,False]
        result=summarize_matched_images(sample)
        self.assertEqual(result["image_units"],2)
        self.assertEqual(result["paired_action_units"],4)
        self.assertEqual(result["baseline_only_both_correct"],1)
        self.assertEqual(result["candidate_only_both_correct"],0)
        self.assertEqual(result["paired_image_unchanged"],1)

    def test_bounded_unique_source_images_and_strict_labels(self):
        valid=rows(2)
        bad=[
            [],valid[:1],valid*2049,
            [valid[0],valid[0]],
            [dict(valid[0],image_sha256="Z"*64),valid[1]],
            [dict(valid[0],domain="unrecognized"),valid[1]],
            [dict(valid[0],source_family=""),valid[1]],
            [dict(valid[0],languages=["en"]),valid[1]],
            [dict(valid[0],languages=["en", "fr"]),valid[1]],
            [dict(valid[0],baseline_correct=[True,1]),valid[1]],
            [dict(valid[0],candidate_correct=[False]),valid[1]],
            [dict(valid[0],unsafe_extra=True),valid[1]],
        ]
        for candidate in bad:
            with self.subTest(case=str(candidate)[:64]),self.assertRaises(ValueError):
                summarize_matched_images(candidate)

    def test_real_h425_scorer_integrates_with_image_cluster_statistics(self):
        matching = []
        for i in range(6):
            domain = ("slider", "slide_resize", "text_highlight")[i % 3]
            case = generate("dev", i, domain)
            old, new = [], []
            for row in case["actions"]:
                target = copy.deepcopy(row["target"])
                good = json.dumps(target, separators=(",", ":"))
                target["end_point"] = [0, 0]
                wrong_end = json.dumps(target, separators=(",", ":"))
                provenance = {
                    "id": row["id"],
                    "image_sha256": case["image_sha256"],
                }
                old.append({**provenance, "raw_prediction": (
                    wrong_end if row["language"] == "en" else good
                )})
                new.append({**provenance, "raw_prediction": good})
            matching.append({
                "index": i, "domain": domain,
                "baseline": old, "candidate": new,
            })
        report = summarize_synthetic_dev_pairs(matching)
        self.assertEqual(report["image_units"], 6)
        self.assertEqual(report["paired_action_units"], 12)
        self.assertEqual(report["baseline_correct_actions"], 6)
        self.assertEqual(report["candidate_correct_actions"], 12)
        self.assertEqual(report["baseline_both_correct_images"], 0)
        self.assertEqual(report["candidate_both_correct_images"], 6)
        self.assertEqual(report["paired_image_improved"], 6)
        self.assertEqual(report["source_evaluation_split"], "firstparty_invented_dev")
        self.assertEqual(report["external_ood_status"], "not_independent")
        self.assertFalse(report["model_or_release_qualified"])
        self.assertNotIn("raw_prediction", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "distinct source images"):
            summarize_synthetic_dev_pairs([matching[0],matching[0]])

    def test_aggregate_never_contains_source_ids_raw_answers_or_prompt(self):
        report=summarize_matched_images(rows(6))
        s=json.dumps(report,sort_keys=True)
        self.assertNotIn(rows(6)[0]["image_sha256"],s)
        self.assertNotIn("invented-ui-family",s)
        self.assertNotIn("raw_prediction",s)
        self.assertNotIn("start_point",s)
        self.assertIn("statistical_unit",report)
        self.assertFalse(report["model_or_release_qualified"])


if __name__=="__main__":
    unittest.main()
