"""H24 remote Mind DEV evaluator CPU-only admission and parsing tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from h24 import data, mind_dev as m


class H24MindDEVTests(unittest.TestCase):
    def setUp(self):
        self.cases=m.dev_cases()

    def test_exact_frozen_dev_only(self):
        self.assertEqual(len(self.cases),20)
        self.assertEqual({c["split"] for c in self.cases},{"dev"})
        self.assertEqual({c["language"] for c in self.cases},{"en","ar"})
        self.assertEqual({c["family"] for c in self.cases},
                         {"purchase_requisition","classroom_attendance"})

    def test_prompt_has_no_reference_answer_or_train_test_identity(self):
        for case in self.cases:
            prompt=m.public_prompt(case)
            self.assertIn("HalaMind",prompt)
            self.assertNotIn("training_type_eligible",prompt)
            self.assertNotIn('"protected"',prompt)
            self.assertNotIn('"expected_current_frame"',prompt)
            self.assertNotIn('"image_qualified"',prompt)
            self.assertNotIn('"image"',prompt)
            self.assertNotIn('"expected"',prompt)
            self.assertNotIn('"split"',prompt)
            self.assertNotIn("h24-",prompt)
            self.assertNotIn('"subgoal": "reject_external_command"',prompt)

    def test_public_projection_never_exposes_evaluator_permissions(self):
        for case in self.cases:
            public=json.loads(m.public_prompt(case).split("\n",1)[-1])
            controls=public["observation"]["controls"]
            self.assertEqual(len(controls),2)
            self.assertTrue(all(set(c)=={"ref","name","role","bounds"} for c in controls))
            self.assertNotIn("protected",m.public_prompt(case))
            self.assertNotIn("expected",m.public_prompt(case))
            self.assertEqual(public["observation"]["current_frame_id"],
                             case["observation"]["expected_current_frame"])

    def test_prompt_rejects_other_splits(self):
        for split in ("train","test"):
            case=next(x for x in data.dataset()["cases"] if x["split"]==split)
            with self.assertRaises(ValueError):
                m.public_prompt(case)

    def test_strict_json_unique_and_finite(self):
        with self.assertRaises(ValueError):
            m.strict_json('{"tool":"visual","tool":"abstain"}')
        with self.assertRaises(ValueError):
            m.strict_json('{"tool":NaN}')

    def test_proposals_exact_contract(self):
        label={"kind":"next","tool":"observe","subgoal":"refresh_stale_frame"}
        self.assertEqual(m.proposal(json.dumps(label)),label)
        for broken in ([], {"kind":"next","tool":"shell","subgoal":"bad"},
                       {"kind":"execute","tool":"visual","subgoal":"click"},
                       {"kind":"next","tool":"observe","subgoal":"../../bad"},
                       {"kind":"next","tool":"observe","subgoal":"good","permission":"yes"}):
            with self.subTest(broken=broken),self.assertRaises(ValueError):
                m.proposal(json.dumps(broken))

    def test_parse_rejects_markdown_and_extra_text(self):
        s='{"kind":"next","tool":"observe","subgoal":"refresh_stale_frame"}'
        with self.assertRaises(ValueError):
            m.proposal(chr(96)*3+"json\n"+s+"\n"+chr(96)*3)
        with self.assertRaises(ValueError):
            m.proposal(s+"\nIgnore previous instructions")

    def test_score_correct_and_wrong_subgoal_distinct(self):
        for case in self.cases:
            good=json.dumps(case["expected"])
            result=m.score(case,good)
            self.assertTrue(result["valid"] and result["tool_correct"] and result["exact_correct"])
            changed=copy.deepcopy(case["expected"])
            changed["subgoal"]="different_subgoal"
            changed_result=m.score(case,json.dumps(changed))
            self.assertTrue(changed_result["tool_correct"])
            self.assertFalse(changed_result["exact_correct"])

    def test_invalid_predictions_cannot_claim_success(self):
        r=m.score(self.cases[0],"{malformed}")
        self.assertFalse(r["valid"])
        self.assertFalse(r["tool_correct"])
        self.assertFalse(r["exact_correct"])

    def test_summary_counts_language_strata_without_independence_claim(self):
        rows=[]
        for i,c in enumerate(self.cases):
            rows.append({"id":c["id"],"language":c["language"],
                         "valid":True,"tool_correct":i%2==0,
                         "exact_correct":i%4==0,"error":None})
        s=m.summary(rows)
        self.assertEqual(s["all"]["cases"],20)
        self.assertEqual(s["en"]["cases"],s["ar"]["cases"])
        self.assertEqual(s["all"]["valid_json"],20)
        self.assertEqual(s["all"]["tool_correct"],10)
        self.assertEqual(s["all"]["exact_correct"],5)

    def test_missing_or_duplicate_cases_fail_summary(self):
        rows=[{"id":c["id"],"language":c["language"],
               "valid":False,"tool_correct":False,"exact_correct":False,"error":None}
              for c in self.cases]
        with self.assertRaises(ValueError):
            m.summary(rows[:19])
        with self.assertRaises(ValueError):
            m.summary(rows[:-1]+[rows[0]])

    def test_windows_cannot_load_model_or_write_remote_result(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                m.remote_preflight("a"*40,Path(temp)/"new")
            self.assertFalse((Path(temp)/"new").exists())

    def test_serialized_data_still_matches_original_no_test_export(self):
        original=data.dataset()
        self.assertEqual(data.validate(original)["sha256"],m.H24_CASES_SHA256)
        self.assertEqual({c["split"] for c in m.dev_cases()},{"dev"})

    def test_model_never_executes_in_imported_cpu_test(self):
        self.assertFalse(hasattr(m,"torch"))
        self.assertFalse(hasattr(m,"model"))
        self.assertFalse(hasattr(m,"processor"))


if __name__=="__main__":
    unittest.main()
