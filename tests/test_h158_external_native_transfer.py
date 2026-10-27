"""H158 independent second-app outcome auditor checks (synthetic fixtures)."""
import copy
import inspect
import tempfile
import unittest

from h158.audit_neural_transfer import paired,audit
from h158.replay_neural_native import native
from h158.original_todomvc_source import COUNTS


class H158CrossApplicationPairingTests(unittest.TestCase):
    def setUp(self):
        self.cases=[]
        self.rows=[]
        index=0
        for task,groups in COUNTS.items():
            for language,n in groups.items():
                for i in range(n):
                    index+=1
                    sha=f"{index:064x}"
                    self.cases.append({"task":task,"language":language,
                                       "image_sha256":sha})
                    for role,successful in (("before",i==0),
                                            ("after",i in (0,1))):
                        self.rows.append({
                            "role":role,"task":task,"language":language,
                            "image_sha":sha,"valid":True,
                            "abstained":False,"click_sent":True,
                            "original_app_task_success":successful})

    def test_16_matching_bilingual_untrained_ood_examples(self):
        report=paired(self.rows,self.cases)
        self.assertEqual(report["paired_new_app_tasks"],16)
        self.assertEqual(report["before_actual_app_success"],6)
        self.assertEqual(report["after_actual_app_success"],12)
        self.assertEqual(report["new_wins"],6)
        self.assertEqual(report["regressions"],0)
        self.assertEqual(report["paired_sign_two_sided_p"],0.03125)
        self.assertTrue(report["original_app_has_only_memory_storage"])

    def test_reject_missing_duplicated_unexecuted_or_mismatched(self):
        for scenario in ("missing","duplicate","wrong_sha","fake_success",
                         "abstained_success"):
            with self.subTest(scenario=scenario):
                rows=copy.deepcopy(self.rows)
                if scenario=="missing":
                    rows.pop()
                elif scenario=="duplicate":
                    rows[-1]=dict(rows[0])
                elif scenario=="wrong_sha":
                    rows[-1]["image_sha"]="f"*64
                elif scenario=="fake_success":
                    rows[0]["click_sent"]=False
                else:
                    rows[0]["abstained"]=True
                with self.assertRaises(ValueError):
                    paired(rows,self.cases)

    def test_no_gold_point_used_by_native_model_executor(self):
        code=inspect.getsource(native)
        self.assertNotIn("reference_point",code)
        self.assertNotIn("native_gold_point",code)
        self.assertIn("page.mouse.click(*click)",code)

    def test_external_score_requires_true_owner_one_shot_report(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                audit(temp)


if __name__=="__main__":
    unittest.main()
