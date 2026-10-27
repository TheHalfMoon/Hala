"""H157 independent paired-native browser success audit guard tests."""
import copy
import tempfile
import unittest

from h157.audit_native_model_gain import compare,audit
from h157.collect_spatial_source import COUNTS


class H157RealModelNativeGate(unittest.TestCase):
    def setUp(self):
        self.cases=[]
        self.episodes=[]
        index=0
        for task,quota in COUNTS["DEV"].items():
            for i in range(quota):
                index+=1
                sha=f"{index:064x}"
                self.cases.append({"task":task,"image_sha256":sha})
                # Synthetic ONLY for testing the auditor, not model scores.
                for role,passed in (("before",i==0),
                                    ("after",i in (0,1,2))):
                    self.episodes.append({
                        "role":role,"task":task,"image_sha":sha,
                        "parse_valid":True,"abstained":False,
                        "model_actions_executed":1,
                        "native_success":passed,
                        "native_reward_positive":passed})

    def test_72_identical_before_after_native_cases(self):
        data=compare(self.episodes,self.cases)
        self.assertEqual(data["paired_model_dev_tasks"],72)
        self.assertEqual(data["baseline_real_native_success"],5)
        self.assertEqual(data["trained_real_native_success"],15)
        self.assertEqual(data["new_wins"],10)
        self.assertEqual(data["regressions"],0)
        self.assertEqual(data["two_sided_exact_p"],0.001953125)
        self.assertTrue(data["one_app_dev_only_not_crossapp_blind"])

    def test_reject_missing_duplicate_wrong_native_rewards(self):
        for scenario in ("missing","duplicate","outside","fake_success",
                         "abstain_success","zero_actions_success"):
            with self.subTest(scenario=scenario):
                rows=copy.deepcopy(self.episodes)
                if scenario=="missing":
                    rows.pop()
                elif scenario=="duplicate":
                    rows[-1]=dict(rows[0])
                elif scenario=="outside":
                    rows[-1]["image_sha"]="f"*64
                elif scenario=="fake_success":
                    rows[0]["native_reward_positive"]=not rows[0]["native_success"]
                elif scenario=="abstain_success":
                    rows[0]["abstained"]=True
                else:
                    rows[0]["model_actions_executed"]=0
                with self.assertRaises(ValueError):
                    compare(rows,self.cases)

    def test_no_model_accuracy_claim_without_real_owner_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises((FileNotFoundError,ValueError)):
                audit(temp,temp,temp)


if __name__=="__main__":
    unittest.main()
