"""Fail-closed V3 source-end sampling; balanced, original, and BLIND-free."""
import collections
import unittest
from h140.train_gpu_v3 import make_balanced_end_recovery_schedule, STEPS, GRAD_ACCUM

class H140EndRecoveryV3Tests(unittest.TestCase):
    def original_rows(self):
        rows=[]
        for name,n in (
            ("vanilla-native-board",86),
            ("react-native-board",86),
            ("react-dndkit-board",84),
        ):
            for i in range(n):
                rows.append({"id":f"owner-{name}-{i}",
                             "app":name,"language":"en" if i%2==0 else "ar"})
        return rows

    def test_full_start_and_oversampled_end_bilingual_roles(self):
        rows=self.original_rows()
        plan=make_balanced_end_recovery_schedule(rows)
        self.assertEqual(len(plan),640)
        self.assertEqual(STEPS*GRAD_ACCUM,640)
        role=collections.Counter(r for i,r in plan)
        self.assertEqual(role,{"start":256,"end":384})
        self.assertEqual({i for i,r in plan if r=="start"},set(range(256)))
        by_role_language=collections.Counter((r,rows[i]["language"]) for i,r in plan)
        self.assertEqual(by_role_language,{
            ("start","en"):128,("start","ar"):128,
            ("end","en"):192,("end","ar"):192
        })
        repeat=collections.Counter((rows[i]["app"],rows[i]["language"])
                                   for i,r in plan if r=="end")
        baseline=collections.Counter((row["app"],row["language"])
                                     for row in rows)
        self.assertEqual(repeat[("react-native-board","en")]
                         -baseline[("react-native-board","en")],32)
        self.assertEqual(repeat[("react-native-board","ar")]
                         -baseline[("react-native-board","ar")],32)
        self.assertEqual(plan,make_balanced_end_recovery_schedule(rows))

    def test_missing_app_or_source_quota_rejected(self):
        rows=self.original_rows()
        rows[0]["app"]="sealed-blind-original"
        with self.assertRaises(ValueError):
            make_balanced_end_recovery_schedule(rows)
        rows=self.original_rows()
        rows[0]["language"]="invalid"
        with self.assertRaises(ValueError):
            make_balanced_end_recovery_schedule(rows)

if __name__=="__main__":
    unittest.main()
