"""Exact paired statistics on eight frozen GUI scenes."""
import unittest
from h133.paired_stats import compare,exact_discordant_p

class H133PairedStatsTests(unittest.TestCase):
    def test_exact_two_sided_probability(self):
        self.assertEqual(exact_discordant_p(0,0),1.0)
        self.assertEqual(exact_discordant_p(8,0),2/256)
        self.assertEqual(exact_discordant_p(3,1),0.625)
        self.assertEqual(exact_discordant_p(1,1),1.0)
        with self.assertRaises(ValueError):exact_discordant_p(-1,2)

    def test_paired_scene_join_never_drops_cases(self):
        cases=[f"case-{i}" for i in range(8)]
        models={
          "specialist":{"per_case":[{"case":x,"actual_gui_task_success":i<3,
                                     "start_hits_handle":i<5} for i,x in enumerate(cases)]},
          "showui":{"per_case":[{"case":x,"actual_gui_task_success":i==2,
                                 "start_hits_handle":i==4} for i,x in enumerate(cases)]}
        }
        result=compare(models,"actual_gui_task_success")
        self.assertEqual(result["specialist_only"],2)
        self.assertEqual(result["both"],1)
        self.assertEqual(result["paired_sample_n"],8)
        self.assertFalse(result["generalized_superiority_proven"])
        models["showui"]["per_case"][0]["case"]="wrong"
        with self.assertRaises(ValueError):compare(models,"actual_gui_task_success")
if __name__=="__main__":unittest.main()
