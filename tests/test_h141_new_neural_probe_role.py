"""Preregister independent screenshot-only heading/source neural probe role."""
import unittest
from h141.visible_goal_source import collect,SCHEMA,EVAL_SCHEMA

class H141FreshVisuallyDefinedModelProbeTests(unittest.TestCase):
    def test_neural_probe_new_ids_and_exact_evaluation_count_required(self):
        self.assertNotEqual(SCHEMA,EVAL_SCHEMA)
        self.assertIn("neural-probe",EVAL_SCHEMA)
        for count,offset in ((16,44000),(8,48000),(16,48001)):
            with self.assertRaises(ValueError):
                collect("no-original", "no-private-folder",count=count,
                        attempt_offset=offset,phase="neural_probe")
        with self.assertRaises(ValueError):
            collect("no-original","no-private-folder",phase="train")
if __name__=="__main__":
    unittest.main()
