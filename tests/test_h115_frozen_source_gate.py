"""H115 frozen future 64-case strict original GUI score/source contract."""
import unittest
from h115.eval import (START_INDEX,PER_DOMAIN,END_INDEX,ARMS,frozen,
                       admit_h114_checkpoint,H114_VERIFIED_ADAPTER_SHA)
from h115.compose import compose_strict_original_drag

class TestH115FrozenNewSource(unittest.TestCase):
    def test_new_original_source_60_to_75(self):
        self.assertEqual((START_INDEX,PER_DOMAIN,END_INDEX),(60,16,76))
        cases=[{"id":str(x)} for x in range(76)]
        result=frozen(cases)
        self.assertEqual(len(result),16)
        self.assertEqual(result[0]["id"],"60")
        self.assertEqual(result[-1]["id"],"75")
        with self.assertRaises(ValueError):frozen(cases[:-1])
        self.assertEqual(ARMS,("h106_original_drag","h114_two_independent_points"))
    def test_unverified_gpu_weights_never_loaded(self):
        self.assertIsNone(H114_VERIFIED_ADAPTER_SHA)
        with self.assertRaises(ValueError):
            admit_h114_checkpoint("/nonexistent/private/source")
    def test_partial_model_outputs_fail_closed(self):
        case={"image_width":1024,"image_height":640,
              "task":{"intent":"Move handle","ordered":True}}
        self.assertIsNone(compose_strict_original_drag(case,"bad","bad"))
if __name__=="__main__":unittest.main()
