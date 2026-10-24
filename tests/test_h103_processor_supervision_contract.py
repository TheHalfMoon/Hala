"""H103 pure CPU preflight invariants for actual Qwen processor probes."""
import unittest
from h103.processor_audit import grid_matches_source,validate_supervision
from h75.processor_audit import select_processor_audit_rows

class H103MaskContract(unittest.TestCase):
    def setUp(self):
        self.args=([1,2,3,4,5,6],[1,2],[-100,-100,3,4,5,6],
                   [0,0,1.0,8.0,8.0,1.0],[False,True,True,False])
    def test_pinned_expected_grid_match(self):
        self.assertTrue(grid_matches_source(1024,640,16,[1,40,64]))
    def test_processor_implicit_resize_detected(self):
        self.assertFalse(grid_matches_source(1024,640,16,[1,39,64]))
        self.assertFalse(grid_matches_source(1024,640,16,[1,40,63]))
    def test_malformed_nonpixel_grids_denied(self):
        for grid in ([0,40,64],[1,False,64],[1,40,0],[2,40,64]):
            if grid==[2,40,64]:
                self.assertFalse(grid_matches_source(1024,640,16,grid))
            else:
                with self.assertRaises(ValueError):
                    grid_matches_source(1024,640,16,grid)
    def test_supervised_coords_not_instruction(self):
        result=validate_supervision(*self.args)
        self.assertEqual(result["coordinate_tokens"],2)
        self.assertEqual(result["prompt_masked_tokens"],2)
    def test_prompt_leakage_denied(self):
        full,prefix,labels,weights,flags=self.args
        with self.assertRaises(ValueError):
            validate_supervision(full,prefix,[-100,2]+labels[2:],weights,flags)
        with self.assertRaises(ValueError):
            validate_supervision(full,prefix,labels,[1,0]+weights[2:],flags)
    def test_coordinate_weight_drift_denied(self):
        full,prefix,labels,weights,flags=self.args
        with self.assertRaises(ValueError):
            validate_supervision(full,prefix,labels,[0,0,1,1,1,1],flags)
        with self.assertRaises(ValueError):
            validate_supervision(full,prefix,labels,weights,[False]*4)
    def test_prompt_and_full_must_align(self):
        full,prefix,labels,weights,flags=self.args
        with self.assertRaises(ValueError):
            validate_supervision(full,[9,2],labels,weights,flags)
        with self.assertRaises(ValueError):
            validate_supervision(full,prefix,labels[:-1],weights,flags)
    def test_existing_h75_bilingual_family_sampler(self):
        rows=[{"language":"en","family":"x"},{"language":"ar","family":"x"},
              {"language":"en","family":"y"},{"language":"en","family":"z"}]
        selected=select_processor_audit_rows(rows,4)
        self.assertEqual(len(selected),4)
        self.assertEqual({x["language"] for x in selected},{"ar","en"})

if __name__=="__main__":unittest.main()
