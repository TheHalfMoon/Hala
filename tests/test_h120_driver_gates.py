"""H120 independent CPU gates for owner-verified checkpoint and anchors."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from h120.eval import admit_h119_checkpoint, _conditioned_inputs, expected_h119_training_identity

class GateTests(unittest.TestCase):
    def test_unverified_h119_not_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                admit_h119_checkpoint(folder,Path("."))

    def test_model_only_anchor_input(self):
        sample=dict(id="scene",image_width=1024,image_height=640,
                    task=dict(intent="Drag the handle right",ordered=True,
                              start_bbox=[1,2,3,4],end_bbox=[900,500,910,510]))
        raw='{"role":"start","point":[200,300],"coordinate_space":"normalized_0_1000"}'
        selected=_conditioned_inputs([sample],{"scene":raw})
        self.assertEqual(len(selected),1)
        self.assertIn("[200,300]",selected[0]["prompt"])
        self.assertNotIn("bbox",selected[0]["prompt"])
        self.assertEqual(_conditioned_inputs([sample],{"scene":"bad"}),[])

    def test_expected_training_identity(self):
        identity=expected_h119_training_identity(Path("."))
        self.assertEqual(len(identity),64)
        self.assertEqual(identity,expected_h119_training_identity(Path(".")))

if __name__=="__main__":
    unittest.main()
