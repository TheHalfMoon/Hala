"""H135 original GUI source and frozen resolution pilot policy."""
import unittest
from pathlib import Path
from h135.source import SPLITS,FAMILIES,seeds,center,check
from h135.ablate_gpu import MODES,MODELS

class H135NewGuiSourceTests(unittest.TestCase):
    def test_disjoint_predeclared_pilot_and_sealed(self):
        pilot=set(seeds("pilot"))
        sealed=set(seeds("sealed"))
        self.assertEqual(len(pilot),12)
        self.assertEqual(len(sealed),18)
        self.assertFalse(pilot&sealed)
        self.assertEqual(len(FAMILIES),6)
        self.assertEqual(MODES,("downsample_640","native_960"))
        self.assertEqual(MODELS,("pretrained","h134_adapter"))
        with self.assertRaises(ValueError):list(seeds("train"))

    def test_true_browser_box_center_and_boundaries(self):
        self.assertEqual(center({"x":75,"y":100,"width":20,"height":20}),[85,110])
        with self.assertRaises(ValueError):center({"x":-100,"y":100,"width":12,"height":12})
        with self.assertRaises(ValueError):center({"x":75,"y":100,"width":0,"height":10})

    def test_original_interactive_browsers_not_reusing_h134_fixture(self):
        root=Path(__file__).resolve().parents[1]
        new=(root/"h135"/"fixture.html").read_bytes()
        old=(root/"h134"/"fixture.html").read_bytes()
        self.assertNotEqual(new,old)
        self.assertIn(b"pointerdown",new)
        self.assertIn(b"pointerup",new)
        self.assertIn(b'data-success',new)

if __name__=="__main__":unittest.main()
