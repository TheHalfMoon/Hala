"""H134 owner-created click-target split safety and DOM-coordinate validation."""
import unittest
from h134.source import SPLITS,FAMILIES,seeds,norm,point

class H134OriginalContracts(unittest.TestCase):
    def test_all_source_splits_disjoint_and_family_balanced(self):
        ids={s:set(seeds(s)) for s in SPLITS}
        self.assertEqual({k:len(v) for k,v in ids.items()},
                         {"train":128,"dev":16,"sealed_test":16})
        self.assertFalse(ids["train"]&ids["dev"])
        self.assertFalse(ids["train"]&ids["sealed_test"])
        self.assertFalse(ids["dev"]&ids["sealed_test"])
        for split,values in ids.items():
            self.assertEqual({family:sum(x%4==i for x in values)
                              for i,family in enumerate(FAMILIES)},
                              {k:len(values)//4 for k in FAMILIES})

    def test_browser_gold_center_is_finite_and_in_view(self):
        self.assertEqual(point({"x":90.5,"y":150.5,"width":21,"height":21}),[101,161])
        self.assertEqual(norm([480,360]),[500,500])
        for box in ({"x":0,"y":0,"width":0,"height":5},
                    {"x":-100,"y":0,"width":10,"height":10},
                    {"x":0,"y":0,"width":float("nan"),"height":10}):
            with self.assertRaises(ValueError):point(box)
if __name__=="__main__":unittest.main()
