"""Synthetic H426 token-mask tests; no real source model or new GPU claim."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("h426_loss",ROOT/"h426/target_aware_loss.py")
m=importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)

def parts():
    return ['{"type":"drag","start_point":[','200',',','300','],"end_point":[',
            '600',',','350','],"ordered":true,"coordinate_space":"normalized_1000"}',
            '<|im_end|>']

class H426OriginalCoordinateWeights(unittest.TestCase):
    def test_resize_targets_end_not_start(self):
        w=m.source_coordinate_token_weights(parts(),"slide_resize")
        self.assertEqual(w[1],8)
        self.assertEqual(w[3],8)
        self.assertEqual(w[5],16)
        self.assertEqual(w[7],16)
        self.assertEqual(w[8],1)

    def test_text_highlight_targets_both_coordinates(self):
        w=m.source_coordinate_token_weights(parts(),"text_highlight")
        self.assertEqual([w[i] for i in (1,3,5,7)],[12]*4)

    def test_slider_retains_equal_source_weight(self):
        w=m.source_coordinate_token_weights(parts(),"slider")
        self.assertEqual([w[i] for i in (1,3,5,7)],[8]*4)

    def test_target_grammar_meta_digits_not_weighted(self):
        w=m.source_coordinate_token_weights(parts(),"slider")
        self.assertEqual(w[8],1)
        self.assertEqual(w[0],1)
        self.assertEqual(w[-1],1)

    def test_character_level_tokenization_preserves_geometric_fields(self):
        raw="".join(parts())
        w=m.source_coordinate_token_weights(list(raw),"slide_resize")
        self.assertEqual(len(w),len(raw))
        self.assertEqual(sum(x==16 for x in w),6)
        self.assertEqual(sum(x==8 for x in w),6)

    def test_malformed_or_extra_source_action_rejected(self):
        for a in (["unknown"], parts()+parts(),
                  ['{"type":"drag","start_point":[100],"end_point":[400,300],"ordered":true,"coordinate_space":"normalized_1000"}'],
                  ['{"type":"drag","start_point":[100,200],"end_point":[1100,300],"ordered":true,"coordinate_space":"normalized_1000"}']):
            with self.subTest(v=a),self.assertRaises(ValueError):
                m.source_coordinate_token_weights(a,"slider")

    def test_unseen_domain_or_token_missing_rejected(self):
        with self.assertRaises(ValueError):
            m.source_coordinate_token_weights(parts(),"unknown")
        with self.assertRaises(ValueError):
            m.source_coordinate_token_weights([],"slide_resize")

    def test_no_external_release_or_training_effect(self):
        source=(ROOT/"h426/target_aware_loss.py").read_text("utf8")
        self.assertNotIn("kaggle kernels push",source)
        self.assertNotIn("model.save_pretrained",source)
        self.assertIn("Unexecuted H426 research experiment loss",source)

if __name__=="__main__":
    unittest.main()
