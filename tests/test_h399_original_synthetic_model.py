"""Original H399 synthetic-only input and bilingual prompt safety tests."""
import hashlib
import json
import unittest
from h399.kaggle.h399_original_qwen3vl_synthetic_lora import (
    MODEL,MODEL_REV,sample)

class H399InventedSourceTests(unittest.TestCase):
    def test_original_self_generated_balanced_source_and_splits(self):
        self.assertEqual(MODEL,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(MODEL_REV,"89644892e4d85e24eaac8bacfd4f463576704203")
        train=[sample(i,3981) for i in range(24)]
        dev=[sample(i,3982) for i in range(6)]
        alt=[sample(i,3983,"unseen") for i in range(6)]
        def fingerprints(rows):
            return {hashlib.sha256(image.tobytes()+text.encode("utf8")).hexdigest()
                    for image,text,_,_,_ in rows}
        a,b,c=map(fingerprints,(train,dev,alt))
        self.assertEqual((len(a),len(b),len(c)),(24,6,6))
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual([sum(row[4]==language for row in train)
                          for language in (0,1)],[12,12])
        for image,text,gold,point,language in train+dev+alt:
            self.assertEqual(image.size,(224,224))
            self.assertIn("point_2d",text)
            self.assertEqual(json.loads(gold),{"point_2d":point})
            self.assertTrue(all(0<=p<=1000 for p in point))
            self.assertIn(language,(0,1))
            self.assertEqual(sample(0,3981)[0].tobytes(),
                             sample(0,3981)[0].tobytes())
if __name__=="__main__":
    unittest.main()
