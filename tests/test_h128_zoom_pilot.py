"""H128 strict crop affine / paired scoring synthetic tests."""
import unittest
from h128.zoom import Crop,anchored_crop,native_to_original,score
from h128.source import PREVIOUS,TOTAL,PER_FAMILY,H126_SHA,propose,FAMILIES

def fixtures():
    cases=[];results=[]
    for i in range(16):
        family=FAMILIES[i//4]
        cases.append({"id":f"synthetic-{i}","family":family,
                      "image_sha256":f"sha-{i}","image_width":800,
                      "image_height":600,"start_bbox":[100,100,110,110],
                      "end_bbox":[600,400,610,410]})
        results.append({"id":f"synthetic-{i}","image_sha256":f"sha-{i}",
                        "direct_start":(105,105),"direct_end":(605,405),
                        "zoom_start":(105,105),"zoom_end":(605,405),
                        "crop_start":anchored_crop(800,600,(105,105)),
                        "crop_end":anchored_crop(800,600,(605,405))})
    return cases,results

class H128CropTests(unittest.TestCase):
    def test_clip_border_crop_and_reverse_pixel_map(self):
        a=anchored_crop(800,600,(0,0))
        self.assertEqual(a,Crop(0,0,400,300))
        self.assertEqual(native_to_original(a,(0,0),800,600),(0,0))
        b=anchored_crop(800,600,(799,599))
        self.assertEqual(b,Crop(400,300,800,600))
        self.assertEqual(native_to_original(b,(399,299),800,600),(799,599))
        c=anchored_crop(800,600,(400,300))
        self.assertEqual(c,Crop(200,150,600,450))
        self.assertEqual(native_to_original(c,(200,150),800,600),(400,300))

    def test_invalid_crop_and_prediction_fail_closed(self):
        for point in [(-1,10),(800,0),(0,float("nan")),(1,2,3)]:
            with self.assertRaises(ValueError):anchored_crop(800,600,point)
        with self.assertRaises(ValueError):
            native_to_original(Crop(200,150,600,450),(400,0),800,600)
        cases,preds=fixtures()
        preds[0]["direct_start"]=(-1,0)
        with self.assertRaises(ValueError):score(cases,preds)

    def test_paired_score_not_model_qualification(self):
        cases,preds=fixtures()
        result=score(cases,preds)
        self.assertEqual(result["aggregate"]["direct_full"],16)
        self.assertEqual(result["aggregate"]["zoom_full"],16)
        self.assertFalse(result["model_accuracy_proven"])
        preds[0]["zoom_start"]=(300,300)
        self.assertEqual(score(cases,preds)["aggregate"]["zoom_full"],15)
        preds[0]["id"]="wrong-id"
        with self.assertRaises(ValueError):score(cases,preds)

    def test_new_source_exclusion_is_mandatory(self):
        self.assertEqual(PREVIOUS,2688)
        self.assertEqual(TOTAL,16)
        self.assertEqual(PER_FAMILY,4)
        self.assertEqual(len(H126_SHA),64)
        with self.assertRaises(ValueError):propose([],None)

if __name__=="__main__":
    unittest.main()
