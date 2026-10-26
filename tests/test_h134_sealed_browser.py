"""Synthetic-only strict one-time sealed native response-to-pixel contract."""
import copy
import unittest
from h134.owner_sealed_browser import verify_native

def pair():
    source={"cases":[]}
    raw=[]
    for i in range(16):
        name=f"synthetic-{i}"
        source["cases"].append({
          "source_id":name,"image_sha256":f"sha-{i}",
          "start_bbox":{"x":95,"y":95,"width":10,"height":10},
          "end_bbox":{"x":195,"y":195,"width":10,"height":10}})
        raw.extend([
          {"source_id":name,"image_sha256":f"sha-{i}",
           "role":"start","raw":'{"point_2d":[104,139]}',
           "prediction":[100,100],"hit":True},
          {"source_id":name,"image_sha256":f"sha-{i}",
           "role":"end","raw":'{"point_2d":[208,278]}',
           "prediction":[200,200],"hit":True}])
    return source,raw

class H134SealedBrowserChecks(unittest.TestCase):
    def test_every_raw_point_must_match_true_case_and_role(self):
        source,raw=pair()
        model=verify_native(source,raw)
        self.assertEqual(len(model),16)
        self.assertEqual(model["synthetic-0"]["start"],[100,100])
        changed=copy.deepcopy(raw)
        changed[0]["source_id"]="wrong"
        with self.assertRaises(ValueError):verify_native(source,changed)
        changed=copy.deepcopy(raw)
        changed[0]["raw"]='{"point_2d":[300,300]}'
        with self.assertRaises(ValueError):verify_native(source,changed)
if __name__=="__main__":unittest.main()
