"""Owner independent native 0-1000 grid and strict role scoring smoke tests."""
import unittest
from h134.owner_replay import recompute_dev

class H134OwnerReplayContracts(unittest.TestCase):
    def test_exact_16_dev_scene_roles_and_source_alignment(self):
        cases=[]
        responses=[]
        for i in range(16):
            item={"source_id":f"synthetic-{i}",
                  "start_bbox":{"x":90,"y":90,"width":20,"height":20},
                  "end_bbox":{"x":190,"y":190,"width":20,"height":20}}
            cases.append(item)
            responses.extend([
              {"case_id":item["source_id"],"role":"start",
               "raw":'{"point_2d":[104,139]}',
               "prediction":[100,100],"hit":True},
              {"case_id":item["source_id"],"role":"end",
               "raw":'{"point_2d":[208,278]}',
               "prediction":[200,200],"hit":True},
            ])
        source={"cases":cases}
        good=recompute_dev(source,responses)
        self.assertEqual(good["start"]["hit"],16)
        self.assertEqual(good["end"]["hit"],16)
        responses[0]["case_id"]="wrong"
        with self.assertRaises(ValueError):
            recompute_dev(source,responses)
        responses[0]["case_id"]="synthetic-0"
        responses[0]["raw"]='{"point_2d":[999,999]}'
        with self.assertRaises(ValueError):
            recompute_dev(source,responses)
if __name__=="__main__": unittest.main()
