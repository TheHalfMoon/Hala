"""H115 strict two independent point tasks to unchanged H61 drag scorer."""
import json,unittest
from h115.compose import role_input,parse_point,compose_strict_original_drag
from h61.official_eval import parse_action,score

def point(role,xy):
    return json.dumps({"role":role,"point":xy,
                       "coordinate_space":"normalized_0_1000"},
                       separators=(",",":"))

class TestH115PointToStrictDrag(unittest.TestCase):
    def setUp(self):
        self.source={
            "id":"synthetic-original-gui-eval",
            "domain":"sheet","image_width":1024,"image_height":640,
            "task":{"intent":"Drag the sheet item from origin to destination",
                    "ordered":True,"start_bbox":[510,318,514,322],
                    "end_bbox":[766,478,770,482]}}
    def test_correct_native_points_produce_actual_original_drag_success(self):
        raw=compose_strict_original_drag(self.source,
             point("start",[500,500]),point("end",[750,750]))
        self.assertEqual(parse_action(raw,1024,640)["start_point"],[512,320])
        self.assertEqual(parse_action(raw,1024,640)["end_point"],[768,480])
        self.assertEqual(score([self.source],{self.source["id"]:raw})["task_success"],1)
    def test_invalid_or_wrong_source_coords_do_not_get_repaired(self):
        fence=chr(96)*3
        variants=[
          point("end",[500,500]),point("start",[-1,0]),
          point("start",[1001,0]),point("start",[True,0]),
          point("start",[50.0,50]),"bad",
          fence+"json\n"+point("start",[500,500])+"\n"+fence]
        for raw in variants:
            with self.subTest(raw=raw[:30]):
                self.assertIsNone(compose_strict_original_drag(
                    self.source,raw,point("end",[750,750])))
    def test_source_input_prompt_uses_only_intent_direction_no_target_coords(self):
        start=role_input(self.source,"start")
        end=role_input(self.source,"end")
        self.assertEqual(start["task"],self.source["task"])
        self.assertEqual(end["task"],self.source["task"])
        self.assertIn("START",start["prompt"])
        self.assertIn("END",end["prompt"])
        for p in (start["prompt"],end["prompt"]):
            self.assertNotIn("bbox",p.lower())
            self.assertNotIn("[510",p)
        with self.assertRaises(ValueError):
            role_input(self.source,"both")
if __name__=="__main__":unittest.main()
