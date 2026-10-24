"""H116 genuine original TRAIN start anchor -> signed END motion protocol."""
import json
import unittest
from h113.dual_point import point_prompt
from h116.delta import (derive_pair,compose_model_drag_from_delta,
                         prompt_end_delta,SPACE,DELTA_ROLE)
from h61.official_eval import parse_action,score

def row(role,pt):
    intent="Drag the visible slider from its current handle to the target"
    return {
     "id":"synthetic-H71-original::"+role,
     "original_source_id":"synthetic-H71-original",
     "endpoint_role":role,
     "split":"train","no_eval_or_test_source":True,
     "source_bound_validation_passed":True,
     "language":"en","family":"source-original",
     "image_path":"source.png","image_sha256":"a"*64,
     "source_image_width":1024,"source_image_height":640,
     "prompt":point_prompt(intent,True,role),
     "completion":json.dumps({"role":role,"point":pt,
                    "coordinate_space":"normalized_0_1000"},sort_keys=True)}

def answer(role,values):
    if role=="start":
        return json.dumps({"role":"start","point":values,
                           "coordinate_space":"normalized_0_1000"})
    return json.dumps({"role":"end_delta","delta":values,
                       "coordinate_space":SPACE})

class TestH116ConditionedDelta(unittest.TestCase):
    def test_exact_original_train_signed_displacement(self):
        a,b=derive_pair(row("start",[500,500]),row("end",[750,750]))
        self.assertEqual(a["completion"],row("start",[500,500])["completion"])
        self.assertEqual(json.loads(b["completion"])["delta"],[250,250])
        self.assertIn("[500,500]",b["prompt"])
        self.assertEqual(b["endpoint_role"],DELTA_ROLE)
        self.assertEqual(b["image_sha256"],a["image_sha256"])
        self.assertEqual(b["split"],"train")
    def test_backward_move_signed_delta_is_not_collapsed(self):
        a,b=derive_pair(row("start",[750,750]),row("end",[250,250]))
        self.assertEqual(json.loads(b["completion"])["delta"],[-500,-500])
    def test_strict_original_H61_GUI_composition_can_be_truly_successful(self):
        src={"image_width":1024,"image_height":640,"domain":"slider",
             "id":"synthetic-only",
             "task":{"intent":"Drag from handle to endpoint","ordered":True,
                     "start_bbox":[510,318,514,322],
                     "end_bbox":[766,478,770,482]}}
        result=compose_model_drag_from_delta(src,answer("start",[500,500]),
                                            answer("end_delta",[250,250]))
        self.assertEqual(parse_action(result,1024,640)["end_point"],[768,480])
        self.assertEqual(score([src],{"synthetic-only":result})["task_success"],1)
    def test_invalid_role_json_and_displacement_fail_closed(self):
        src={"image_width":1024,"image_height":640,
             "task":{"intent":"Drag handle to target","ordered":True}}
        start=answer("start",[500,500])
        for raw in (answer("end_delta",[1001,0]),
                    answer("end_delta",[-2000,0]),
                    answer("end_delta",[True,1]),
                    answer("end_delta",[500.5,1]),
                    json.dumps({"role":"end","delta":[1,2],
                                "coordinate_space":SPACE}),"not JSON"):
            with self.subTest(raw=raw[:30]):
                self.assertIsNone(compose_model_drag_from_delta(src,start,raw))
        self.assertIsNone(compose_model_drag_from_delta(
            src,start,answer("end_delta",[501,0])))
        with self.assertRaises(ValueError):
            derive_pair(row("start",[500,500]),row("start",[750,750]))
        with self.assertRaises(ValueError):
            prompt_end_delta("Drag handle",[1001,0])
if __name__=="__main__":unittest.main()
