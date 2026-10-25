"""H118 independent 64 fresh DragOn scene evaluation contract, not performance."""
import json
import unittest
from h118.eval import (START,PER_DOMAIN,END,ARMS,frozen,
                       admit_h117_checkpoint,H117_ADAPTER_SHA,
                       strict_start_anchor,input_proof)
from h116.delta import prompt_end_delta,compose_model_drag_from_delta
from h61.official_eval import parse_action,score

def answer(role,key,coord,space):
    return json.dumps({"role":role,key:coord,"coordinate_space":space},
                      sort_keys=True,separators=(",",":"))

class TestH118PreRegisteredGui(unittest.TestCase):
    def test_new_original_eval_76_through_91(self):
        self.assertEqual((START,PER_DOMAIN,END),(76,16,92))
        self.assertEqual(ARMS,("h106_original_drag",
                               "h114_independent_points",
                               "h117_conditioned_signed_end"))
        full=[{"id":str(i)} for i in range(92)]
        self.assertEqual([r["id"] for r in frozen(full)],
                         [str(i) for i in range(76,92)])
        with self.assertRaises(ValueError):frozen(full[:-1])
    def test_H117_checkpoint_fail_closed_without_real_binary(self):
        self.assertEqual(H117_ADAPTER_SHA,"32222c4c75f2046cf5914caf1c9554c6bd15bb02c33ac4c4075ea9f1b8c577cf")
        with self.assertRaises(ValueError):
            admit_h117_checkpoint("/unknown/owner/checkpoint")
    def test_only_model_start_may_anchor_new_signed_end(self):
        scene={"id":"synthetic","domain":"slider","image_width":1024,"image_height":640,
            "task":{"intent":"Drag handle to destination","ordered":True,
                    "start_bbox":[510,318,514,322],
                    "end_bbox":[766,478,770,482]}}
        start=answer("start","point",[500,500],"normalized_0_1000")
        delta=answer("end_delta","delta",[250,250],"normalized_displacement_1000")
        predicted=strict_start_anchor(start,1024,640)
        self.assertEqual(predicted,[500,500])
        actual_prompt=prompt_end_delta(scene["task"]["intent"],predicted)
        self.assertIn("[500,500]",actual_prompt)
        self.assertNotIn("start_bbox",actual_prompt)
        complete=compose_model_drag_from_delta(scene,start,delta)
        self.assertEqual(parse_action(complete,1024,640)["end_point"],[768,480])
        self.assertEqual(score([scene],{"synthetic":complete})["task_success"],1)
        self.assertIsNone(strict_start_anchor("not json",1024,640))
        self.assertIsNone(strict_start_anchor(
             answer("end_delta","delta",[100,100],"normalized_displacement_1000"),
             1024,640))
    def test_exact_original_media_and_prompt_identify_neural_attempts(self):
        import hashlib
        src=[{"id":"a","image_sha256":"0"*64,"prompt":"Original source instruction"}]
        proof=[{"id":"a","image_sha256":"0"*64,
                "prompt_sha256":hashlib.sha256(src[0]["prompt"].encode()).hexdigest()}]
        input_proof(src,proof)
        proof[0]["prompt_sha256"]="f"*64
        with self.assertRaises(ValueError):
            input_proof(src,proof)
if __name__=="__main__":unittest.main()
