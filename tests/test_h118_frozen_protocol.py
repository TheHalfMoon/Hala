"""H118 future 64 fresh original GUI scenes and strict endpoint contracts."""
import hashlib,json,unittest
from h118.eval import (frozen,START,END,PER_DOMAIN,ARMS,
                       H117_ADAPTER_SHA,admit_h117_checkpoint,
                       strict_start_anchor,input_proof)
from h116.delta import compose_model_drag_from_delta,SPACE
from h61.official_eval import parse_action,score

def point(role,xy):
    return json.dumps({"role":role,"point":xy,
                       "coordinate_space":"normalized_0_1000"})
def delta(xy):
    return json.dumps({"role":"end_delta","delta":xy,"coordinate_space":SPACE})

class TestH118FutureStrict(unittest.TestCase):
    def test_no_prior_dev_eval_source_reused(self):
        self.assertEqual((START,END,PER_DOMAIN),(76,92,16))
        rows=[{"id":str(i)} for i in range(92)]
        self.assertEqual([x["id"] for x in frozen(rows)],
                         [str(i) for i in range(76,92)])
        with self.assertRaises(ValueError):frozen(rows[:-1])
        self.assertEqual(ARMS,("h106_original_drag",
                               "h114_independent_points",
                               "h117_conditioned_signed_end"))
    def test_checkpoint_requires_exact_owner_verified_binary(self):
        self.assertEqual(H117_ADAPTER_SHA,"32222c4c75f2046cf5914caf1c9554c6bd15bb02c33ac4c4075ea9f1b8c577cf")
        with self.assertRaises(ValueError):
            admit_h117_checkpoint("/no/private/adapter")
    def test_actual_model_predicted_start_required_no_oracle(self):
        valid=point("start",[500,500])
        self.assertEqual(strict_start_anchor(valid,1024,640),[500,500])
        for txt in ("garbage",point("end",[500,500]),point("start",[1001,3])):
            self.assertIsNone(strict_start_anchor(txt,1024,640))
    def test_source_pixel_original_scorer_end_delta_bridge(self):
        scene={"id":"synthetic-scene","domain":"sheet",
               "image_width":1024,"image_height":640,
               "task":{"intent":"Drag sheet item","ordered":True,
                       "start_bbox":[510,318,514,322],
                       "end_bbox":[766,478,770,482]}}
        raw=compose_model_drag_from_delta(
            scene,point("start",[500,500]),delta([250,250]))
        self.assertEqual(parse_action(raw,1024,640)["end_point"],[768,480])
        self.assertEqual(score([scene],{scene["id"]:raw})["task_success"],1)
        self.assertIsNone(compose_model_drag_from_delta(
            scene,point("start",[500,500]),delta([700,0])))
    def test_actual_image_prompt_evidence_source_identity(self):
        scene={"id":"fixed","image_sha256":"a"*64,"prompt":"Original source"}
        good=[{"id":"fixed","image_sha256":"a"*64,
               "prompt_sha256":hashlib.sha256(b"Original source").hexdigest()}]
        input_proof([scene],good)
        with self.assertRaises(ValueError):
            input_proof([scene],[dict(good[0],prompt_sha256="b"*64)])
if __name__=="__main__":unittest.main()
