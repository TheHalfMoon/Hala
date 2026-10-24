"""H114 source-bound SFT pilot schedule tests; no GPU or model."""
import json
import unittest
from h114.schedule import qualify,SCHEDULE

def point(role,i):
    original=f"TRAIN-source-{i}"
    return {
      "schema":"hala.h113.original-train-frozen-two-points-relative1000.v1",
      "id":original+"::"+role,"original_source_id":original,
      "split":"train","original_source_split":"train","language":"ar" if i%4==0 else "en",
      "family":"original_source_family","endpoint_role":role,
      "source_image_width":1024,"source_image_height":640,
      "image_path":f"media/img{i}.png","image_sha256":f"{i+1:064x}",
      "prompt":f"Locate ONLY the {role.upper()} endpoint in the original screenshot.",
      "completion":json.dumps({"point":[250,500] if role=="start" else [750,500],
               "role":role,"coordinate_space":"normalized_0_1000"},sort_keys=True),
      "source_bound_validation_passed":True,"no_eval_or_test_source":True}

def rows(n):
    return [z for i in range(n) for z in (point("start",i),point("end",i))]

class TestH114Schedule(unittest.TestCase):
    def test_original_source_pairs_and_role_balanced_steps(self):
        result=qualify(rows(4),expected_pairs=4,selected_pairs=2)
        self.assertEqual(result.summary["selected_dual_point_optimizer_steps"],4)
        self.assertEqual(result.summary["start_examples"],2)
        self.assertEqual(result.summary["end_examples"],2)
        self.assertEqual(len(result.selected),4)
        self.assertTrue(result.summary["raw_point_coordinates_in_receipt"] is False)
        self.assertEqual(result.summary["schedule"],SCHEDULE)
    def test_eval_labels_and_broken_original_pairs_refused(self):
        for mode in ("wrong_split","role_missing","other_image","duplicate","wrong_label"):
            items=rows(4)
            if mode=="wrong_split":items[0]["split"]="eval"
            elif mode=="role_missing":items[1]["endpoint_role"]="start"
            elif mode=="other_image":items[1]["image_sha256"]="f"*64
            elif mode=="duplicate":items[2]["id"]=items[0]["id"]
            else:items[0]["completion"]=json.dumps({
                  "point":[1001,10],"role":"start","coordinate_space":"normalized_0_1000"})
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    qualify(items,expected_pairs=4,selected_pairs=2)
    def test_does_not_modify_private_labels_or_pair_order(self):
        data=rows(4)
        before=json.dumps(data,sort_keys=True)
        plan=qualify(data,4,2)
        self.assertEqual(before,json.dumps(data,sort_keys=True))
        self.assertEqual([r["endpoint_role"] for r in plan.selected],
                         ["start","end","start","end"])
    def test_inconsistent_shape_and_counts_fails_closed(self):
        with self.assertRaises(ValueError):qualify(rows(4),4,5)
        with self.assertRaises(ValueError):qualify(rows(3),4,2)

if __name__=="__main__":
    unittest.main()



class TestH114TrainerOverlay(unittest.TestCase):
    def test_original_source_image_and_h71_prompt_unmodified(self):
        from h114.schedule import trainer_row
        derived=point("start",0)
        original={"id":derived["original_source_id"],
                  "split":"train","image_path":derived["image_path"],
                  "image_sha256":derived["image_sha256"],
                  "image_format":"JPEG","language":derived["language"],
                  "family":derived["family"],"prompt":"original H71 full drag",
                  "completion":"original H71 two-point drag","source_schema":"original"}
        old=dict(original)
        new=trainer_row(original,derived)
        self.assertEqual(original,old)
        self.assertEqual(new["image_format"],"JPEG")
        self.assertEqual(new["image_sha256"],original["image_sha256"])
        self.assertEqual(new["completion"],derived["completion"])
        self.assertEqual(new["prompt"],derived["prompt"])
        self.assertNotEqual(new["prompt"],original["prompt"])
    def test_original_media_identity_mismatch_rejected(self):
        from h114.schedule import trainer_row
        d=point("end",0)
        o={"id":d["original_source_id"],"image_path":"other.png",
           "image_sha256":d["image_sha256"],"language":d["language"],
           "family":d["family"]}
        with self.assertRaises(ValueError):
            trainer_row(o,d)

    def test_opt_in_does_not_reinterpret_original_h106(self):
        from training.scripts.train_adapter import train,H114_POINT_SCOPE_NONCE
        self.assertIn("h114.",H114_POINT_SCOPE_NONCE)
        with self.assertRaises(ValueError):
            train(None,None,None,None,None,joint_lora=False,
                  sampling_strategy=None,point_sft_path="fake")
