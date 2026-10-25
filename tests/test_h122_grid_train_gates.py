"""H122 source-only strict absolute END spatial grid and training isolation."""
import json
import tempfile
import unittest
from pathlib import Path
from h122.grid import grid_cell,end_prompt,end_completion,parse_end,verify_overlay
from h122.schedule import qualify_owner_private,ORIGINAL,TRAIN,HOLDOUT
from h122.training_plan import checkpoint_due,STEPS,get_plan
from h122.supervision import coarse_fine_weights
from h106.research_train import config as old_config
from training.scripts.train_adapter import train,H122_GRID_SCOPE_NONCE
from training.preflight import TrainingConfig

class H122Contracts(unittest.TestCase):
    def test_coarse_to_fine_roundtrips_edges_exactly(self):
        for x,y in ((0,0),(99,99),(100,100),(999,999),(1000,1000),(500,400)):
            completion=end_completion([x,y])
            cell=grid_cell([x,y])
            self.assertEqual(cell,[min(9,x//100),min(9,y//100)])
            self.assertEqual(parse_end(completion,1024,640),
                             [min(1023,round(x*1024/1000)),min(639,round(y*640/1000))])
        self.assertIsNone(parse_end(end_completion([300,200]).replace('[3,2]','[9,9]'),1024,640))
        self.assertIsNone(parse_end(end_completion([300,200]).replace('"end"','"start"'),1024,640))
        self.assertIsNone(parse_end('{"point":[10,20],"point":[30,40],"cell":[0,0],"role":"end","coordinate_space":"normalized_0_1000"}',1024,640))

    def test_exact_original_source_and_box_gate(self):
        base={"id":"scene1","split":"train","image_path":"images/scene1.jpg",
            "image_sha256":"a"*64,"language":"ar","family":"slider",
            "image_width":1000,"image_height":1000,"end_bbox":[400,300,450,360]}
        overlay={"original_source_id":"scene1","split":"train",
            "image_path":"images/scene1.jpg","image_sha256":"a"*64,
            "language":"ar","family":"slider","endpoint_role":"end",
            "schema":"hala.h122.original-end-absolute-grid10-finepoint.v1",
            "source_image_width":1000,"source_image_height":1000,
            "completion":end_completion([420,330])}
        self.assertTrue(verify_overlay(base,overlay))
        for mutation in ({"split":"eval"},{"image_sha256":"b"*64},
                         {"completion":end_completion([950,330])}):
            with self.assertRaises(ValueError):
                verify_overlay(base,{**overlay,**mutation})

    def test_no_holdout_optimizer_h122_448(self):
        self.assertEqual((ORIGINAL,TRAIN,HOLDOUT,STEPS),(512,448,64,448))
        self.assertEqual([i for i in range(1,449) if checkpoint_due(i)],
                         list(range(64,449,64)))
        for n in (0,True,449):
            with self.assertRaises(ValueError): checkpoint_due(n)
        for n in (256,512,1024):
            with self.assertRaises(ValueError):get_plan([],Path("."),None,n)

    def test_coarse_fine_weights_metadata_digits_not_weighted(self):
        completion=end_completion([420,330])
        toks=list(range(len(completion)))
        weights=coarse_fine_weights(toks,lambda ids:completion[ids[0]],8.0)
        self.assertEqual(len(weights),len(completion))
        self.assertGreater(sum(x>1 for x in weights),0)
        string=completion
        metadata_start=string.index('normalized_0_1000')
        self.assertTrue(all(w==1 for w in weights[metadata_start:metadata_start+len('normalized_0_1000')]))

    def test_train_path_fail_closed_prior_models_stay_separate(self):
        self.assertIn("h122",H122_GRID_SCOPE_NONCE)
        for kwargs in (
            dict(joint_lora=False),
            dict(joint_lora=True,sampling_strategy=None),
            dict(joint_lora=True,sampling_strategy="old"),
            dict(joint_lora=True,sampling_strategy="hala.h106.joint-vision-language-h105-source-prefix256.v1",resume=True),
            dict(joint_lora=True,sampling_strategy="hala.h106.joint-vision-language-h105-source-prefix256.v1",robust_anchor_sft_path="legacy"),
            dict(joint_lora=True,sampling_strategy="hala.h106.joint-vision-language-h105-source-prefix256.v1",point_sft_path="legacy"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    train("missing","missing","missing","missing","missing",
                          end_grid_sft_path="private_source",**kwargs)

if __name__=="__main__":unittest.main()
