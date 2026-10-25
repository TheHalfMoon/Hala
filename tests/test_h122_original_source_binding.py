"""Regression: H61 SFT deliberately has no original END boxes/dimensions.

H122 must recover these from original TRAIN shards, verify exact original
drag completion, and never invent an END box from an SFT prompt.
"""
import json
import unittest

from h122.grid import end_completion,verify_overlay
from h122.source_bound import original_for_sft


def fixtures():
    original={
      "id":"source-train-001","split":"train","action_type":"drag",
      "image_sha256":"a"*64,"image_path":"slider/images/001.jpg",
      "image_width":1000,"image_height":600,
      "language":"en","family":"hcompany_dragon_h61_slider_train",
      "ordered":True,
      "start_bbox":[80,80,120,120],
      "end_bbox":[680,420,720,460],
      "completion":{"type":"drag","start_point":[100,100],
                    "end_point":[700,440],
                    "coordinate_space":"source_image_pixels","ordered":True}
    }
    sft={
      "id":original["id"],"split":"train",
      "image_sha256":original["image_sha256"],
      "image_path":original["image_path"],
      "language":original["language"],"family":original["family"],
      "completion":json.dumps(original["completion"],sort_keys=True)}
    assert "end_bbox" not in sft and "image_width" not in sft
    derived={
      "id":original["id"]+"::h122_end",
      "original_source_id":original["id"],
      "split":"train","original_source_split":"train",
      "schema":"hala.h122.original-end-absolute-grid10-finepoint.v1",
      "endpoint_role":"end",
      "image_sha256":original["image_sha256"],
      "image_path":original["image_path"],
      "language":original["language"],"family":original["family"],
      "source_image_width":1000,"source_image_height":600,
      "completion":end_completion([700,733])}
    return original,sft,derived


class OriginalSourceProof(unittest.TestCase):
    def test_sft_missing_box_still_admitted_via_original_source(self):
        original,sft,derived=fixtures()
        restored=original_for_sft(sft,{original["id"]:original})
        self.assertIs(restored,original)
        self.assertTrue(verify_overlay(restored,derived))
        with self.assertRaises((KeyError,ValueError)):
            verify_overlay(sft,derived)

    def test_mismatched_image_sha_refused_before_end_score(self):
        original,sft,_=fixtures()
        changed={**sft,"image_sha256":"b"*64}
        with self.assertRaises(ValueError):
            original_for_sft(changed,{original["id"]:original})

    def test_original_action_or_end_box_drift_rejected(self):
        original,sft,_=fixtures()
        changed={**original,"end_bbox":[10,10,20,20]}
        with self.assertRaises(ValueError):
            original_for_sft(sft,{changed["id"]:changed})
        changed={**original,"completion":{**original["completion"],
                                           "end_point":[750,440]}}
        with self.assertRaises(ValueError):
            original_for_sft(sft,{changed["id"]:changed})

    def test_eval_original_cannot_substitute_train_original(self):
        original,sft,_=fixtures()
        changed={**original,"split":"eval"}
        with self.assertRaises(ValueError):
            original_for_sft(sft,{original["id"]:changed})
        sft["split"]="eval"
        with self.assertRaises(ValueError):
            original_for_sft(sft,{original["id"]:original})

if __name__=="__main__":
    unittest.main()
