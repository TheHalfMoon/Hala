"""H420 counterfactual same-image TWO original native-drag action contract."""
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from h420 import source_counterfactual_dev as x


def original(case_id,end=(800,320),instruction=None, ordered=True):
    return {
      "id":case_id,"split":"train","language":"en",
      "domain":"slider",
      "image_path":"images/slider/verified-original-source.jpg",
      "image_sha256":"a"*64,
      "image_width":1024,"image_height":640,
      "source_task_sha256":"b"*64,
      "ordered":ordered,
      "intent":instruction or f"Drag the source slider to {end[0]} pixels",
      "completion":{
          "type":"drag","start_point":[120,320],
          "end_point":list(end),"ordered":ordered,
          "coordinate_space":"source_image_pixels"
      }
    }


class H420CounterfactualOriginalTests(unittest.TestCase):
    def test_two_original_targets_on_same_image_are_separated(self):
        first=original("h60-slider-fixture-first",(350,320))
        second=original("h60-slider-fixture-second",(850,320))
        candidate=x.candidate_pair([first,second])
        self.assertIsNotNone(candidate)
        rank,a,b,rawa,rawb=candidate
        self.assertEqual(a["image_sha256"],b["image_sha256"])
        self.assertNotEqual(a["body"],b["body"])
        self.assertNotEqual(a["end"],b["end"])
        self.assertGreater(rank[0],0.15)
        self.assertEqual(a["kind"],"original")
        self.assertTrue(a["ordered"])

    def test_same_instruction_is_not_a_counterfactual_even_if_end_changes(self):
        one=original("h60-slider-fixture-one",(250,320),"Drag the source slider to right")
        two=original("h60-slider-fixture-two",(860,320),"Drag the source slider to right")
        self.assertIsNone(x.candidate_pair([one,two]))

    def test_same_target_is_not_counterfactual_even_if_text_changes(self):
        one=original("h60-slider-fixture-one",(650,320),"Drag the unique slider")
        two=original("h60-slider-fixture-two",(650,320),"Move the unique slider")
        self.assertIsNone(x.candidate_pair([one,two]))

    def test_small_geometry_difference_rejected(self):
        one=original("h60-slider-fixture-one",(750,320))
        two=original("h60-slider-fixture-two",(760,320))
        self.assertIsNone(x.candidate_pair([one,two]))

    def test_mixed_original_screen_sha_rejected(self):
        a=original("h60-slider-fixture-one",(200,320))
        b=original("h60-slider-fixture-two",(800,320))
        b["image_sha256"]="c"*64
        self.assertIsNone(x.candidate_pair([a,b]))

    def test_strict_frozen_source_sample_counts(self):
        self.assertEqual(x.PER_DOMAIN,64)
        self.assertEqual(x.TARGET_PAIRS,192)
        self.assertEqual(x.REQUIRED_ACTIONS,384)
        self.assertEqual(x.DOMAINS,("slider","slide_resize","text_highlight"))

    def test_create_only_source_receipt_cannot_claim_model_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=[{"pair_id":0,"image_sha256":"a"*64}]
            meta={"original_unseen_source_pairs_by_domain":
                  {"slider":64,"slide_resize":64,"text_highlight":64},
                  "original_annotated_actions":500554,
                  "source_task_sha_excluded":True,
                  "min_source_coord_target_ratio":0.15}
            report=x.export(fake,meta,tmp)
            self.assertFalse(report["model_evaluation_executed"])
            self.assertFalse(report["model_training_executed"])
            self.assertFalse(report["independent_external_ood"])
            self.assertFalse(report["model_release_qualified"])
            self.assertEqual(report["owner_heldout_source_image_pairs"],192)
            self.assertEqual(report["native_two_instruction_eval_cases"],384)
            with self.assertRaises(FileExistsError):
                x.export(fake,meta,tmp)

if __name__=="__main__":
    unittest.main()
