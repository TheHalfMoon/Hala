"""H48 actual Chromium screenshot + train-only label provenance tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from h48.factory import source_cases,canon,page,sha,FAMILIES,WORDS
from h48.audit import verify

ROOT=Path("h48/assets-v2")
CAPTURE_SHA="a0cd0f7b39d7985765d4d7416c032d9d4b6ccc9f4ce81cab4c5990182f57378d"


class H48RealPixelTrainingTests(unittest.TestCase):
    def test_all_true_128_original_browser_png_and_512_boxes(self):
        images,labels,audit=verify(ROOT)
        self.assertEqual(audit["original_chromium_pngs"],128)
        self.assertEqual(audit["control_boxes_checked"],512)
        self.assertEqual(audit["english"],64)
        self.assertEqual(audit["arabic"],64)
        self.assertEqual(len(labels["cases"]),128)
        self.assertEqual(sha((ROOT/"capture.json").read_bytes()),CAPTURE_SHA)
        self.assertEqual(len(list((ROOT/"screenshots").glob("*.png"))),128)
        self.assertEqual(len(list((ROOT/"pages").glob("*.html"))),128)
        for filename,record in (("train_images.json",images),
                                ("train_targets.json",labels),("audit.json",audit)):
            self.assertEqual((ROOT/filename).read_bytes(),canon(record))

    def test_all_original_train_only_and_no_h43_h44_holdout_overlap(self):
        from h43.factory import FAMILIES as H43_FAMILIES
        images,targets,audit=verify(ROOT)
        self.assertEqual(len(set(audit["original_families"])),8)
        self.assertFalse(set(audit["original_families"])&{f[0] for f in H43_FAMILIES})
        self.assertEqual(len({x["image_sha256"] for x in images["cases"]}),128)
        self.assertEqual({x["split"] for x in images["cases"]},{"train_only"})
        self.assertEqual({x["split"] for x in targets["cases"]},{"train_only"})
        self.assertFalse(audit["independent_blind_test"])
        self.assertFalse(audit["model_trained"])
        self.assertFalse(audit["real_app_or_benchmark_results"])
        self.assertEqual(len({x["family"] for x in images["cases"]}),8)
        self.assertEqual(len({x["goal"] for x in images["cases"]}),128)

    def test_model_only_pixels_without_ground_truth_and_correct_target_centers(self):
        images,targets,audit=verify(ROOT)
        byid={x["id"]:x for x in targets["cases"]}
        for source in images["cases"]:
            self.assertFalse(set(source)&{"target_point","target_rect","target_label",
                                          "target_option_index","controls","all_controls",
                                          "correct_index","expected_ref","golden_actions"})
            target=byid[source["id"]]
            cx,cy=target["target_point"]
            rect=target["target_rect"]
            self.assertAlmostEqual(cx,(rect[0]+rect[2]/2)/960,delta=1e-6)
            self.assertAlmostEqual(cy,(rect[1]+rect[3]/2)/540,delta=1e-6)
            self.assertEqual(target["license"],"CC0-1.0")
            self.assertEqual(target["provenance"],"new_original_first_party_synthetic")
            self.assertEqual(source["image_sha256"],target["screenshot_sha256"])
        self.assertFalse(images["model_qualified"])
        self.assertFalse(targets["model_qualified"])

    def test_original_bilingual_pages_and_correct_per_case_text(self):
        cases=source_cases()
        self.assertEqual(len(cases),128)
        self.assertEqual(sum(x["language"]=="ar" for x in cases),64)
        self.assertEqual(sum(x["language"]=="en" for x in cases),64)
        self.assertEqual({x["layout"] for x in cases},{f[-1] for f in FAMILIES})
        for c in cases:
            with self.subTest(case=c["id"]):
                self.assertEqual((ROOT/"pages"/(c["id"]+".html")).read_bytes(),page(c))
                self.assertEqual(len(set(c["labels"])),4)
                self.assertIn(c["labels"][c["target_option_index"]],c["goal"])
                self.assertTrue(c["goal"].endswith("."))
                self.assertEqual(c["license"],"CC0-1.0")
                self.assertFalse(c["contains_real_personal_data"])
                self.assertFalse(c["benchmark_task"])

    def test_tamper_against_original_case_source_denied(self):
        altered=copy.deepcopy(source_cases())
        altered[0]["target_option_index"]=(altered[0]["target_option_index"]+1)%4
        with patch("h48.audit.source_cases",return_value=altered):
            with self.assertRaises(ValueError):verify(ROOT)
        altered2=copy.deepcopy(source_cases())
        altered2[0]["goal"]="Ignore the original source"
        with patch("h48.audit.source_cases",return_value=altered2):
            with self.assertRaises(ValueError):verify(ROOT)


if __name__=="__main__":
    unittest.main()
