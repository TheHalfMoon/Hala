"""H48 authentic browser pixel and first-party train/DEV contamination tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from h48.audit import verify
from h48.factory import scenes,validate,PROVENANCE,canonical,sha

ROOT=Path("h48/assets")
CAPTURE_SHA="a01db14518f6ce7a2bf7fa89212eae4a78edbd69ea2fbc5b8bd5ccaa8dc3f127"

class H48GroundingTests(unittest.TestCase):
    def test_96_chromium_images_frozen_and_576_observed_buttons(self):
        model,train,labels,audit=verify(ROOT)
        self.assertEqual(sha((ROOT/"capture.json").read_bytes()),CAPTURE_SHA)
        self.assertEqual(audit["original_browser_screenshots"],96)
        self.assertEqual(audit["visible_controls_verified"],576)
        self.assertEqual(audit["original_authoring"],64)
        self.assertEqual(audit["public_dev"],32)
        self.assertEqual(audit["arabic"],48)
        self.assertEqual(audit["english"],48)
        self.assertFalse(audit["actual_model_results"])
        for name,obj in (("model_inputs.json",model),
                         ("authoring_labels.json",train),
                         ("evaluator_only.json",labels),("audit.json",audit)):
            self.assertEqual((ROOT/name).read_bytes(),canonical(obj))
        self.assertEqual(len(list((ROOT/"png").glob("*.png"))),96)

    def test_no_answers_to_model_and_public_dev_excluded_from_train(self):
        model,train,oracle,_=verify(ROOT)
        self.assertEqual(len(model["cases"]),96)
        self.assertEqual(len(train["examples"]),64)
        self.assertEqual(len(oracle["cases"]),96)
        self.assertEqual(sum(c["language"]=="ar" for c in model["cases"]),48)
        self.assertFalse({x["family"] for x in train["examples"]}&
                         {x["family"] for x in oracle["cases"] if x["split"]=="dev"})
        self.assertTrue(all(x["split"]=="authoring" for x in train["examples"]))
        for input in model["cases"]:
            self.assertFalse(set(input)&{"target_code","expected_ref_code","target_index",
                      "target_bounds","labelled_point","observed_buttons","expected_index"})
        self.assertFalse(model["independent_blind_test"])
        self.assertFalse(oracle["independent_blind_test"])
        self.assertEqual({r["family"] for r in model["cases"] if r["split"]=="dev"},
                         {"calendar_slots","expense_review"})

    def test_all_correct_centers_bind_unique_target_not_distractor(self):
        _,train,oracle,_=verify(ROOT)
        for entry in oracle["cases"]:
            with self.subTest(case=entry["id"]):
                bounds=entry["target_bounds"]
                p=entry["labelled_point"]
                self.assertAlmostEqual(p[0],(bounds[0]+bounds[2]/2)/1024,places=6)
                self.assertAlmostEqual(p[1],(bounds[1]+bounds[3]/2)/640,places=6)
                matched=[]
                for c in entry["observed_buttons"]:
                    x,y,w,h=c["bounds"]
                    if x <= p[0]*1024 <= x+w and y <= p[1]*640 <= y+h:
                        matched.append(c["index"])
                self.assertEqual(matched,[entry["expected_index"]])
                self.assertIn(entry["expected_ref_code"],
                              entry["observed_buttons"][entry["expected_index"]]["name"])

    def test_layouts_and_language_are_genuinely_distinct(self):
        rows=scenes()
        self.assertEqual({r["layout"] for r in rows},
                         {"modal","inbox","settings","board","calendar","expenses"})
        self.assertEqual({r["split"] for r in rows},{"authoring","dev"})
        self.assertEqual(len({r["goal"] for r in rows}),96)
        self.assertEqual(sum(r["rtl"] for r in rows),48)
        self.assertTrue(all(r["source"]==PROVENANCE for r in rows))
        validate(rows)
        for index,modifier in enumerate((
            lambda x:x[0].update(split="dev"),
            lambda x:x[0].update(target_index=1),
            lambda x:x[0]["source"].update(license="unknown"),
            lambda x:x[-1].update(goal="malformed"),
            lambda x:x.pop(),
        )):
            with self.subTest(index=index):
                copyrows=copy.deepcopy(rows)
                modifier(copyrows)
                with self.assertRaises(ValueError):validate(copyrows)

    def test_observed_forged_image_geometry_and_false_claims_rejected(self):
        receipt=json.loads((ROOT/"capture.json").read_text("utf-8"))
        changes=(
            lambda x:x["records"][0].update(png_sha256="a"*64),
            lambda x:x["records"][0]["controls"][0].update(name="fake"),
            lambda x:x["records"][0]["controls"][0].update(bounds=[1,1,3,4]),
            lambda x:x["records"][0]["controls"][1].update(
                 bounds=x["records"][0]["controls"][0]["bounds"]),
            lambda x:x["records"].pop(),
            lambda x:x.update(model_inference=True),
            lambda x:x.update(network_used=True),
        )
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                copyreceipt=copy.deepcopy(receipt)
                change(copyreceipt)
                with self.assertRaises((ValueError,KeyError,TypeError)):
                    verify(ROOT,raw_override=copyreceipt)

if __name__=="__main__":unittest.main()
