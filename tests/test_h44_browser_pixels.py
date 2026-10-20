"""H44 original Edge pixel/DOM source split and tamper regression."""
import copy
import hashlib
from pathlib import Path
import unittest

from h44.audit import verify
from h43.factory import canonical

ROOT=Path("h44/assets")


class H44ActualPixelTests(unittest.TestCase):
    def test_all_24_real_browser_pixels_and_oracle_bindings(self):
        model,oracle,report=verify(ROOT)
        self.assertEqual(report["actual_chromium_screenshots"],24)
        self.assertEqual(report["english"],12)
        self.assertEqual(report["arabic"],12)
        self.assertTrue(report["all_controls_unique_visible_no_overlap"])
        self.assertEqual((ROOT/"model_inputs.json").read_bytes(),canonical(model))
        self.assertEqual((ROOT/"evaluator_only.json").read_bytes(),canonical(oracle))
        self.assertEqual((ROOT/"audit.json").read_bytes(),canonical(report))
        self.assertEqual(hashlib.sha256((ROOT/"capture.json").read_bytes()).hexdigest(),
                         "ba87f75f412ba235903ec72e119fc8fb021455a646f8a0c9becb5adc455cd14c")

    def test_no_oracle_answers_or_dom_bounds_in_model_projection(self):
        model,oracle,_=verify(ROOT)
        self.assertEqual(len(model["cases"]),len(oracle["cases"]))
        for source,target in zip(model["cases"],oracle["cases"]):
            self.assertEqual(source["id"],target["id"])
            self.assertFalse(set(source)&{"expected_ref","target_bounds","controls","target_point",
                                          "golden_actions","desired_state","evaluator"})
            self.assertEqual(target["expected_ref"],"row_target")
            self.assertTrue(target["model_not_shown_oracle"])
            self.assertNotIn("row_target",source["goal"])
        self.assertFalse(model["independent_blind_test"])
        self.assertFalse(oracle["independent_blind_test"])

    def test_malformed_browser_metadata_and_forged_targets_fail_closed(self):
        import json
        original=json.loads((ROOT/"capture.json").read_text("utf-8"))
        changes=[
            lambda r:r["rendered_records"][0]["controls"][0].update(name="Wrong"),
            lambda r:r["rendered_records"][0]["controls"][0].update(bounds=[0,0,10,10]),
            lambda r:r["rendered_records"][0].update(image_sha256="a"*64),
            lambda r:r["rendered_records"].pop(),
            lambda r:r.update(native_actions=True),
            lambda r:r.update(no_external_network=False),
            lambda r:r["rendered_records"][0]["controls"][1].update(
               bounds=r["rendered_records"][0]["controls"][0]["bounds"]),
        ]
        for index,fn in enumerate(changes):
            with self.subTest(index=index):
                receipt=copy.deepcopy(original)
                fn(receipt)
                with self.assertRaises((ValueError,KeyError)):
                    verify(ROOT,capture_override=receipt)

    def test_retained_original_english_arabic_and_public_splits(self):
        model,oracle,report=verify(ROOT)
        self.assertEqual(sum(r["split"]=="authoring" for r in model["cases"]),16)
        self.assertEqual(sum(r["split"]=="dev" for r in model["cases"]),8)
        self.assertEqual(sum(r["language"]=="ar" for r in model["cases"]),12)
        self.assertEqual({r["family"] for r in model["cases"] if r["split"]=="dev"},
                         {"harbor_inventory","summit_scheduling"})
        self.assertFalse({r["family"] for r in model["cases"] if r["split"]=="dev"}&
                         {r["family"] for r in model["cases"] if r["split"]=="authoring"})


if __name__=="__main__":
    unittest.main()
