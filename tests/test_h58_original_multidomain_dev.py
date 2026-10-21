"""H58 real Edge evidence and first-party-only fixture model-protocol tests."""
from __future__ import annotations
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from h58.audit import audit,strict_file,ORIGINAL_SHA
from h58.score import SCHEMA,score,parse_action
ROOT=Path(__file__).resolve().parents[1]/"h58/assets"
def fixture(mode="forward"):
    r=strict_file(ROOT/"capture.json")
    cases=[]
    for c in r["cases"]:
        start,end=c["start_point"],c["end_point"]
        if mode=="reverse":start,end=end,start
        elif mode=="wrong_end":end=[end[0],end[1]+90]
        elif mode=="wrong_order":pass
        action=dict(type="drag",start_point=start,end_point=end,
            coordinate_space="source_image_pixels",
            ordered=not c["ordered"] if mode=="wrong_order" else c["ordered"])
        cases.append(dict(id=c["id"],image_sha256=c["image_sha256"],
                          raw_output=json.dumps(action,sort_keys=True)))
    return dict(schema=SCHEMA,source_sha256=ORIGINAL_SHA,
        origin="synthetic_unit_fixture",cases=cases)

class H58RealEdgeDomainTests(unittest.TestCase):
    def test_24_actual_browser_original_multidomain_and_language(self):
        r=audit(ROOT)
        self.assertEqual(r["actual_original_edge_screens"],24)
        self.assertEqual(r["domains"],{"sheet":8,"slide_resize":8,"text_highlight":8})
        self.assertEqual((r["ar"],r["en"]),(12,12))
        self.assertEqual((r["ordered_true"],r["ordered_false"]),(8,16))
        self.assertEqual(r["original_real_browser_correct_drags"],24)
        self.assertEqual(r["original_real_browser_reverse_unordered_accepted"],16)
        self.assertEqual(r["original_real_browser_reverse_ordered_rejected"],8)
        self.assertEqual(r["original_real_browser_wrong_target_rejected"],24)
        self.assertFalse(r["training_eligible"])
        self.assertFalse(r["blind_test"])
        self.assertFalse(r["model_neural_inference_performed"])

    def test_author_oracle_fixture_is_not_genuine_inference(self):
        a=score(ROOT,fixture())
        self.assertEqual((a["total_valid"],a["ordered_semantics_valid"],a["geometrically_correct"]),(24,24,24))
        self.assertEqual(a["origin"],"synthetic_unit_fixture")
        self.assertFalse(a["actual_neural_inference_provenance_independently_verified"])
        self.assertFalse(a["real_browser_model_task_success_verified"])
        self.assertFalse(a["benchmark_comparison_qualified"])

    def test_unordered_reverse_accepted_but_directional_resize_not(self):
        a=score(ROOT,fixture("reverse"))
        self.assertEqual(a["geometrically_correct"],16)
        self.assertEqual(a["reverse_accepted"],16)
        self.assertEqual(a["by_domain_language"]["slide_resize/ar"]["geometrically_correct"],0)
        self.assertEqual(a["by_domain_language"]["sheet/ar"]["geometrically_correct"],4)
        self.assertEqual(a["by_domain_language"]["text_highlight/en"]["geometrically_correct"],4)

    def test_wrong_end_and_wrong_order_always_fail(self):
        a=score(ROOT,fixture("wrong_end"))
        self.assertEqual(a["geometrically_correct"],0)
        a=score(ROOT,fixture("wrong_order"))
        self.assertEqual(a["total_valid"],24)
        self.assertEqual(a["ordered_semantics_valid"],0)
        self.assertEqual(a["geometrically_correct"],0)

    def test_forged_duplicate_or_mutated_image_hash_fails_closed(self):
        x=fixture();x["cases"][0]=copy.deepcopy(x["cases"][1])
        with self.assertRaises(ValueError):score(ROOT,x)
        x=fixture();x["cases"][0]["image_sha256"]="0"*64
        with self.assertRaises(ValueError):score(ROOT,x)
        x=fixture();x["cases"].pop()
        with self.assertRaises(ValueError):score(ROOT,x)

    def test_invalid_native_drag_text_is_not_success(self):
        raws=[
            'not a JSON object',
            '{"type":"drag","type":"drag","start_point":[1,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":true}',
            '{"type":"drag","start_point":[1,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":NaN}',
            '{"type":"click","start_point":[1,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":false}',
            '{"type":"drag","start_point":[true,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":false}',
            '{"type":"drag","start_point":[1.5,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":false}',
            '{"type":"drag","start_point":[1,2],"end_point":[5,650],"coordinate_space":"source_image_pixels","ordered":false}',
            '{"type":"drag","start_point":[1,2],"end_point":[5,6],"coordinate_space":"source_image_pixels","ordered":false,"extra":true}'
        ]
        self.assertTrue(all(parse_action(x) is None for x in raws))

    def test_reject_missing_pixel_evidence_not_just_receipt_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"html").mkdir();(root/"png").mkdir()
            for f in ("source.json","plans.json","capture.json"):
                shutil.copyfile(ROOT/f,root/f)
            with self.assertRaises((ValueError,FileNotFoundError)):
                audit(root)
if __name__=="__main__":unittest.main()
