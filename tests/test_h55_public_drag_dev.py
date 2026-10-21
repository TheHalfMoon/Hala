"""H55 actual Edge source/capture audit and public DEV scorer CPU tests only."""
import copy,json,tempfile,unittest,shutil
from pathlib import Path
from h55.audit import audit
from h55.score import parse_action,score,SCHEMA
ROOT=Path(__file__).resolve().parents[1]/"h55/assets"
def fixture(mode="oracle"):
    d=json.loads((ROOT/"capture.json").read_text("utf8"))
    cases=[]
    for c in d["cases"]:
        action=dict(type="drag",start_point=c["start_point"],
             end_point=c["end_point"],coordinate_space="source_image_pixels",ordered=True)
        if mode=="wrong":action["end_point"]=[0,0]
        cases.append(dict(id=c["id"],image_sha256=c["image_sha256"],raw_output=json.dumps(action)))
    return dict(schema=SCHEMA,source_sha256=d["source_sha256"],
                origin="synthetic_unit_fixture",cases=cases)

class RealEdgePublicDevTests(unittest.TestCase):
    def test_real_browser_bilingual_screenshot_and_negatives(self):
        r=audit(ROOT)
        self.assertEqual(r["actual_first_party_public_dev"],24)
        self.assertEqual(r["language"],{"ar":12,"en":12})
        self.assertEqual(r["genuine_browser_accepted_drags"],24)
        self.assertEqual(r["real_wrong_end_rejected"],24)
        self.assertEqual(r["real_wrong_origin_rejected"],24)
        self.assertEqual(r["training_screens"],0)
        self.assertFalse(r["model_inference_executed"])

    def test_public_dev_not_blind_or_training_or_competitor(self):
        s=json.loads((ROOT/"source.json").read_text("utf8"))
        self.assertFalse(s["rights"]["external_images"])
        self.assertFalse(s["rights"]["live_services"])
        self.assertFalse(s["rights"]["train_eligible"])
        self.assertFalse(s["rights"]["official_benchmark"])
        self.assertFalse(s["blind_test"])
        self.assertEqual(len(list((ROOT/"png").glob("*.png"))),24)

    def test_synthetic_oracle_geometry_cannot_prove_model(self):
        r=score(ROOT,fixture())
        self.assertEqual(r["total_valid"],24)
        self.assertEqual(r["both_endpoints_hit"],24)
        self.assertEqual(r["origin"],"synthetic_unit_fixture")
        self.assertFalse(r["actual_browser_model_task_success_verified"])
        self.assertFalse(r["official_benchmark_comparison_qualified"])

    def test_wrong_points_not_passed_even_if_json_valid(self):
        r=score(ROOT,fixture("wrong"))
        self.assertEqual(r["total_valid"],24)
        self.assertEqual(r["both_endpoints_hit"],0)

    def test_duplicate_missing_or_wrong_image_hash_rejected(self):
        x=fixture();x["cases"][1]=copy.deepcopy(x["cases"][0])
        with self.assertRaises(ValueError):score(ROOT,x)
        x=fixture();x["cases"][0]["image_sha256"]="0"*64
        with self.assertRaises(ValueError):score(ROOT,x)
        x=fixture();x["cases"].pop()
        with self.assertRaises(ValueError):score(ROOT,x)

    def test_invalid_native_drag_json_not_mistaken_for_success(self):
        values=[
         '{"type":"click","start_point":[1,2],"end_point":[3,4],"coordinate_space":"source_image_pixels","ordered":true}',
         '{"type":"drag","start_point":[1,2],"end_point":[3,4],"coordinate_space":"source_image_pixels","ordered":true,"secret":true}',
         '{"type":"drag","start_point":[1,2],"end_point":[3,4],"coordinate_space":"source_image_pixels","ordered":true,"ordered":true}',
         '{"type":"drag","start_point":[1.5,2],"end_point":[3,4],"coordinate_space":"source_image_pixels","ordered":true}',
         '{"type":"drag","start_point":[1,2],"end_point":[3000,4],"coordinate_space":"source_image_pixels","ordered":true}',
         '{"type":"drag","start_point":[1,2],"end_point":[3,4],"coordinate_space":"source_image_pixels","ordered":NaN}',
         'not JSON'
        ]
        self.assertTrue(all(parse_action(s) is None for s in values))
        x=fixture();x["cases"][0]["raw_output"]=values[0]
        r=score(ROOT,x)
        self.assertEqual(r["total_valid"],23)
        self.assertEqual(r["both_endpoints_hit"],23)

    def test_missing_screenshot_source_fails_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"png").mkdir();(root/"html").mkdir()
            for n in ("plans.json","source.json","capture.json"):shutil.copyfile(ROOT/n,root/n)
            with self.assertRaises((FileNotFoundError,ValueError)):audit(root)

if __name__=="__main__":
    unittest.main()
