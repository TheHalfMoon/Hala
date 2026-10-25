"""H125 preregistered native GUI2B original64 START/END strict gate."""
import hashlib,unittest
from unittest.mock import patch
from h125.eval_plan import score,MIN_FULL,MIN_VALID
from h125.source import Schedule,FAMILIES,TOTAL
from h125.model_contracts import MODELS


def fixtures(arm="uground_2b",weak_family=None):
    cases=[];observations=[];roles=[]
    for family_i,family in enumerate(FAMILIES):
        for i in range(16):
            index=family_i*16+i
            sx=70+index*8
            ex=sx+80
            sy=200+index*7
            ey=sy+70
            token=hashlib.sha256(f"h125-new-img-{index}".encode()).hexdigest()
            start=f"({sx},{sy})" if arm=="uground_2b" else f"[{sx/1000},{sy/1000}]"
            new_end=ex if weak_family!=family else min(990,ex+100)
            end=f"({new_end},{ey})" if arm=="uground_2b" else f"[{new_end/1000},{ey/1000}]"
            case={
              "id":f"new-gui-{index}","split":"train",
              "source_split":"TRAIN_ONLY_FRESH_NEVER_TRAINED",
              "family":family,"image_sha256":token,
              "image_width":1000,"image_height":600,
              "prompt_start":"Locate START original", "prompt_end":"Locate END original",
              "start_bbox":[sx-2,round(sy*.6)-2,sx+2,round(sy*.6)+2],
              "end_bbox":[ex-2,round(ey*.6)-2,ex+2,round(ey*.6)+2]}
            out={"arm":arm,"id":case["id"],"image_sha256":token,
              "raw_start":start,"raw_end":end}
            for name in ("start","end"):
                out["prompt_"+name+"_sha256"]=hashlib.sha256(
                    case["prompt_"+name].encode("utf8")).hexdigest()
                out["raw_"+name+"_sha256"]=hashlib.sha256(
                    out["raw_"+name].encode("utf8")).hexdigest()
            cases.append(case);observations.append(out)
            roles.append({"id":case["id"],"family":family,"image_sha256":token})
    schedule=Schedule(tuple(roles),tuple(range(TOTAL)),{})
    return cases,observations,schedule


class H125FrozenNativeScoreTests(unittest.TestCase):
    def test_UGround_all64_valid_genuine_original_full_drags(self):
        a,b,c=fixtures()
        with patch("h125.eval_plan.get_plan",return_value=None):
            result=score(a,b,c,"uground_2b")
        self.assertTrue(result["internal_gate_passed"])
        self.assertEqual(result["strict_original_full_drag_hits"],TOTAL)
        self.assertEqual(result["native_valid_both_points"],TOTAL)

    def test_ShowUI_different_units_same_ground_truth(self):
        a,b,c=fixtures("showui_2b")
        with patch("h125.eval_plan.get_plan",return_value=None):
            result=score(a,b,c,"showui_2b")
        self.assertTrue(result["internal_gate_passed"])
        self.assertEqual(result["strict_original_full_drag_hits"],TOTAL)

    def test_one_domain_failure_does_not_pass_stratified_gate(self):
        a,b,c=fixtures(weak_family=FAMILIES[0])
        with patch("h125.eval_plan.get_plan",return_value=None):
            result=score(a,b,c,"uground_2b")
        self.assertFalse(result["internal_gate_passed"])
        self.assertEqual(result["strict_original_full_drag_hits"],48)

    def test_tampered_source_or_neural_sha_rejected(self):
        a,b,c=fixtures()
        b[0]["raw_start"]="(0,0)"
        with patch("h125.eval_plan.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(a,b,c,"uground_2b")
        a,b,c=fixtures()
        a[0]["split"]="eval"
        with patch("h125.eval_plan.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(a,b,c,"uground_2b")
