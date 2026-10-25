"""H124 frozen original full drag START AND END scorer, CPU only."""
import hashlib,unittest
from unittest.mock import patch
from h124.objective import completion
from h124.heldout_gate import score
from h124.schedule import Schedule
from h124.training_plan import TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA
from h123.schedule import FAMILIES


def fixtures(successful=None):
    successful=set(FAMILIES) if successful is None else successful
    cases=[];observations=[];roles=[]
    for group,family in enumerate(FAMILIES):
        for j in range(32):
            i=group*32+j
            sx=70+(i*7)%650
            sy=100+(i*11)%700
            ex=sx+100
            ey=sy+70
            ok=(family in successful and j<16)
            observed_end=[ex,ey] if ok else [min(950,ex+190),ey]
            token=hashlib.sha256(f"h124-original-image-{i}".encode()).hexdigest()
            prompt="Move the selected interface item to the requested place"
            case={"id":f"h124-original-{i}","split":"train",
                  "source_split":"TRAIN_ONLY_HELDOUT",
                  "family":family,"language":"en","image_sha256":token,
                  "prompt":prompt,"image_width":1000,"image_height":600,
                  "start_bbox":[sx-2,round(sy*.6)-2,sx+2,round(sy*.6)+2],
                  "end_bbox":[ex-2,round(ey*.6)-2,ex+2,round(ey*.6)+2]}
            raw=completion([sx,sy],observed_end)
            case_obs={"id":case["id"],"image_sha256":token,
                      "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                      "raw_output":raw,
                      "output_sha256":hashlib.sha256(raw.encode()).hexdigest()}
            cases.append(case);observations.append(case_obs)
            roles.append({"original_source_id":case["id"],
                          "family":family,"image_sha256":token,"prompt":prompt})
    schedule=Schedule(tuple({"original_source_id":f"train-{i}"}
                            for i in range(896)),
                      tuple(roles),tuple(range(896)),
                      tuple(range(896,1024)),
                      {"train_private_sha256":TRAIN_SHA,
                       "heldout_private_sha256":HELDOUT_SHA,
                       "schedule_sha256":SCHEDULE_SHA})
    return cases,observations,schedule


class H124FullDragGateTests(unittest.TestCase):
    def test_all_four_domains_true_START_and_END_pass(self):
        c,m,s=fixtures()
        with patch("h124.heldout_gate.get_plan",return_value=None):
            out=score(c,m,s)
        self.assertTrue(out["heldout_gate_passed"])
        self.assertEqual(out["native_valid_joint_geometry"],128)
        self.assertEqual(out["strict_original_full_drag_hits"],64)
        self.assertEqual(len(out["full_drag_hits_per_family"]),4)

    def test_one_domain_only_end_matches_does_not_qualify(self):
        c,m,s=fixtures(successful={FAMILIES[0]})
        with patch("h124.heldout_gate.get_plan",return_value=None):
            out=score(c,m,s)
        self.assertFalse(out["heldout_gate_passed"])
        self.assertEqual(out["strict_original_full_drag_hits"],16)

    def test_original_sha_and_evaluation_source_fail_closed(self):
        c,m,s=fixtures()
        m[0]["raw_output"]="tampered"
        with patch("h124.heldout_gate.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(c,m,s)
        c,m,s=fixtures()
        c[0]["split"]="eval"
        with patch("h124.heldout_gate.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(c,m,s)
