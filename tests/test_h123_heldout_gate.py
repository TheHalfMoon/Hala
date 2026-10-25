"""H123 preregistered original TRAIN-only complex-domain heldout scorer."""
import hashlib
import unittest
from unittest.mock import patch
from h123.heldout_gate import score
from h123.objective import completion
from h123.schedule import Schedule,FAMILIES
from h123.training_plan import TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA


def fixture(hit_families=None):
    if hit_families is None:
        hit_families=set(FAMILIES)
    cases=[];outputs=[];role=[]
    for group,family in enumerate(FAMILIES):
        for ordinal in range(32):
            i=group*32+ordinal
            x=60+(i*7)%800
            y=60+(i*13)%800
            source_x=x
            source_y=round(y*.6)
            good=family in hit_families and ordinal<16
            point=[x,y] if good else [min(990,x+50),y]
            img=hashlib.sha256(f"original-img-{i}".encode()).hexdigest()
            task="Move the window to another location in the screen."
            case={"id":f"original-h123-TRAIN-{i}","split":"train",
                  "source_split":"TRAIN_ONLY_HELDOUT",
                  "family":family,"language":"en","image_sha256":img,
                  "prompt":task,"image_width":1000,"image_height":600,
                  "end_bbox":[source_x-2,source_y-2,source_x+2,source_y+2]}
            raw=completion(point)
            obs={"id":case["id"],"image_sha256":img,
                 "prompt_sha256":hashlib.sha256(task.encode()).hexdigest(),
                 "output_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                 "raw_output":raw}
            cases.append(case)
            outputs.append(obs)
            role.append({"original_source_id":case["id"],"image_sha256":img,
                         "prompt":task,"family":family})
    schedule=Schedule(tuple({"original_source_id":f"optimizer-{i}"}
                            for i in range(896)),
        tuple(role),tuple(range(896)),tuple(range(896,1024)),
        {"source_split":"TRAIN_ONLY",
         "train_private_sha256":TRAIN_SHA,
         "heldout_private_sha256":HELDOUT_SHA,
         "schedule_sha256":SCHEDULE_SHA})
    return cases,outputs,schedule


class H123HeldoutTests(unittest.TestCase):
    def test_four_domain_real_source_gate_can_pass(self):
        cases,raw,source=fixture()
        with patch("h123.heldout_gate.get_plan",return_value=None):
            result=score(cases,raw,source)
        self.assertTrue(result["heldout_gate_passed"])
        self.assertEqual(result["strict_END_hits"],64)
        self.assertEqual(result["strict_valid_END_dualgrid_and_point"],128)
        self.assertEqual(len(result["hits_per_family"]),4)

    def test_one_complex_domain_cannot_substitute_other_three(self):
        cases,raw,source=fixture(hit_families={FAMILIES[0]})
        with patch("h123.heldout_gate.get_plan",return_value=None):
            result=score(cases,raw,source)
        self.assertFalse(result["heldout_gate_passed"])
        self.assertEqual(result["strict_END_hits"],16)
        self.assertEqual(sum(x==0 for x in result["hits_per_family"].values()),3)

    def test_raw_neural_SHA_and_source_split_cannot_be_forged(self):
        cases,raw,source=fixture()
        changed=[dict(x) for x in raw]
        changed[0]["raw_output"]="tampered"
        with patch("h123.heldout_gate.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(cases,changed,source)
        cases[0]["split"]="eval"
        with patch("h123.heldout_gate.get_plan",return_value=None):
            with self.assertRaises(ValueError):score(cases,raw,source)


if __name__=="__main__":
    unittest.main()
