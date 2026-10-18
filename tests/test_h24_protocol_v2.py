"""V2 bounded subgoal protocol; CPU tests only, not proof of model quality."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h24 import data, mind_dev as m
from tests.test_h24_dev_audit import fake_measured, SOURCE
from tools import build_h24_kaggle as notebook
from tools import verify_h24_dev as audit


class H24ProtocolV2Tests(unittest.TestCase):
    def test_all_dev_prompts_explicit_pairs_without_oracle_labels(self):
        for case in m.dev_cases():
            prompt=m.public_prompt_v2(case)
            self.assertIn("locate_verified_target", prompt)
            self.assertIn("refresh_stale_frame", prompt)
            self.assertIn("resolve_destination", prompt)
            self.assertIn("ignore_page_instruction", prompt)
            self.assertIn("reject_external_command", prompt)
            self.assertNotIn('"protected"',prompt)
            self.assertNotIn('"expected"',prompt)
            self.assertNotIn('"split"',prompt)
            self.assertNotIn("training_type_eligible",prompt)
            payload=json.loads(prompt.split("\n",1)[1])
            self.assertEqual(len(payload["observation"]["controls"]),2)
            self.assertEqual(payload["observation"]["locale"],case["observation"]["locale"])

    def test_never_exposes_train_or_test_through_new_prompt(self):
        for split in ("train","test"):
            c=next(x for x in data.dataset()["cases"] if x["split"]==split)
            with self.assertRaises(ValueError):
                m.public_prompt_v2(c)

    def test_allowed_pairs_are_explicit_and_bounded(self):
        self.assertEqual({k:len(v) for k,v in m.V2_PAIR_SUBGOALS.items()},
                         {"visual":1,"observe":2,"clarify":1,"abstain":1})
        for case in m.dev_cases():
            raw=json.dumps(case["expected"])
            self.assertEqual(m.proposal_v2(raw),case["expected"])
            scored=m.score_v2(case,raw)
            self.assertTrue(scored["valid"] and scored["exact_correct"])

    def test_crossed_pairs_and_free_text_fail_closed(self):
        invalid=({"kind":"next","tool":"visual","subgoal":"reject_external_command"},
                 {"kind":"next","tool":"observe","subgoal":"locate_verified_target"},
                 {"kind":"next","tool":"clarify","subgoal":"This is a click command."},
                 {"kind":"next","tool":"shell","subgoal":"ls"},
                 {"kind":"next","tool":"visual","subgoal":"run_powershell"})
        for candidate in invalid:
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                m.proposal_v2(json.dumps(candidate))
        invalid_raw=json.dumps(invalid[0])
        self.assertFalse(m.score_v2(m.dev_cases()[0],invalid_raw)["valid"])

    def test_auditor_checks_separate_v2_identity_and_rescores_every_case(self):
        dataobj=fake_measured()
        dataobj["schema"]="hala.h24.mind_dev_baseline.v2"
        dataobj["prompt_protocol"]="fixed_allowed_pairs_v2"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"v2.json"
            raw=(json.dumps(dataobj,sort_keys=True)+"\n").encode()
            p.write_bytes(raw)
            result=audit.verify(p,hashlib.sha256(raw).hexdigest(),SOURCE,protocol="v2")
            self.assertEqual(result["schema"],"hala.h24.independent_dev_audit.v2")
            self.assertEqual(result["strata"]["all"]["exact_correct"],20)
            with self.assertRaises(ValueError):
                audit.verify(p,hashlib.sha256(raw).hexdigest(),SOURCE,protocol="v1")
            dataobj["prompt_protocol"]="unaudited"
            changed=(json.dumps(dataobj,sort_keys=True)+"\n").encode()
            p.write_bytes(changed)
            with self.assertRaises(ValueError):
                audit.verify(p,hashlib.sha256(changed).hexdigest(),SOURCE,protocol="v2")

    def test_v2_private_notebook_separate_kernel_and_version_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"v2"
            notebook.generate(SOURCE,dest,protocol="v2")
            md=json.loads((dest/"kernel-metadata.json").read_text(encoding="utf-8"))
            nb=json.loads((dest/"h24-mind-dev.ipynb").read_text(encoding="utf-8"))
            self.assertTrue(md["is_private"] and md["enable_gpu"] and md["enable_internet"])
            self.assertEqual(nb["metadata"]["hala_h24_protocol"],"v2")
            self.assertNotEqual(md["id"], notebook.KERNEL)
            src="".join(nb["cells"][-1]["source"])
            self.assertIn('"--protocol","v2"',src)
            self.assertIn(SOURCE,src)
            with self.assertRaises(FileExistsError):
                notebook.generate(SOURCE,dest,protocol="v2")

    def test_invalid_version_never_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"unknown"
            with self.assertRaises(ValueError):
                notebook.generate(SOURCE,dest,protocol="v3")
            self.assertFalse(dest.exists())


if __name__=="__main__":
    unittest.main()
