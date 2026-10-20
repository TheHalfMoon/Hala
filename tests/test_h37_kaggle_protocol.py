"""H37 Kaggle replay, source admission, and zero-cost-only notebook smoke."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h37.audit import audit
from h37.runner import admit,inputs,MODEL,REVISION,PROMPT_SHA,PROTOCOL
from tools.build_h37_kaggle import FILES,generate

ASSETS=Path("h37/assets")


def fixture(root:Path,rows:int=64) -> str:
    root.mkdir()
    model=json.loads((ASSETS/"model_inputs.json").read_text("utf-8"))
    oracle=json.loads((ASSETS/"evaluator_only.json").read_text("utf-8"))
    body=[]
    for item,case in list(zip(model,oracle))[:rows]:
        target=next(c for c in case["controls"] if c["element_ref"]==case["expected_control_id"])
        x,y,w,h=target["bounds"]
        point={"point":[(x+w/2)/960,(y+h/2)/540]}
        body.append({"case_id":item["id"],"split":item["split"],
                     "language":item["language"],"image_sha256":case["image_sha256"],
                     "prompt_sha256":PROMPT_SHA,"raw":json.dumps(point),"error":None})
    journal=("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in body)).encode("utf-8")
    (root/"raw_attempts.jsonl").write_bytes(journal)
    meta={"protocol":PROTOCOL,"model":MODEL,"revision":REVISION,
          "prompt_sha256":PROMPT_SHA,"source_commit":"c"*40,"actions_executed":False}
    (root/"run_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
    return hashlib.sha256(journal).hexdigest()


class H37KaggleTests(unittest.TestCase):
    def test_baseline_input_admission(self):
        rows,manifest,root=inputs()
        self.assertEqual(len(rows),64)
        self.assertEqual(manifest["dev"],32)
        self.assertEqual(manifest["test"],32)
        self.assertEqual(len({r["id"] for r in rows}),64)
        self.assertEqual(len(FILES),len(set(FILES)))
        self.assertGreaterEqual(len(FILES),152)
        self.assertTrue(all(Path(f).is_file() for f in FILES))

    def test_oracle_fixture_replay_is_not_model_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"output"
            sha=fixture(p)
            result=audit(p,sha)
            self.assertTrue(result["complete"])
            self.assertEqual(result["attempts"],64)
            self.assertEqual(result["totals"]["target_hit"],64)
            self.assertEqual(result["split"]["dev"]["target_hit"],32)
            self.assertEqual(result["split"]["test"]["target_hit"],32)
            self.assertFalse(result["model_qualified"])
            self.assertFalse(result["blind_test_qualified"])
            self.assertFalse(result["external_app_captured"])

    def test_partial_attempts_keep_total_denominator_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"output"
            sha=fixture(p,rows=3)
            result=audit(p,sha)
            self.assertEqual(result["attempts"],3)
            self.assertFalse(result["complete"])
            self.assertEqual(result["unconditional_denominator"],64)
            self.assertEqual(result["totals"]["error"],61)
            records=[json.loads(x) for x in (p/"raw_attempts.jsonl").read_text().splitlines()]
            records[0]["case_id"]=records[1]["case_id"]
            bad="".join(json.dumps(x)+"\n" for x in records).encode()
            (p/"raw_attempts.jsonl").write_bytes(bad)
            with self.assertRaises(ValueError):
                audit(p,hashlib.sha256(bad).hexdigest())
            with self.assertRaises(ValueError):
                audit(p,sha)

    def test_duplicate_json_point_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"output"
            sha=fixture(p,rows=64)
            rows=[json.loads(x) for x in (p/"raw_attempts.jsonl").read_text().splitlines()]
            rows[0]["raw"]='{"point":[0.4,0.5],"point":[0.6,0.5]}'
            journal=("".join(json.dumps(x)+"\n" for x in rows)).encode("utf-8")
            (p/"raw_attempts.jsonl").write_bytes(journal)
            result=audit(p,hashlib.sha256(journal).hexdigest())
            self.assertEqual(result["totals"]["parse_valid"],63)
            self.assertEqual(result["totals"]["target_hit"],63)

    def test_non_kaggle_and_no_explicit_optin_rejected(self):
        with patch("platform.system",return_value="Windows"):
            with self.assertRaises(RuntimeError):
                admit()

    def test_private_notebook_not_submitted_and_no_paid_compute(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/"new_built_bundle"
            fake={name:"a"*64 for name in FILES}
            with patch("tools.build_h37_kaggle.inventory",return_value=fake):
                result=generate(Path(t),"b"*40,dest,research_allowed=False)
            self.assertFalse(result["submitted"])
            self.assertFalse(result["research_allowed"])
            meta=json.loads((dest/"kernel-metadata.json").read_text("utf-8"))
            nb=json.loads((dest/"hala-h37-browser-grounding-dev.ipynb").read_text("utf-8"))
            self.assertTrue(meta["is_private"] and meta["enable_gpu"] and not meta["enable_tpu"])
            self.assertFalse(nb["metadata"]["hala_research_opt_in"])
            self.assertFalse(nb["metadata"]["hala_model_qualified"])
            self.assertFalse(nb["metadata"]["hala_blind_test"])
            for cell in nb["cells"]:
                if cell["cell_type"]=="code":
                    compile("".join(cell["source"]),"<H37 notebook>","exec")
            with self.assertRaises(FileExistsError):
                with patch("tools.build_h37_kaggle.inventory",return_value=fake):
                    generate(Path(t),"b"*40,dest,research_allowed=True)


if __name__=="__main__":
    unittest.main()
