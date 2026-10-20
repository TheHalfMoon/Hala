"""H46 original 8 DEV first-frame Qwen-model pilot protocol tests.

All unittest outputs are *fake injected* Qwen responses for checking the
external evaluator. They are not measured model results.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from h34.runner import MODEL,REVISION
from h46.experiment import cases,sha,PROTOCOL,PROMPT_SHA,INPUT_SHA,ORACLE_SHA,CAPTURE_SHA,replay,admit,source_gate
from tools.build_h46_kaggle import generate,KAGGLE_ID


class H46ModelPilotTests(unittest.TestCase):
    def test_preregistered_first_frame_dev_8_new_screenshots_only(self):
        x=cases()
        self.assertEqual(len(x),8)
        self.assertEqual(sum(c["language"]=="en" for c in x),4)
        self.assertEqual(sum(c["language"]=="ar" for c in x),4)
        self.assertEqual({c["family"] for c in x},{"harbor_inventory","summit_scheduling"})
        self.assertTrue(all(c["split"]=="dev" and c["model_trained"] is False for c in x))
        self.assertFalse(any(set(c)&{"controls","target_bounds","target_point",
                                     "expected_ref","expected_state"} for c in x))

    def test_kaggle_and_source_admission_never_operate_on_windows(self):
        with self.assertRaises(RuntimeError):
            admit()
        with self.assertRaises(ValueError):
            source_gate("not-an-exact-head")

    def test_external_replay_correct_fake_points_and_strict_denominators(self):
        items=cases()
        source=json.loads(Path("h44/assets/evaluator_only.json").read_text("utf-8"))["cases"]
        source=[e for e in source if e["split"]=="dev"]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            meta={"protocol":PROTOCOL,"model":MODEL,"revision":REVISION,
                  "input_sha256":INPUT_SHA,"oracle_sha256_not_read":ORACLE_SHA,
                  "capture_sha256":CAPTURE_SHA,"prompt_sha256":PROMPT_SHA,
                  "native_actions":False,"model_qualified":False,
                  "source_commit":"a"*40}
            (root/"run_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
            rows=[]
            for item,expected in zip(items,source):
                rect=next(x["bounds"] for x in expected["controls"] if x["ref"]=="row_target")
                px=(rect[0]+rect[2]/2)/960
                py=(rect[1]+rect[3]/2)/540
                rows.append({"case_id":item["id"],"split":"dev",
                   "language":item["language"],"image_sha256":item["image_sha256"],
                   "prompt_sha256":PROMPT_SHA,"raw":json.dumps({"point":[px,py]}),"error":None})
            raw="".join(json.dumps(v,sort_keys=True)+"\n" for v in rows).encode()
            (root/"raw_attempts.jsonl").write_bytes(raw)
            report=replay(root,sha(raw))
            self.assertEqual(report["denominator"],8)
            self.assertEqual(report["totals"]["target_hit"],8)
            self.assertEqual(report["per_language_denominator"],{"en":4,"ar":4})
            self.assertFalse(report["model_qualified"])
            self.assertFalse(report["independent_blind_test"])
            self.assertFalse(report["competitor_win"])
            self.assertFalse(report["native_actions"])
            with self.assertRaises(ValueError):replay(root,"f"*64)
            (root/"raw_attempts.jsonl").write_bytes(raw[:-3])
            with self.assertRaises(ValueError):replay(root,sha(raw[:-3]))
            (root/"raw_attempts.jsonl").write_bytes(raw)
            rows[0]["case_id"]=rows[1]["case_id"]
            altered="".join(json.dumps(v,sort_keys=True)+"\n" for v in rows).encode()
            (root/"raw_attempts.jsonl").write_bytes(altered)
            with self.assertRaises(ValueError):replay(root,sha(altered))
            # Even a valid external SHA must not override false model qualification.
            (root/"raw_attempts.jsonl").write_bytes(raw)
            meta["model_qualified"]=True
            (root/"run_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
            with self.assertRaises(ValueError):replay(root,sha(raw))

    def test_private_create_only_notebook_explicit_optin_no_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"new"
            with patch("tools.build_h46_kaggle.inventory",return_value={"h46/experiment.py":"a"*64}):
                result=generate(Path("."),"2"*40,out,research_allowed=False)
            self.assertFalse(result["submitted"])
            n=json.loads((out/"hala-h46-original-dev-qwen3vl.ipynb").read_text("utf-8"))
            m=json.loads((out/"kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(m["id"],KAGGLE_ID)
            self.assertTrue(m["is_private"] and m["enable_gpu"] and m["enable_internet"])
            self.assertFalse(m["enable_tpu"])
            sources=["".join(c["source"]) for c in n["cells"] if c["cell_type"]=="code"]
            self.assertEqual(len(sources),2)
            for code in sources:compile(code,"<H46 notebook>","exec")
            self.assertIn("research opt-in required",sources[0])
            self.assertIn("nvidia-smi",sources[0])
            self.assertNotIn("evaluator_only.json",sources[0]+sources[1])
            self.assertFalse(n["metadata"]["model_qualified"])
            with self.assertRaises(FileExistsError):
                with patch("tools.build_h46_kaggle.inventory",return_value={}):
                    generate(Path("."),"2"*40,out,research_allowed=True)
            permitted=Path(tmp)/"permitted"
            with patch("tools.build_h46_kaggle.inventory",return_value={}):
                generate(Path("."),"2"*40,permitted,research_allowed=True)
            nb=json.loads((permitted/"hala-h46-original-dev-qwen3vl.ipynb").read_text("utf-8"))
            self.assertTrue(nb["metadata"]["research_optin"])
            self.assertFalse(nb["metadata"]["blind_benchmark"])


if __name__=="__main__":unittest.main()
