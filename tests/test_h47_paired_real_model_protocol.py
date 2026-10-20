"""H47 offline protocol checks. All responses injected here are FAKE model data."""
from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h47.experiment import PROTOCOL,PROMPTS,PROMPT_SHA,plan,plan_sha,replay,sha,admit,source_inventory,MODEL,REVISION,INPUT_SHA,ORACLE_SHA,CAPTURE_SHA
from h46.experiment import cases
from tools.build_h47_kaggle import generate,KAGGLE_ID


class H47PairedTests(unittest.TestCase):
    def test_exact_16_counterbalanced_original_bilingual_public_dev_pairs(self):
        p=plan(); self.assertEqual(len(p),16)
        self.assertEqual(len({x["case_id"] for x in p}),8)
        for c in cases():
            arms=[x["arm"] for x in p if x["case_id"]==c["id"]]
            self.assertEqual(set(arms),set(PROMPTS))
            self.assertEqual(len(arms),2)
        first=Counter(row["arm"] for row in p[::2])
        self.assertEqual(first,{"numeric_example":4,"no_numeric_example":4})
        self.assertEqual(len(plan_sha()),64)
        self.assertIn('[0.25,0.75]',PROMPTS["numeric_example"])
        self.assertNotIn('[0.25,0.75]',PROMPTS["no_numeric_example"])
        self.assertNotEqual(PROMPT_SHA["numeric_example"],PROMPT_SHA["no_numeric_example"])

    def test_worker_requires_free_batch_and_clean_signed_source(self):
        with self.assertRaises(RuntimeError):admit()
        with self.assertRaises(ValueError):source_inventory("not_sha")

    def test_fake_paired_external_raw_replay_and_wrong_arm_rejected(self):
        src=[r for r in json.loads(Path("h44/assets/evaluator_only.json").read_text(encoding="utf-8"))["cases"] if r["split"]=="dev"]
        ids={x["id"]:x for x in src}
        rows=[]
        for i,row in enumerate(plan()):
            controls=ids[row["case_id"]]["controls"]
            chosen=next(c["bounds"] for c in controls if c["ref"]==("row_target" if row["arm"]=="no_numeric_example" else "row_decoy"))
            pt=[(chosen[0]+chosen[2]/2)/960,(chosen[1]+chosen[3]/2)/540]
            rows.append({**row,"raw":json.dumps({"point":pt}),"error":None})
        meta={"protocol":PROTOCOL,"plan_sha256":plan_sha(),
              "model":MODEL,"revision":REVISION,
              "prompt_hashes":PROMPT_SHA,"input_sha256":INPUT_SHA,
              "evaluator_sha256_not_read":ORACLE_SHA,"capture_sha256":CAPTURE_SHA,
              "model_qualified":False,"native_actions":False,"blind_test":False}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"run_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
            raw=("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows)).encode()
            (root/"raw_attempts.jsonl").write_bytes(raw)
            report=replay(root,sha(raw))
            self.assertEqual(report["attempted"],16)
            self.assertEqual(report["denominator_per_arm"],8)
            self.assertEqual(report["arm_outcomes"]["no_numeric_example"]["target_hit"],8)
            self.assertEqual(report["arm_outcomes"]["numeric_example"]["target_hit"],0)
            self.assertEqual(report["pair_result"]["no_numeric_better"],8)
            self.assertTrue(report["not_independent"])
            self.assertFalse(report["model_qualified"])
            self.assertFalse(report["competitor_win"])
            with self.assertRaises(ValueError):replay(root,"f"*64)
            broken=copy.deepcopy(rows)
            broken[1]["arm"]="numeric_example"
            altered=("".join(json.dumps(r,sort_keys=True)+"\n" for r in broken)).encode()
            (root/"raw_attempts.jsonl").write_bytes(altered)
            with self.assertRaises(ValueError):replay(root,sha(altered))
            (root/"raw_attempts.jsonl").write_bytes(raw)
            meta["model_qualified"]=True
            (root/"run_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
            with self.assertRaises(ValueError):replay(root,sha(raw))

    def test_incomplete_attempts_count_as_missing_failures(self):
        src=plan()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            meta={"protocol":PROTOCOL,"plan_sha256":plan_sha(),"model":MODEL,
                  "revision":REVISION,"prompt_hashes":PROMPT_SHA,
                  "input_sha256":INPUT_SHA,"evaluator_sha256_not_read":ORACLE_SHA,
                  "capture_sha256":CAPTURE_SHA,"model_qualified":False,
                  "native_actions":False,"blind_test":False}
            (root/"run_metadata.json").write_text(json.dumps(meta))
            missing=("".join(json.dumps({**r,"raw":None,"error":"inference_interrupted"})+"\n" for r in src[:4])).encode()
            (root/"raw_attempts.jsonl").write_bytes(missing)
            report=replay(root,sha(missing))
            self.assertFalse(report["complete"])
            self.assertEqual(report["attempted"],4)
            self.assertEqual(report["total_expected_attempts"],16)
            self.assertTrue(report["pair_result"]["incomplete_pairs"]>=6)

    def test_private_source_notebook_builder_is_create_only(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/"payload"
            with patch("tools.build_h47_kaggle.source_inventory",return_value={"h47/experiment.py":"a"*64}):
                result=generate(Path("."),"1"*40,out,research_allowed=True)
            self.assertFalse(result["submitted"])
            self.assertTrue(result["research_allowed"])
            nb=json.loads((out/"hala-h47-paired-original-vs-no-number.ipynb").read_text(encoding="utf-8"))
            m=json.loads((out/"kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(m["id"],KAGGLE_ID)
            self.assertTrue(m["is_private"] and m["enable_gpu"] and m["enable_internet"])
            self.assertFalse(m["enable_tpu"])
            self.assertTrue(nb["metadata"]["research_optin"])
            self.assertEqual(nb["metadata"]["attempts"],16)
            self.assertFalse(nb["metadata"]["model_qualified"])
            code="\n".join("".join(x["source"]) for x in nb["cells"] if x["cell_type"]=="code")
            self.assertNotIn("evaluator_only.json",code)
            for part in nb["cells"]:
                if part["cell_type"]=="code":compile("".join(part["source"]),"<H47>", "exec")
            with patch("tools.build_h47_kaggle.source_inventory",return_value={}):
                with self.assertRaises(FileExistsError):
                    generate(Path("."),"1"*40,out,research_allowed=False)

if __name__=="__main__":unittest.main()
