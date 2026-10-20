"""Offline safety/intent checks for prospective paired H36 Kaggle DEV."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h34.corpus import build_cases, case_sha256
from h36.audit import audit
from h36.protocol import (ARMS, MODEL, NUMERIC_EXAMPLE, NO_NUMERIC_EXAMPLE,
                          PROTOCOL, PROMPTS, REVISION, plan, plan_sha, prompt_sha)
from h36.runner import admit, source_identity
from tools.build_h36_kaggle import FILES, generate

H34_V2 = Path("reviews/h34-kaggle-run-20261020-v2")


def fixture(output: Path, *, rows: int = 64) -> str:
    """Synthetic responses and copied genuine H34 PNGs; NOT model results."""
    cases = build_cases()
    output.mkdir(parents=True)
    for file in ("image_manifest.json", "cases.sha256"):
        shutil.copyfile(H34_V2 / file, output / file)
    for case in cases:
        src = H34_V2 / (case["id"] + ".png")
        shutil.copyfile(src, output / src.name)
    (output / "plan.sha256").write_bytes((plan_sha() + "\n").encode("ascii"))
    imgs = json.loads((output / "image_manifest.json").read_text(encoding="utf-8"))
    image_sha = {x["case_id"]: x["sha256"] for x in imgs}
    lookup = {c["id"]: c for c in cases}
    records = []
    for t in plan()[:rows]:
        case = lookup[t["case_id"]]
        target = next(c for c in case["controls"] if c["is_expected"])
        x, y, w, h = target["bounds"]
        raw = json.dumps({"point": [(x+w/2)/case["width"], (y+h/2)/case["height"]]})
        records.append({"case_id": t["case_id"], "language": case["language"],
                        "arm": t["arm"], "image_sha256": image_sha[t["case_id"]],
                        "prompt_sha256": prompt_sha(t["arm"]), "raw": raw, "error": None})
    payload = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n" for x in records).encode()
    (output/"raw_attempts.jsonl").write_bytes(payload)
    meta = {"protocol": PROTOCOL, "plan_sha256": plan_sha(),
            "source_commit": "a"*40, "model": MODEL, "revision": REVISION,
            "split": "exposed-synthetic-dev", "actions_executed": False}
    (output/"run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return hashlib.sha256(payload).hexdigest()


class H36ProtocolTests(unittest.TestCase):
    def test_actual_exact_H34_prompt_and_no_numeric_example(self):
        from h34.runner import PROMPT
        self.assertEqual(NUMERIC_EXAMPLE, PROMPT)
        self.assertIn('{"point":[0.25,0.75]}', NUMERIC_EXAMPLE)
        self.assertNotIn("0.25,0.75", NO_NUMERIC_EXAMPLE)
        self.assertIn('"abstain":true', NO_NUMERIC_EXAMPLE)
        self.assertIn("normalized center coordinates", NO_NUMERIC_EXAMPLE)
        self.assertEqual(set(PROMPTS), set(ARMS))
        self.assertNotEqual(prompt_sha(ARMS[0]), prompt_sha(ARMS[1]))

    def test_balanced_within_case_order_and_no_oracle(self):
        tasks = plan()
        self.assertEqual(len(tasks), 64)
        self.assertEqual(sum(x["arm"] == ARMS[0] for x in tasks), 32)
        self.assertEqual(sum(x["arm"] == ARMS[1] for x in tasks), 32)
        for i, case in enumerate(build_cases()):
            pair = tasks[2*i:2*i+2]
            self.assertEqual({x["case_id"] for x in pair}, {case["id"]})
            self.assertEqual({x["arm"] for x in pair}, set(ARMS))
            self.assertEqual(pair[0]["arm"], ARMS[i%2])
            self.assertTrue(all("expected_label" not in x for x in pair))
        for lang in ("en", "ar"):
            firsts = [tasks[2*i]["arm"] for i,c in enumerate(build_cases()) if c["language"] == lang]
            self.assertEqual(firsts.count(ARMS[0]), 8)
            self.assertEqual(firsts.count(ARMS[1]), 8)

    def test_success_fixture_is_not_scientific_model_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"evidence"
            digest = fixture(root)
            result = audit(root, digest)
            self.assertTrue(result["complete"])
            self.assertEqual(result["attempted_records"], 64)
            self.assertEqual(result["arms"][ARMS[0]]["target_hit"], 32)
            self.assertEqual(result["arms"][ARMS[1]]["target_hit"], 32)
            self.assertEqual(result["paired"]["both_correct"], 32)
            self.assertFalse(result["model_qualified"])
            self.assertTrue(result["exposed_dev_only"])

    def test_partial_results_and_order_tamper_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"evidence"
            digest = fixture(root, rows=3)
            result = audit(root, digest)
            self.assertFalse(result["complete"])
            self.assertEqual(result["paired"]["both_attempted"], 1)
            self.assertEqual(result["attempted_records"], 3)
            raw_path = root/"raw_attempts.jsonl"
            records = [json.loads(x) for x in raw_path.read_text().splitlines()]
            records[0]["arm"] = records[1]["arm"]
            raw_path.write_text("".join(json.dumps(x)+"\n" for x in records))
            changed_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                audit(root, changed_hash)
            with self.assertRaisesRegex(ValueError, "digest"):
                audit(root, digest)

    def test_image_hash_rebinding_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"evidence"
            digest = fixture(root, rows=2)
            imgs = json.loads((root/"image_manifest.json").read_text())
            first, second = (root/x["path"] for x in imgs[:2])
            a,b = first.read_bytes(),second.read_bytes()
            first.write_bytes(b)
            second.write_bytes(a)
            with self.assertRaises(ValueError):
                audit(root, digest)

    def test_actual_sha_bound_private_kaggle_trial_stays_unqualified(self):
        from h34.protocol import parse_raw, score_attempt
        evidence = Path("reviews/h36-kaggle-run-20261020")
        raw = evidence / "raw_attempts.jsonl"
        sha = "761cceef06f588a67286093580e6f1a1cfe3d3d7605c2b650bf3c9e7c815355e"
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), sha)
        report = audit(evidence, sha)
        self.assertTrue(report["complete"])
        self.assertEqual(report["attempted_records"], 64)
        self.assertEqual(report["source_commit_observed"],
                         "ae3f684580c5cd3fba63442738b945de88ad1085")
        self.assertEqual([report["arms"][a]["target_hit"] for a in ARMS], [8, 6])
        self.assertEqual([report["arms"][a]["parse_valid"] for a in ARMS], [32, 22])
        self.assertEqual([report["arms"][a]["wrong_target"] for a in ARMS], [2, 5])
        self.assertEqual(report["paired"], {
            "denominator": 32, "both_attempted": 32, "both_correct": 3,
            "only_numeric_correct": 5, "only_no_numeric_correct": 3,
            "neither_correct": 21})
        self.assertFalse(report["model_qualified"])
        self.assertFalse(report["independent_test_qualified"])
        canonical = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
        self.assertEqual((evidence/"audit.json").read_bytes(), canonical.encode("utf-8"))
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        h34_rows = {r["case_id"]: r for r in
                    (json.loads(line) for line in
                     (H34_V2/"raw_attempts.jsonl").read_text(encoding="utf-8").splitlines())}
        control = [r for r in rows if r["arm"] == ARMS[0]]
        self.assertEqual(len(control), 32)
        for row in control:
            previous = h34_rows[row["case_id"]]
            self.assertEqual((row["raw"], row["error"]),
                             (previous["raw"], previous["error"]))
        previous_images = {x["case_id"]: x["sha256"] for x in
                           json.loads((H34_V2/"image_manifest.json").read_text("utf-8"))}
        current_images = {x["case_id"]: x["sha256"] for x in
                          json.loads((evidence/"image_manifest.json").read_text("utf-8"))}
        self.assertEqual(current_images, previous_images)
        invalid = [r for r in rows if r["arm"] == ARMS[1] and
                   not score_attempt(next(c for c in build_cases() if c["id"] == r["case_id"]),
                                     r["raw"], r["error"])["parse_valid"]]
        self.assertEqual(len(invalid), 10)
        self.assertTrue(all(r["raw"].count('"point"') == 2 for r in invalid))
        lookup = {c["id"]: c for c in build_cases()}
        examples = []
        for row in rows:
            if row["arm"] != ARMS[0]:
                continue
            case = lookup[row["case_id"]]
            px = parse_raw(row["raw"], case["width"], case["height"])
            if px and abs(px[1] / case["height"] - 0.75) < 0.0001:
                examples.append(score_attempt(case, row["raw"], row["error"]))
        self.assertEqual(len(examples), 8)
        self.assertTrue(all(not x.get("gate_valid_unique", False) for x in examples))

    def test_billing_and_platform_fail_closed(self):
        with patch("platform.system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                admit()
        self.assertIn("h36/runner.py", FILES)
        self.assertIn("h34/frozen/cases.json", FILES)

    def test_notebook_generation_is_only_private_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bundle"
            fake = {"sha256": "b"*64, "files": {f: "a"*64 for f in FILES}}
            with patch("tools.build_h36_kaggle.inventory", return_value=fake):
                result = generate(Path(folder), "c"*40, path, research_allowed=False)
            self.assertFalse(result["submitted"])
            meta = json.loads((path/"kernel-metadata.json").read_text())
            nb = json.loads((path/"hala-h36-paired-prompt-dev.ipynb").read_text())
            self.assertTrue(meta["is_private"])
            self.assertTrue(meta["enable_gpu"])
            self.assertFalse(nb["metadata"]["hala_research_opt_in"])
            self.assertFalse(nb["metadata"]["hala_model_qualified"])
            self.assertEqual(nb["metadata"]["hala_plan_sha256"], plan_sha())
            for item in nb["cells"][1:]:
                compile("".join(item["source"]), "<notebook>", "exec")
            with self.assertRaises(FileExistsError):
                with patch("tools.build_h36_kaggle.inventory", return_value=fake):
                    generate(Path(folder), "c"*40, path, research_allowed=True)


if __name__ == "__main__":
    unittest.main()
