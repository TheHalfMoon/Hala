"""H45 real Edge interactive original source/PNG/DOM state evidence tests.

CPU replayer: CI does not need Edge or model/GPU. Negative synthetic browser
states are not independent real-app authorization or external verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from h43.factory import canonical
from h45.audit import audit

ROOT=Path("h45/assets")
CAPTURE_SHA="0d0b4d4f430d3d26eb7b4ce71318523e053bdd2293909a0fe73224a0b9ad2dd0"


class H45InteractiveEdgeTests(unittest.TestCase):
    def test_exact_real_edge_capture_all_frames_and_metrics(self):
        model,oracle,metrics=audit(ROOT)
        self.assertEqual(metrics["source_tasks"],24)
        self.assertEqual(metrics["real_edge_screen_pngs"],144)
        self.assertEqual(metrics["correct_golden_trajectories"],24)
        self.assertEqual(metrics["wrong_target_refused"],24)
        self.assertEqual(metrics["unrelated_mutation_refused"],24)
        self.assertEqual(metrics["model_only_steps"],96)
        self.assertFalse(metrics["native_desktop_actions"])
        self.assertFalse(metrics["independent_benchmark"])
        self.assertEqual(hashlib.sha256((ROOT/"capture.json").read_bytes()).hexdigest(),CAPTURE_SHA)
        for name,obj in (("model_only.json",model),
                         ("evaluator_only.json",oracle),("audit.json",metrics)):
            self.assertEqual((ROOT/name).read_bytes(),canonical(obj))

    def test_images_24_cases_96_model_inputs_and_hidden_oracle_fields(self):
        model,oracle,_=audit(ROOT)
        self.assertEqual(len(model["input_examples"]),96)
        self.assertEqual(len(oracle["reference_results"]),144)
        for m in model["input_examples"]:
            self.assertFalse(set(m)&{"expected_state","expected_success","controls",
                                     "target_bounds","true_ref","golden_actions",
                                     "selected","amount","confirmed","unrelated_changed"})
        self.assertEqual(sum(x["language"]=="ar" for x in model["input_examples"]),48)
        self.assertEqual(sum(x["split"]=="dev" for x in model["input_examples"]),32)
        self.assertFalse(model["independent_blind_test"])
        self.assertFalse(model["model_qualified"])

    def test_correct_action_in_browser_and_wrong_or_unrelated_rejected(self):
        _,oracle,_=audit(ROOT)
        by_id={}
        for row in oracle["reference_results"]:
            by_id.setdefault(row["id"],[]).append(row)
        self.assertEqual(len(by_id),24)
        for name,rows in by_id.items():
            with self.subTest(task=name):
                self.assertEqual(len(rows),6)
                self.assertEqual([x["expected_success"] for x in rows],
                                 [False,False,False,True,False,False])
                self.assertFalse(rows[4]["expected_state"]["confirmed"])
                self.assertTrue(rows[5]["expected_state"]["confirmed"])
                self.assertTrue(rows[5]["expected_state"]["unrelated_changed"])

    def test_changed_png_dom_final_state_or_privileged_claim_fails(self):
        capture=json.loads((ROOT/"capture.json").read_text("utf-8"))
        changes=[
            lambda r:r["records"][0]["states"][0].update(image_sha256="0"*64),
            lambda r:r["records"][0]["states"][3].update(confirmed=False),
            lambda r:r["records"][0]["negatives"][0].update(confirmed=True),
            lambda r:r["records"][0]["negatives"][1].update(unrelated_changed=False),
            lambda r:r["records"][0]["states"][1].update(selected="different"),
            lambda r:r["records"][0]["states"][0]["controls"][0].update(name="replaced"),
            lambda r:r["records"][0]["states"][0]["controls"][1].update(
                bounds=r["records"][0]["states"][0]["controls"][0]["bounds"]),
            lambda r:r.update(actual_model_inference=True),
            lambda r:r.update(external_network_requests=True),
            lambda r:r["records"].pop(),
            lambda r:r["records"][0]["states"].pop(),
        ]
        for i,fn in enumerate(changes):
            with self.subTest(i=i):
                item=copy.deepcopy(capture)
                fn(item)
                with self.assertRaises((ValueError,KeyError,TypeError)):
                    audit(ROOT,override=item)


if __name__=="__main__":
    unittest.main()
