"""H42 CPU-only, no fake leaderboard results; all examples ephemeral synthetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.h42_paired_audit import analyze, read_receipt, wilson, mcnemar_exact


def illustrative_receipt():
    rows=[]
    outcomes=[
        (True,False,"en"),(True,False,"en"),(True,True,"ar"),
        (False,True,"ar"),(True,False,"ar"),(False,False,"en"),
        (True,False,"ar"),(True,True,"en")
    ]
    for i,(a,b,lang) in enumerate(outcomes):
        rows.append({"task_id":f"synthetic-case-{i}","family":"synthetic-unseen-2",
                     "language":lang,"platform":"linux",
                     "candidate_success":a,"baseline_success":b,
                     "candidate_unsafe":False,"baseline_unsafe":False,
                     "candidate_latency_ms":100+i,"baseline_latency_ms":120+i,
                     "candidate_cost_usd":0.0,"baseline_cost_usd":0.0})
    return {"schema":"hala.h42.paired-task-receipt.v1",
            "benchmark":"synthetic-h42-fixture","benchmark_release":"v0",
            "harness_sha256":"a"*64,"task_manifest_sha256":"b"*64,
            "environment_sha256":"9"*64,"max_actions_per_task":50,
            "max_wall_seconds_per_task":120,"max_retries_per_task":0,
            "split":"sealed_test","independent_oracle_claim":True,
            "training_families":["different-train-family"],
            "candidate":{"model_id":"fake-hala","source_commit":"1"*40,
                         "weights_sha256":"c"*64,"runtime_sha256":"d"*64,
                         "agent_surface":"gui_only"},
            "baseline":{"model_id":"fake-other","source_commit":"2"*40,
                        "weights_sha256":"e"*64,"runtime_sha256":"f"*64,
                        "agent_surface":"gui_only"},
            "rows":rows}


class H42PairedAuditTests(unittest.TestCase):
    def test_paired_success_cost_strata_but_never_best_claim(self):
        report=analyze(illustrative_receipt())
        self.assertEqual(report["overall"]["candidate_hits"],6)
        self.assertEqual(report["overall"]["baseline_hits"],3)
        self.assertEqual(report["overall"]["paired_win"],4)
        self.assertEqual(report["overall"]["paired_loss"],1)
        self.assertEqual(report["overall"]["mcnemar_two_sided_p"],.375)
        self.assertEqual(report["language"]["en"]["n"],4)
        self.assertEqual(report["language"]["ar"]["n"],4)
        self.assertEqual(report["declared_common_max_actions"],50)
        self.assertEqual(report["declared_common_max_seconds"],120)
        self.assertEqual(report["declared_common_max_retries"],0)
        self.assertTrue(report["descriptive_only"])
        self.assertFalse(report["claim_top_rank_qualified"])
        self.assertFalse(report["independent_oracle_verified"])
        self.assertFalse(report["blind_holdout_independently_verified"])

    def test_completely_one_sided_p_value_is_still_not_independent_proof(self):
        x=illustrative_receipt()
        for row in x["rows"]:
            row["candidate_success"]=True
            row["baseline_success"]=False
        result=analyze(x)
        self.assertEqual(result["overall"]["mcnemar_two_sided_p"],.0078125)
        self.assertFalse(result["claim_top_rank_qualified"])
        self.assertEqual(wilson(0,8)[0],0.0)
        self.assertEqual(wilson(8,8)[1],1.0)
        self.assertEqual(mcnemar_exact(0,0),1.0)

    def test_missing_duplicates_and_weak_evidence_rejected(self):
        initial=illustrative_receipt()
        mutations=[
            ("duplicate",lambda v:v["rows"][1].update(task_id=v["rows"][0]["task_id"])),
            ("missing",lambda v:v["rows"][0].pop("candidate_success")),
            ("extra",lambda v:v["rows"][0].update(bypass=True)),
            ("non_bool",lambda v:v["rows"][0].update(candidate_success=1)),
            ("bad_cost",lambda v:v["rows"][0].update(candidate_cost_usd=float("nan"))),
            ("inf_lat",lambda v:v["rows"][0].update(candidate_latency_ms=float("inf"))),
            ("negative_latency",lambda v:v["rows"][0].update(candidate_latency_ms=-1)),
            ("leak_family",lambda v:v["training_families"].append(v["rows"][0]["family"])),
            ("wrong_language",lambda v:v["rows"][0].update(language="mixed")),
            ("wrong_platform",lambda v:v["rows"][0].update(platform="unvalidated")),
            ("bad_digest",lambda v:v.update(task_manifest_sha256="none")),
            ("bad_environment",lambda v:v.update(environment_sha256="fake")),
            ("missing_action_limit",lambda v:v.pop("max_actions_per_task")),
            ("zero_action_limit",lambda v:v.update(max_actions_per_task=0)),
            ("bool_action_limit",lambda v:v.update(max_actions_per_task=True)),
            ("unbounded_seconds",lambda v:v.update(max_wall_seconds_per_task=20000)),
            ("negative_retry",lambda v:v.update(max_retries_per_task=-1)),
            ("same_model",lambda v:v["baseline"].update(model_id="fake-hala")),
            ("surface_mismatch",lambda v:v["baseline"].update(agent_surface="api_tools")),
            ("not_list",lambda v:v.update(rows={})),
            ("not_sealed_or_dev",lambda v:v.update(split="guessed_test")),
            ("pretend_winner",lambda v:v.update(leaderboard_qualified=True)),
        ]
        for name,mutate in mutations:
            with self.subTest(name=name):
                item=copy.deepcopy(initial)
                mutate(item)
                with self.assertRaises(ValueError):
                    analyze(item)

    def test_unsafe_event_stays_blocked_even_if_accuracy_high(self):
        item=illustrative_receipt()
        item["rows"][0]["candidate_unsafe"]=True
        verdict=analyze(item)
        self.assertTrue(verdict["unsafe_event_blocker"])
        self.assertFalse(verdict["claim_top_rank_qualified"])

    def test_unbalanced_language_never_invented(self):
        item=illustrative_receipt()
        for row in item["rows"]:row["language"]="en"
        result=analyze(item)
        self.assertEqual(result["language"]["ar"]["n"],0)
        self.assertEqual(result["language"]["ar"]["inference"],"INSUFFICIENT_STRATUM")

    def test_nonfinite_and_duplicate_json_keys_denied_even_from_disk(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"receipt.json"
            path.write_text('{"schema":"x","schema":"y"}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                read_receipt(path)
            path.write_text('{"x":NaN}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Nonfinite"):
                read_receipt(path)
            path.write_text(json.dumps(illustrative_receipt()),encoding="utf-8")
            self.assertEqual(analyze(read_receipt(path))["n"],8)


if __name__=="__main__":
    unittest.main()
