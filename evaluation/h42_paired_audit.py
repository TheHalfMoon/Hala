"""H42 strict descriptive paired task comparator. NO automatic leaderboard claims.

This module validates CPU-readable outcome receipts. It cannot authenticate
independent oracles, sealed tasks, licensing, or execution provenance.
No model, native action, network or GPU is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SCHEMA = "hala.h42.paired-task-receipt.v1"
OUT_SCHEMA = "hala.h42.descriptive-audit.v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,159}\Z")
META_KEYS = {"schema", "benchmark", "benchmark_release", "harness_sha256",
             "task_manifest_sha256", "environment_sha256",
             "max_actions_per_task", "max_wall_seconds_per_task",
             "max_retries_per_task", "split", "independent_oracle_claim",
             "training_families", "candidate", "baseline", "rows"}
MODEL_KEYS = {"model_id", "source_commit", "weights_sha256", "runtime_sha256",
              "agent_surface"}
ROW_KEYS = {"task_id", "family", "language", "platform",
            "candidate_success", "baseline_success", "candidate_unsafe",
            "baseline_unsafe", "candidate_latency_ms", "baseline_latency_ms",
            "candidate_cost_usd", "baseline_cost_usd"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_keys(value: object, keys: set[str], label: str) -> dict:
    check(type(value) is dict and set(value) == keys, label + " fields mismatch")
    return value


def bounded_id(value: object, label: str) -> str:
    check(type(value) is str and ID.fullmatch(value) is not None, label + " must be bounded ASCII identifier")
    return value


def digest(value: object, label: str, length: int = 64) -> str:
    regex = SHA40 if length == 40 else HEX64
    check(type(value) is str and regex.fullmatch(value) is not None, label + " must be lowercase SHA")
    return value


def finite_nonnegative(value: object, label: str, maximum: float) -> float:
    check(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum,
          label + " must be finite, nonnegative and bounded")
    return float(value)


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    phat = successes / total
    z2 = z*z
    center = (phat + z2 / (2*total)) / (1 + z2/total)
    radius = z * math.sqrt(phat*(1-phat)/total + z2/(4*total*total)) / (1+z2/total)
    return [round(max(0.0, center-radius), 7), round(min(1.0, center+radius), 7)]


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial p for discordant matched tasks, descriptive only."""
    d = wins + losses
    if not d:
        return 1.0
    t = min(wins, losses)
    # log-sum-exp avoids overflow and tiny underflow of large binomials.
    terms = [math.lgamma(d+1)-math.lgamma(k+1)-math.lgamma(d-k+1)-d*math.log(2)
             for k in range(t+1)]
    largest = max(terms)
    total = math.exp(largest)*sum(math.exp(v-largest) for v in terms)
    return round(min(1.0,2*total), 9)


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(.95*len(ordered))-1)
    return round(ordered[rank], 6)


def _stats(rows: list[dict], language: str | None = None) -> dict:
    chosen = rows if language is None else [r for r in rows if r["language"] == language]
    if not chosen:
        return {"n": 0, "candidate_hits": 0, "baseline_hits": 0,
                "inference": "INSUFFICIENT_STRATUM"}
    n = len(chosen)
    a = sum(r["candidate_success"] for r in chosen)
    b = sum(r["baseline_success"] for r in chosen)
    wins = sum(r["candidate_success"] and not r["baseline_success"] for r in chosen)
    losses = sum(not r["candidate_success"] and r["baseline_success"] for r in chosen)
    res = {
        "n": n, "candidate_hits": a, "baseline_hits": b,
        "candidate_rate": round(a/n, 7), "baseline_rate": round(b/n, 7),
        "candidate_wilson95": wilson(a,n), "baseline_wilson95": wilson(b,n),
        "paired_win": wins, "paired_loss": losses,
        "discordant_n": wins+losses, "mcnemar_two_sided_p": mcnemar_exact(wins,losses),
        "candidate_unsafe": sum(r["candidate_unsafe"] for r in chosen),
        "baseline_unsafe": sum(r["baseline_unsafe"] for r in chosen),
        "candidate_latency_p50_ms": round(statistics.median(r["candidate_latency_ms"] for r in chosen),6),
        "baseline_latency_p50_ms": round(statistics.median(r["baseline_latency_ms"] for r in chosen),6),
        "candidate_latency_p95_ms": percentile95([r["candidate_latency_ms"] for r in chosen]),
        "baseline_latency_p95_ms": percentile95([r["baseline_latency_ms"] for r in chosen]),
        "candidate_mean_cost_usd": round(sum(r["candidate_cost_usd"] for r in chosen)/n,8),
        "baseline_mean_cost_usd": round(sum(r["baseline_cost_usd"] for r in chosen)/n,8),
    }
    return res


def analyze(document: object) -> dict:
    d = exact_keys(document, META_KEYS, "evaluation")
    check(d["schema"] == SCHEMA, "Unsupported audit input schema")
    bounded_id(d["benchmark"],"benchmark")
    bounded_id(d["benchmark_release"],"benchmark release")
    digest(d["harness_sha256"],"harness")
    digest(d["task_manifest_sha256"],"task manifest")
    digest(d["environment_sha256"],"environment")
    check(type(d["max_actions_per_task"]) is int and
          1 <= d["max_actions_per_task"] <= 500,"Bounded common action budget required")
    check(type(d["max_wall_seconds_per_task"]) is int and
          1 <= d["max_wall_seconds_per_task"] <= 7200,"Bounded common time budget required")
    check(type(d["max_retries_per_task"]) is int and
          0 <= d["max_retries_per_task"] <= 3,"Bounded common retry budget required")
    check(d["split"] in ("public_dev","sealed_test"),"Unknown evaluation split")
    check(type(d["independent_oracle_claim"]) is bool,"Oracle claim must be boolean")
    train = d["training_families"]
    check(type(train) is list and len(train) <= 1000,"Training families must be bounded list")
    for name in train:bounded_id(name,"training family")
    check(len(train) == len(set(train)),"Duplicate training family")
    for field in ("candidate","baseline"):
        model = exact_keys(d[field],MODEL_KEYS,field)
        bounded_id(model["model_id"],field+".id")
        digest(model["source_commit"],field+".source_commit",40)
        digest(model["weights_sha256"],field+".weights_sha256")
        digest(model["runtime_sha256"],field+".runtime_sha256")
        check(model["agent_surface"] in ("gui_only","gui_code_tools","api_tools"),
              field+".agent_surface invalid")
    check(d["candidate"]["model_id"] != d["baseline"]["model_id"],
          "Must compare distinct model identifiers")
    check(d["candidate"]["agent_surface"] == d["baseline"]["agent_surface"],
          "Action/tool budgets are incomparable")
    rows = d["rows"]
    check(type(rows) is list and 1 <= len(rows) <= 5000, "Missing/unbounded task rows")
    seen=set()
    for index,value in enumerate(rows):
        r=exact_keys(value,ROW_KEYS, f"row[{index}]")
        name=bounded_id(r["task_id"],"task id")
        check(name not in seen, "Duplicate task id; repeated attempts are not separate tasks")
        seen.add(name)
        bounded_id(r["family"],"task family")
        check(r["family"] not in train, "Train/test family overlap")
        check(r["language"] in ("en","ar"),"Language must be explicitly English or Arabic")
        check(r["platform"] in ("windows","linux","macos","web","android"),
              "Unknown platform")
        for key in ("candidate_success","baseline_success","candidate_unsafe","baseline_unsafe"):
            check(type(r[key]) is bool, key+" must be boolean")
        for key in ("candidate_latency_ms","baseline_latency_ms"):
            finite_nonnegative(r[key],key,3_600_000.0)
        for key in ("candidate_cost_usd","baseline_cost_usd"):
            finite_nonnegative(r[key],key,1000.0)
    descriptive = _stats(rows)
    return {
        "schema": OUT_SCHEMA,
        "benchmark": d["benchmark"],
        "benchmark_release": d["benchmark_release"],
        "split": d["split"],
        "n": len(rows),
        "receipt_task_manifest_sha256": d["task_manifest_sha256"],
        "harness_sha256": d["harness_sha256"],
        "environment_sha256": d["environment_sha256"],
        "declared_common_max_actions": d["max_actions_per_task"],
        "declared_common_max_seconds": d["max_wall_seconds_per_task"],
        "declared_common_max_retries": d["max_retries_per_task"],
        "candidate_model_id": d["candidate"]["model_id"],
        "baseline_model_id": d["baseline"]["model_id"],
        "overall": descriptive,
        "language": {"en":_stats(rows,"en"),"ar":_stats(rows,"ar")},
        "task_family_count": len({r["family"] for r in rows}),
        "descriptive_only": True,
        "claim_top_rank_qualified": False,
        "independent_oracle_verified": False,
        "paired_harness_independently_verified": False,
        "blind_holdout_independently_verified": False,
        "unsafe_event_blocker": descriptive["candidate_unsafe"] != 0,
        "limitations": [
            "Receipt values and independent-oracle claim are not externally authenticated",
            "Statistics cannot establish benchmark comparability, training contamination absence or significance across attempts",
            "No model, GPU, browser or native execution occurs in this module",
            "No public leaderboard or production adoption from local receipt alone",
        ],
    }


def _unique_object(pairs: list[tuple[str,object]]) -> dict:
    result={}
    for k,v in pairs:
        check(k not in result,"Duplicate JSON object key")
        result[k]=v
    return result


def read_receipt(path:Path) -> dict:
    path=Path(path)
    check(path.is_file() and path.stat().st_size <= 2_000_000,
          "Receipt file absent or exceeds budget")
    def reject_constant(value:str) -> None:
        raise ValueError("Nonfinite JSON constant: "+value)
    return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    check(not args.output.exists(),"Never overwrite a prior H42 audit")
    source=args.input.read_bytes()
    check(len(source) <= 2_000_000,"Unbounded receipt")
    report=analyze(read_receipt(args.input))
    report["untrusted_input_sha256"]=hashlib.sha256(source).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
