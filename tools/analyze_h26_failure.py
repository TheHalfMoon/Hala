"""Post-hoc, hash-bound H26 DEV failure forensics. Never an action/authority gate.

No model invocation, calibration, hidden TEST or rewriting of historical results.
The recorded source/oracle and original raw bytes are independently audited first.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from h26 import runner
from h24.mind_dev import strict_json
from tools.verify_h26_dev import inspect


SCHEMA = "hala.h27.h26_failure_forensics.v1"


def _counts(items):
    counts = Counter(items)
    return {key: counts[key] for key in sorted(counts)}


def describe(outcomes):
    """Derive descriptive counts, after source replay; never infer confidence."""
    if type(outcomes) is not list:
        raise ValueError("Full measured outcome list required")
    # Enforces exact original order, row metadata and per-case source-oracle scoring.
    replay = runner.aggregate(outcomes)
    original = runner.validated_rows()
    by_language = {}
    for lang in ("en", "ar", "all"):
        rows = outcomes if lang == "all" else [
            row for row in outcomes if row["language"] == lang
        ]
        c1 = [row for row in rows if row["raw"] == "C1"]
        proposed = [row for row in rows if row["suggestion"] is not None]
        valid_nonwithheld = [
            row for row in rows if row["format_valid"] and not row["withheld"]
        ]
        risky = [row for row in rows if row["state"] == "unsafe_user"]
        by_language[lang] = {
            "cases": len(rows),
            "observed_codes": _counts(
                row["raw"] if row["raw"] is not None else "INFERENCE_ERROR"
                for row in rows
            ),
            "c1_selected": len(c1),
            "c1_correct": sum(row["semantic_correct"] is True for row in c1),
            "c0_withheld": sum(row["raw"] == "C0" for row in rows),
            "nonwithheld_valid": len(valid_nonwithheld),
            "nonwithheld_semantic_correct": sum(
                row["semantic_correct"] is True for row in valid_nonwithheld
            ),
            "typed_proposals": len(proposed),
            "unsafe_user_cases": len(risky),
            "unsafe_user_explicit_c7": sum(row["raw"] == "C7" for row in risky),
            "unsafe_user_withheld_c0": sum(row["raw"] == "C0" for row in risky),
            "unsafe_user_wrong_proposal": sum(
                row["suggestion"] is not None
                and row["suggestion"]["tool"] != "abstain"
                for row in risky
            ),
            "semantic_correct": replay[lang]["semantic_correct"],
        }
        if by_language[lang]["unsafe_user_wrong_proposal"] != replay[lang]["unsafe_user_misrouted"]:
            raise ValueError("Source scoring mismatch in unsafe routing")
    confusion = {}
    for state, expected in runner.corpus.STATE_TO_CODE.items():
        scoped = [row for row in outcomes if row["state"] == state]
        confusion[state] = {
            "expected": expected,
            "observed": _counts(
                row["raw"] if row["raw"] is not None else "INFERENCE_ERROR"
                for row in scoped
            ),
            "correct": replay["all"]["by_state"][state]["correct"],
            "cases": len(scoped),
        }
    return {
        "schema": SCHEMA,
        "source_protocol": runner.RUN_PROTOCOL,
        "partition": "public_dev_posthoc",
        "independent_test": False,
        "new_model_inference": False,
        "causal_root_cause_established": False,
        "confidence_calibrated": False,
        "model_or_release_admissible": False,
        "native_execution_authorized": False,
        "note": "Constrained syntax does not establish semantic correctness or safe refusal. C0 is withholding, not verified C7 refusal. No ablations, logits or independent bilingual adjudication.",
        "by_language": by_language,
        "by_state": confusion,
    }


def analyze(raw_path, expected_sha256, source_commit):
    """Verify original bytes AND H26 case replay before descriptive diagnostics."""
    # Avoid treating arbitrary or forged source-oracle JSON as real experiments.
    checked = inspect(raw_path, expected_sha256, source_commit)
    raw = Path(raw_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("H26 raw file changed after independent source replay")
    record = strict_json(raw.decode("utf-8"))
    result = describe(record["outcomes"])
    result["source_commit"] = source_commit
    result["raw_sha256"] = expected_sha256
    result["original_source_sha256"] = runner.EXACT_DATA_SHA
    result["source_replay_complete"] = checked["all_42_recomputed"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_result")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", required=True, help="Create a new report; never overwrite")
    args = p.parse_args()
    report = analyze(args.raw_result, args.expected_sha256, args.source_commit)
    with Path(args.output).open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
