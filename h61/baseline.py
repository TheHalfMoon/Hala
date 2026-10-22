"""H61 the public DEV baseline receipt, derived from the H60 real inference evidence.

The training gate requires a pinned real unchanged-model baseline with a
measured deficit. The H60 evidence file already holds that measurement, so the
H61 receipt is *derived* from it instead of being written by hand, and the
held-out dataset digest binds the two public DEV sources that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "hala.h61.public-dev-baseline-receipt.v1"
HELD_OUT_SOURCE = "original_firstparty_public_dev_h55_h58_source_disjoint_not_training"
TASK_KIND = "mixed_ordered_unordered_drag"


def build_baseline_receipt(evidence_path) -> dict:
    """Build the training gate baseline receipt from the H60 real evidence."""
    evidence = json.loads(Path(evidence_path).read_text("utf8"))
    if (
        evidence.get("evidence") != "real_inference"
        or evidence.get("model_changed") is not False
        or evidence.get("independent_blind_test_completed") is not False
    ):
        raise ValueError("A real unchanged-model public DEV measurement is required")
    totals = evidence["totals"]
    cases = int(totals["cases"])
    success = int(totals["real_edge_gui_success"])
    if cases <= 0 or not 0 <= success <= cases:
        raise ValueError("The measured public DEV totals are inconsistent")
    held_out = "|".join(
        sorted(
            evidence["splits"][name]["source_sha256"]
            for name in evidence["splits"]
        )
    )
    return {
        "schema": SCHEMA,
        "model_id": evidence["model_id"],
        "revision": evidence["revision"],
        "evidence": "real_inference",
        "task_kind": TASK_KIND,
        "held_out_source": HELD_OUT_SOURCE,
        "evaluated_tasks": cases,
        "verified_success_rate": success / cases,
        "deficit": evidence["deficit"],
        "held_out_dataset_sha256": hashlib.sha256(held_out.encode("utf8")).hexdigest(),
        "independent_blind_test_completed": False,
        "official_benchmark_comparison_qualified": False,
        "provider": evidence.get("provider"),
        "gpu": evidence.get("gpu"),
        "paid_compute_units_consumed": evidence.get("paid_compute_units_consumed", 0),
        "source_evidence_sha256": hashlib.sha256(
            Path(evidence_path).read_bytes()
        ).hexdigest(),
        "measured_totals": totals,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build_baseline_receipt(args.evidence)
    Path(args.output).write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
