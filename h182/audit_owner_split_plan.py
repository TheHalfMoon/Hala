"""H182 source-role metadata only: owner-local safe preflight CLI."""
from __future__ import annotations

import argparse
import sys

from h174.source_candidates import CANDIDATES
from h182.source_role_plan import (
    audit_preregistered_sources, preregistered_plan,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H182 public-source TRAIN/DEV proposed roles; NO DATASET"
    )
    parser.add_argument("--historical-root", action="append", required=True)
    parser.add_argument("--pickr-donor")
    parser.add_argument("--interact-donor")
    parser.add_argument("--mediumzoom-donor")
    args = parser.parse_args(argv)
    donor_args = (
        args.pickr_donor, args.interact_donor, args.mediumzoom_donor
    )
    if all(v is None for v in donor_args):
        roots = None
    elif all(type(v) is str and len(v) > 0 for v in donor_args):
        roots = {
            "pickr": args.pickr_donor,
            "interact-dropzones": args.interact_donor,
            "medium-zoom": args.mediumzoom_donor,
        }
    else:
        print("H182_FAIL_CLOSED_PARTIAL_DONOR_SOURCE_PATHS", file=sys.stderr)
        return 2
    try:
        proof = audit_preregistered_sources(
            preregistered_plan(), args.historical_root, roots
        )
    except (ValueError, OSError, TypeError, KeyError):
        print("H182_FAIL_CLOSED_SOURCE_ROLE_PLAN", file=sys.stderr)
        return 2
    print(
        f"{proof['status']} "
        f"train_sources={proof['prospective_train_source_count']} "
        f"dev_sources={proof['prospective_dev_source_count']} "
        f"research_only={proof['exposed_research_unassigned_source_count']} "
        f"ood_reserved_sources={proof['tentatively_reserved_ood_source_count']} "
        f"original_donor_byte_pins={proof['pinned_pristine_original_donor_files_verified']} "
        f"old_source_checkouts={proof['owner_old_git_checkouts_scanned']} "
        f"plan_sha256={proof['source_role_plan_fingerprint_sha256']}"
    )
    print("HISTORY_INCOMPLETE_NO_CASE_DATA_NO_TRAINING_NO_OOD_CAPTURE")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
