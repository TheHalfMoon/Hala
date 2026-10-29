"""H179 offline pinned original-source novelty check.

The optional donor paths must all point to pristine original Git checkouts;
they are read-only and cannot touch previously consumed evaluation images.
"""
from __future__ import annotations

import argparse
import sys

from h179.source_novelty import screen_h174_candidates


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="H179 fixed H174 donor SOURCE ONLY novelty; never admission"
    )
    p.add_argument("--historical-root", action="append", required=True)
    p.add_argument("--pickr-donor")
    p.add_argument("--interact-donor")
    p.add_argument("--mediumzoom-donor")
    a = p.parse_args(argv)
    values = (a.pickr_donor, a.interact_donor, a.mediumzoom_donor)
    if all(value is None for value in values):
        roots = None
    elif all(value is not None for value in values):
        roots = {
            "pickr": a.pickr_donor,
            "interact-dropzones": a.interact_donor,
            "medium-zoom": a.mediumzoom_donor,
        }
    else:
        print("H179_FAIL_CLOSED_INCOMPLETE_ORIGINAL_DONOR_SET", file=sys.stderr)
        return 2
    try:
        result = screen_h174_candidates(a.historical_root, roots)
    except (ValueError, OSError, TypeError, KeyError):
        # Never echo owner paths, origin URLs, or private historical identities.
        print("H179_FAIL_CLOSED_ORIGINAL_SOURCE_IDENTITY", file=sys.stderr)
        return 2
    print(
        f"{result['status']} "
        f"candidates={result['candidate_source_families']} "
        f"old_checkouts={result['historical_git_checkouts_screened']} "
        f"old_repos={result['historical_unique_repositories_screened']} "
        f"donor_file_pins={result['independent_local_source_byte_pins_verified']} "
        f"candidate_fp_sha256={result['candidate_source_fingerprint_sha256']} "
        f"history_fp_sha256={result['historical_identity_fingerprint_sha256']}"
    )
    print("HISTORICAL_COMPLETENESS_UNPROVEN_NO_TRAIN_DEV_OOD_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
