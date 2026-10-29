"""H181 owner-only read-only SOURCE-reservation command.

Print public Gantt source metadata fingerprints and aggregate counts ONLY.
Never display owner historical repo names, paths, task labels or OOD scenes.
"""
from __future__ import annotations

import argparse
import sys

from h181.ood_source_reservation import (
    check_reserved_source_only, preregistered_identity,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H181 original Gantt OOD source ID only, NOT OOD evaluation"
    )
    parser.add_argument("--historical-root", required=True, action="append")
    args = parser.parse_args(argv)
    try:
        result = check_reserved_source_only(
            preregistered_identity(), args.historical_root
        )
    except (ValueError, TypeError, OSError, KeyError):
        print("H181_FAIL_CLOSED_ORIGINAL_OOD_SOURCE_IDENTITY",
              file=sys.stderr)
        return 2
    print(
        f"{result['status']} "
        f"old_repositories={result['owner_historical_git_repositories_screened']} "
        f"old_checkouts={result['owner_historical_git_checkouts_screened']} "
        f"source_reservation_sha256={result['source_reservation_fingerprint_sha256']} "
        f"source_history_binding_sha256={result['source_history_snapshot_binding_sha256']}"
    )
    print("NO_OOD_IMAGES_NO_MODEL_CALLS_HISTORY_COMPLETENESS_UNPROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
