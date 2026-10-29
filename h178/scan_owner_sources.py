"""H178 owner-only source checkout scan and optional four-manifest crosscheck.

No data leaves the owner machine, no network access, no historical images,
gold, task prompts, predictions, model calls, optimizer, or GPU.
"""
from __future__ import annotations

import argparse
import sys

from h176.audit_owner_history import _read_metadata_json
from h178.owner_git_inventory import (
    audit_source_inventory, inventory_local_original_sources,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H178 original Git-origin inventory, NOT dataset admission"
    )
    parser.add_argument(
        "--historical-root", action="append", required=True,
        help="local directory with private-hNN source-checkout folders",
    )
    parser.add_argument("--candidates")
    parser.add_argument("--prior-pixel-hashes")
    parser.add_argument("--prior-source-history")
    parser.add_argument("--paired-source-history")
    args = parser.parse_args(argv)
    four = (
        args.candidates, args.prior_pixel_hashes,
        args.prior_source_history, args.paired_source_history,
    )
    try:
        if all(value is None for value in four):
            # Inventory-only mode is not training approval. Its counts and
            # hashed receipt do not establish past historical completeness.
            result = inventory_local_original_sources(args.historical_root)
            count = result["source_checkouts_inspected"]
            distinct = result["distinct_repositories"]
            fingerprint = result["source_identity_fingerprint_sha256"]
            status = "H178_OWNER_SOURCE_ONLY_INVENTORY_NOT_ADMISSION"
        elif all(value is not None for value in four):
            result = audit_source_inventory(
                _read_metadata_json(args.candidates),
                _read_metadata_json(args.prior_pixel_hashes),
                _read_metadata_json(args.prior_source_history),
                _read_metadata_json(args.paired_source_history),
                args.historical_root,
            )
            count = result["source_checkouts_inspected"]
            distinct = result["distinct_observed_repositories"]
            fingerprint = result["source_identity_fingerprint_sha256"]
            status = result["status"]
        else:
            raise ValueError("partial owner manifests cannot qualify")
    except (ValueError, OSError, UnicodeError, TypeError, KeyError):
        # Do not echo file paths, remotes, repo names, Git stderr, or data.
        print("H178_FAIL_CLOSED_OWNER_ORIGINAL_GIT_CHECK", file=sys.stderr)
        return 2

    print(
        f"{status} source_checkouts={count} distinct_repos={distinct} "
        f"source_fingerprint_sha256={fingerprint}"
    )
    print("HISTORY_COMPLETENESS_UNPROVEN_NO_DATASET_NO_MODEL_TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
