"""H177 owner-only public-source crosscheck CLI; strictly metadata-only.

Copy-first: reuse H176's bounded JSON reader and four-argument interface.
It never opens historical media, writes files or prints owner source rows,
private paths, gold, predictions or screenshot data.
"""
from __future__ import annotations

import argparse
import sys

from h176.audit_owner_history import _read_metadata_json
from h177.public_origin_pins import audit_public_origin_pins


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H177 owner-only historic original-source audit; NOT training approval"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prior-pixel-hashes", required=True)
    parser.add_argument("--prior-source-history", required=True)
    parser.add_argument("--paired-source-history", required=True)
    args = parser.parse_args(argv)

    try:
        proof = audit_public_origin_pins(
            _read_metadata_json(args.candidates),
            _read_metadata_json(args.prior_pixel_hashes),
            _read_metadata_json(args.prior_source_history),
            _read_metadata_json(args.paired_source_history),
        )
    except (ValueError, OSError, UnicodeError, TypeError, KeyError):
        # Do not reveal file paths, user source aliases or exception details.
        print("H177_FAIL_CLOSED_OWNER_SOURCE_METADATA", file=sys.stderr)
        return 2

    print(
        f"{proof['status']} "
        f"public_pins={proof['public_git_repository_pin_pairs_checked']} "
        f"historical_apps_at_least={proof['known_public_previous_app_ids_at_least']} "
        f"fingerprint_sha256={proof['owner_attribution_fingerprint_sha256']}"
    )
    print("HISTORICAL_COMPLETENESS_UNPROVEN_NO_DATASET_NO_MODEL_TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
