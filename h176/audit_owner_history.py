"""H176 owner-only CLI for *paired source metadata*, never previous media.

Reads exactly four small JSON identity manifests from the owner's local disk.
Prints only readiness-blocked status, counts and SHA-256 metadata fingerprints.
It does not create datasets, training runs, screenshots or network calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from h176.source_family_attribution import audit_attribution

MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _read_metadata_json(filename: str) -> dict:
    path = Path(filename)
    if path.is_symlink() or not path.is_file():
        raise ValueError("not a regular metadata file")
    if not (0 < path.stat().st_size <= MAX_MANIFEST_BYTES):
        raise ValueError("oversized or empty metadata file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("oversized metadata file")
    obj = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates,
        parse_constant=_reject_constant,
    )
    if type(obj) is not dict:
        raise ValueError("metadata JSON root must be an object")
    return obj


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H176 local source identity preflight; no dataset admission"
    )
    parser.add_argument("--candidates", required=True, help="fresh candidate metadata JSON")
    parser.add_argument("--prior-pixel-hashes", required=True, help="owner SHA-only registry JSON")
    parser.add_argument("--prior-source-history", required=True, help="legacy source history JSON")
    parser.add_argument("--paired-source-history", required=True, help="owner paired source identities JSON")
    args = parser.parse_args(argv)
    try:
        result = audit_attribution(
            _read_metadata_json(args.candidates),
            _read_metadata_json(args.prior_pixel_hashes),
            _read_metadata_json(args.prior_source_history),
            _read_metadata_json(args.paired_source_history),
        )
    except (ValueError, OSError, UnicodeError, TypeError):
        # Never echo owner-only identity values, paths, or raw parse exceptions.
        print("H176_FAIL_CLOSED_OWNER_METADATA_CHECK", file=sys.stderr)
        return 2
    print(
        f"{result['status']} families={result['paired_source_families']} "
        f"repos={result['distinct_historical_repositories']} "
        f"frozen_pairs={result['frozen_h173_pairs_matched']} "
        f"fingerprint_sha256={result['attribution_sha256']}"
    )
    print("HISTORY_COMPLETENESS_UNPROVEN_NO_TRAIN_DEV_OOD_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
