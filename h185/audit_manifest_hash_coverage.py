"""H185 authorized Windows SHA-only legacy manifest coverage audit CLI.

Print only aggregate counts and one digest. No private SHA values, filenames,
owner paths, gold, screenshot pixels or model state are emitted or saved.
"""
from __future__ import annotations

import argparse
import sys

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H185 old manifest SHA fields ONLY; not dataset approval"
    )
    parser.add_argument("--historical-root", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = scan_legacy_manifest_sha_fields(args.historical_root)
    except (OSError, TypeError, KeyError, ValueError):
        print("H185_FAIL_CLOSED_OWNER_METADATA_COVERAGE", file=sys.stderr)
        return 2
    print(
        f"{receipt['status']} "
        f"phases={receipt['historical_phases_enumerated']} "
        f"manifests={receipt['manifest_files_scanned']} "
        f"png_sha_manifests={receipt['manifests_with_png_sha_fields']} "
        f"rgba_sha_manifests={receipt['manifests_with_decoded_rgba_sha_fields']} "
        f"unique_png_sha_candidates={receipt['distinct_png_sha_candidates']} "
        f"unique_rgba_sha_candidates={receipt['distinct_decoded_rgba_sha_candidates']} "
        f"sha_metadata_fingerprint={receipt['sha_only_metadata_fingerprint_sha256']}"
    )
    print("H185_HISTORY_COMPLETENESS_UNPROVEN_NO_IMAGE_MEDIA_OPENED_NO_TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
