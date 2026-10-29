"""H186 source-only legacy H141 dual PNG alias coverage receipt.

Owner-local only; print counts, no owner-specific SHA values, paths or
captured images. All underlying source bytes are read from allowlisted
old manifest.json files only. Never claim historical completeness.
"""
from __future__ import annotations

import argparse
import sys

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="H186 original H141 drag-phase PNG SHA exclusions only"
    )
    p.add_argument("--historical-root", action="append", required=True)
    args = p.parse_args(argv)
    try:
        data = scan_legacy_manifest_sha_fields(args.historical_root)
        at_rest = data["h141_at_rest_png_sha_occurrences"]
        in_drag = data["h141_during_drag_png_sha_occurrences"]
        unique = data["distinct_old_drag_phase_png_sha_candidates"]
        if not (type(at_rest) is int and type(in_drag) is int
                and type(unique) is int and at_rest >= 0
                and in_drag >= 0 and unique >= 0):
            raise ValueError("invalid H141 old PNG alias counts")
    except (ValueError, TypeError, KeyError, OSError):
        print("H186_FAIL_CLOSED_LEGACY_ALIAS_METADATA", file=sys.stderr)
        return 2

    print(
        "H186_OWNER_ONLY_H141_OLD_DRAG_PHASE_PNG_EXCLUSION "
        f"manifests={data['manifest_files_scanned']} "
        f"old_png_hashes={data['distinct_png_sha_candidates']} "
        f"h141_at_rest_occurrences={at_rest} "
        f"h141_during_drag_occurrences={in_drag} "
        f"unique_h141_drag_phase_png_hashes={unique} "
        f"old_rgba_hashes={data['distinct_decoded_rgba_sha_candidates']} "
        f"historical_png_set_fingerprint_sha256="
        f"{data['sha_only_metadata_fingerprint_sha256']}"
    )
    print("H186_OWNER_HISTORY_INCOMPLETE_NO_OLD_IMAGES_NO_MODEL_NO_TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
