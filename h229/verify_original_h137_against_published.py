"""H229: strict owner-only H137 four public JS/CSS SHA-256 vs published npm.

Requires H228's original vendor/provenance/H176 check FIRST, then compares
all 4 original local public files to independently SHA-pinned npm publication.
No old H137 screenshots/gold, H173 OOD or model use; no network, copies,
writes, output of original paths or private values. Owner-only opt-in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from h228.original_npm_asset_integrity import (
    H137_VENDOR_SOURCES,
    _read_bounded_json,
    _regular_file_bytes_sha,
    verify_h137_original_vendor_assets,
)
from h229.verified_published_npm_assets import (
    PINNED_PUBLISHED_ORIGINAL_SHA256,
)

def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H229 original owner H137 npm publication byte-check blocked: " + why)


def verify_original_h137_matches_real_npm_published_bytes(
    runtime: Path, private_owner_h176_attribution: dict,
) -> dict:
    """Only trusted full-archive H229 hardcoded publisher SHA256 values.

    Synthetic tests can separately patch the imported dict, but the CLI
    has NO user argument or environmental override for publisher pins.
    """
    previous = verify_h137_original_vendor_assets(
        runtime, private_owner_h176_attribution,
    )
    _require(previous["public_asset_files_verified"] == 4
             and previous["js_against_h176_historical_apps"] == 3
             and previous["original_css_assets_verified"] == 1,
             "H228 original local four-asset evidence incomplete")
    _require(
        set(PINNED_PUBLISHED_ORIGINAL_SHA256) == set(H137_VENDOR_SOURCES),
        "H137 original four assets do not match pinned npm publication list",
    )
    for name in sorted(H137_VENDOR_SOURCES):
        expected_sha = PINNED_PUBLISHED_ORIGINAL_SHA256[name]
        _require(
            type(expected_sha) is str and len(expected_sha) == 64
            and all(c in "0123456789abcdef" for c in expected_sha),
            "invalid separately pinned published npm SHA256",
        )
        actual_sha = _regular_file_bytes_sha(runtime / "vendor" / name)
        _require(actual_sha == expected_sha,
                 "original owner file differs from independently published npm bytes")
    return {
        "status": "H229_ORIGINAL_H137_ALL_FOUR_PUBLISHED_NPM_BYTES_MATCHED",
        "original_files_checked": 4,
        "published_npm_packages_pinned": 3,
        "publisher_attestation_verified": False,
        "historical_sources_complete": False,
        "official_training_approved": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "consumed_ood_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Owner-only read-only original H137 public JS+CSS vs npm tarballs"
    )
    parser.add_argument("--original-h137-runtime", type=Path, required=True)
    parser.add_argument("--private-owner-h176-attribution",
                        type=Path, required=True)
    args = parser.parse_args()
    try:
        owner_attribution = _read_bounded_json(
            args.private_owner_h176_attribution,
        )
        receipt = verify_original_h137_matches_real_npm_published_bytes(
            args.original_h137_runtime, owner_attribution,
        )
    except (ValueError, TypeError, OSError, SyntaxError):
        raise SystemExit(
            "H229_ORIGINAL_H137_INDEPENDENT_NPM_BYTE_MATCH_BLOCKED"
        ) from None
    print(
        receipt["status"],
        "assets=4 published_packages=3",
        "history_complete=false official_TRAIN=0",
        "model_calls=0 optimizer_steps=0 consumed_ood=0",
    )


if __name__ == "__main__":
    main()
