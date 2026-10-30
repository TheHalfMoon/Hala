"""H228 read-only four-file H137 original MIT npm asset provenance verifier.

Validates actual previously-copied PUBLIC vendor JS/CSS byte digests against
the owner-only original H137 runtime provenance and the three H176 historical
npm identities. No old screenshots/gold, no model and no official admission.

Important: equality of three owner-controlled records does NOT authenticate
publisher artifacts or prove complete historical source coverage.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from h176.source_family_attribution import (
    KNOWN_NPM_SOURCES, NPM_ROW_FIELDS, SCHEMA as H176_SCHEMA,
)

SCHEMA = "hala.h228.h137-four-public-original-npm-assets.v1"
H137_VENDOR_SOURCES = {
    "nouislider.min.js": ("nouislider", "15.8.1", "dist/nouislider.min.js"),
    "nouislider.min.css": ("nouislider", "15.8.1", "dist/nouislider.min.css"),
    "sortable.min.js": ("sortablejs", "1.15.6", "Sortable.min.js"),
    "split.min.js": ("split.js", "1.6.5", "dist/split.min.js"),
}
JS_TO_H176 = {
    "nouislider.min.js": "nouislider",
    "sortable.min.js": "sortablejs",
    "split.min.js": "splitjs",
}
MAX_VENDOR_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_H137_SOURCE_BYTES = 128 * 1024


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H228 H137 owner-only public asset byte check: " + reason)


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for k, v in pairs:
        _require(k not in result, "duplicate manifest JSON keys")
        result[k] = v
    return result


def _no_constant(_: str) -> None:
    raise ValueError("H228 nonstandard JSON constant blocked")


def _read_bounded_json(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), "missing public metadata")
    size = path.stat().st_size
    _require(0 < size <= MAX_METADATA_BYTES, "public metadata size out of bounds")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    _require(len(raw) == size, "metadata byte count changed")
    try:
        obj = json.loads(raw.decode("utf-8"),
                         object_pairs_hook=_no_duplicate_pairs,
                         parse_constant=_no_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H228 invalid JSON metadata") from None
    _require(type(obj) is dict, "manifest root must be a dictionary")
    return obj


def _source_contract() -> None:
    """Read ONLY Hala checked-in public H137 source. Never import H137 runtime."""
    source = Path(__file__).resolve().parents[1] / "h137" / "source.py"
    _require(source.is_file() and not source.is_symlink()
             and 0 < source.stat().st_size <= MAX_H137_SOURCE_BYTES,
             "original checked-in H137 source missing or changed")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    vendor_bindings = [
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "VENDOR_SOURCES"
                for t in node.targets)
    ]
    _require(len(vendor_bindings) == 1,
             "ambiguous H137 original vendor source constant")
    _require(ast.literal_eval(vendor_bindings[0]) == H137_VENDOR_SOURCES,
             "H137 original four JS/CSS vendor file map drifted")


def _regular_file_bytes_sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(),
             "original public vendor asset missing or linked")
    count = path.stat().st_size
    _require(0 < count <= MAX_VENDOR_BYTES, "original public vendor size invalid")
    digest = sha256()
    observed = 0
    with path.open("rb") as reader:
        while piece := reader.read(1024 * 1024):
            observed += len(piece)
            _require(observed <= count, "vendor content grew during hash")
            digest.update(piece)
    _require(observed == count and path.stat().st_size == count,
             "vendor content changed during hash")
    return digest.hexdigest()


def verify_h137_original_vendor_assets(
    runtime: Path, owner_attribution: dict,
) -> dict:
    """Verify only 4 original public assets + original H137 manifest.

    The supplied H176 historical metadata is owner-only. The matcher
    verifies consistency, not independent npm publisher authenticity.
    """
    _require(isinstance(runtime, Path) and runtime.is_absolute()
             and runtime.is_dir() and not runtime.is_symlink(),
             "explicit regular absolute original H137 runtime required")
    _source_contract()
    vendor_dir = runtime / "vendor"
    _require(vendor_dir.is_dir() and not vendor_dir.is_symlink(),
             "original H137 vendor directory missing or linked")
    entries = list(vendor_dir.iterdir())
    _require(len(entries) == 4
             and {p.name for p in entries} == set(H137_VENDOR_SOURCES),
             "H137 original npm JS/CSS list incomplete or has extra entries")
    provenance = _read_bounded_json(runtime / "provenance.json")
    _require(set(provenance) == set(H137_VENDOR_SOURCES),
             "H137 provenance must record all four public vendor assets")

    _require(type(owner_attribution) is dict
             and set(owner_attribution) == {"schema", "families"}
             and owner_attribution["schema"] == H176_SCHEMA
             and type(owner_attribution["families"]) is list,
             "exact H176 owner attribution metadata required")
    rows = [row for row in owner_attribution["families"]
            if type(row) is dict and row.get("app") in set(KNOWN_NPM_SOURCES)]
    _require(len(rows) == len(KNOWN_NPM_SOURCES)
             and {row["app"] for row in rows} == set(KNOWN_NPM_SOURCES),
             "exactly three original H137 npm historical app entries required")
    npm_by_app = {}
    for row in rows:
        _require(set(row) == NPM_ROW_FIELDS,
                 "npm historical row cannot contain fake Git/source fields")
        app, package, version, old_sha = (
            row["app"], row["package"], row["version"], row["artifact_sha256"]
        )
        _require(type(package) is str and type(version) is str
                 and (package, version) == KNOWN_NPM_SOURCES[app],
                 "historical npm package/version changed")
        _require(type(old_sha) is str and len(old_sha) == 64
                 and all(c in "0123456789abcdef" for c in old_sha),
                 "historical original H137 npm SHA invalid")
        npm_by_app[app] = old_sha
    observed_hashes = {}
    for name, (expected_package, expected_version, _source_path) in H137_VENDOR_SOURCES.items():
        row = provenance[name]
        _require(type(row) is dict
                 and set(row) == {"package", "version", "sha256", "license"}
                 and type(row["package"]) is str
                 and type(row["version"]) is str
                 and row["package"] == expected_package
                 and row["version"] == expected_version
                 and row["license"] == "MIT",
                 "original H137 public npm provenance mismatch")
        sha = row["sha256"]
        _require(type(sha) is str and len(sha) == 64
                 and all(c in "0123456789abcdef" for c in sha),
                 "original public artifact digest is invalid")
        actual = _regular_file_bytes_sha(vendor_dir / name)
        _require(actual == sha,
                 "public original vendor bytes do not match owner provenance")
        observed_hashes[name] = actual
        if name in JS_TO_H176:
            _require(actual == npm_by_app[JS_TO_H176[name]],
                     "H176 original npm app artifact hash disagrees with H137 JS")
    _require(observed_hashes["nouislider.min.js"]
             != observed_hashes["nouislider.min.css"],
             "noUiSlider CSS must be a distinct original public asset")
    return {
        "status": "H228_H137_FOUR_ORIGINAL_PUBLIC_ASSET_BYTES_MATCH_OWNER_PROVENANCE_ONLY",
        "public_asset_files_verified": 4,
        "js_against_h176_historical_apps": 3,
        "original_css_assets_verified": 1,
        "historical_source_completeness_proven": False,
        "publisher_artifact_authenticity_proven": False,
        "official_train_admitted": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "consumed_ood_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only H137 original four public npm vendor JS/CSS byte audit"
    )
    parser.add_argument("--original-h137-runtime", type=Path, required=True)
    parser.add_argument("--owner-h176-attribution", type=Path, required=True)
    args = parser.parse_args()
    try:
        owner = _read_bounded_json(args.owner_h176_attribution)
        receipt = verify_h137_original_vendor_assets(
            args.original_h137_runtime, owner,
        )
    except (OSError, ValueError, TypeError, SyntaxError):
        # Never echo owner paths, previous H137 bytes or source records.
        raise SystemExit("H228_H137_PUBLIC_NPM_FOUR_ASSET_CHECK_BLOCKED") from None
    print(
        receipt["status"],
        "assets=4 js_source_pairs=3 css=1",
        "historical_complete=false TRAIN=0 DEV=0 OOD=0",
        "model_calls=0 optimizer_steps=0",
    )


if __name__ == "__main__":
    main()
