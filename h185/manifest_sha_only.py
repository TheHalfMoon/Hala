"""H185 owner-only legacy manifest SHA-field coverage (NOT data admission).

Reads allowlisted *manifest.json* bytes only to find exact quoted SHA-only
image/decoded-RGBA digest fields; never opens old .png/.jpg, old gold,
predictions, raw model responses, weights, archives or score files. It
does NOT JSON-decode case rows or use their reference actions/labels.

A field occurrence is untrusted evidence of metadata, NOT proof that the
image existed, of source provenance, of historical registry completeness,
or of fresh training eligibility. Print only aggregate counts/fingerprint.
"""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import re

from h178.owner_git_inventory import _safe_directory, PHASE_NAME

MAX_PHASES = 100
MAX_MANIFESTS = 512
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_DEPTH = 4

# Only exactly named hashes, never generic "sha256", which could be a model
# checkpoint, reference label, artifact, tokenizer or unrelated provenance.
RAW = re.compile(rb'"(?:image_sha256|png_sha256)"\s*:\s*"([0-9a-f]{64})"')
VISUAL = re.compile(
    rb'"(?:visual_rgba_sha256|rgba_sha256)"\s*:\s*"([0-9a-f]{64})"'
)
IGNORE_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "upstream", "vendor", "vendor-apps", "donors", "sources",
    "models", "model", "adapters", "adapter", "checkpoints",
    "weights", "screenshots", "images", "frames", "captures",
    "gold", "predictions", "outputs", "cache", "site-packages",
})


def _require(ok: bool, why: str) -> None:
    # Never echo owner paths, manifest content or digest values in errors.
    if not ok:
        raise ValueError("H185 owner manifest SHA field coverage: " + why)


def _historical_phases(roots: list[str | Path]) -> list[Path]:
    _require(type(roots) is list and 0 < len(roots) <= MAX_PHASES,
             "bounded historical root list required")
    phases: list[Path] = []
    keys: set[str] = set()
    for value in roots:
        _require(type(value) in (str, Path), "historical root type invalid")
        root = Path(value)
        _require(_safe_directory(root), "unsafe historical root")
        if PHASE_NAME.fullmatch(root.name):
            found = [root]
        else:
            found = []
            for candidate in root.iterdir():
                if PHASE_NAME.fullmatch(candidate.name):
                    _require(_safe_directory(candidate),
                             "unsafe historical phase name")
                    found.append(candidate)
        for phase in sorted(found, key=lambda path: path.name.casefold()):
            _require(_safe_directory(phase), "unsafe historical phase")
            key = os.path.normcase(str(phase.resolve()))
            _require(key not in keys, "duplicate historical phase")
            keys.add(key)
            phases.append(phase)
    _require(0 < len(phases) <= MAX_PHASES,
             "historical source phases missing")
    return phases


def scan_legacy_manifest_sha_fields(roots: list[str | Path]) -> dict:
    """Bounded, no-write SHA-field census, not a trusted old-media inventory.

    Only manifests, never image bytes/JSON case labels. A SHA field may be
    missing in older schemas, so counts cannot prove historical coverage.
    """
    phases = _historical_phases(roots)
    raw: set[str] = set()
    rgba: set[str] = set()
    manifest_count = 0
    raw_manifest_count = 0
    rgba_manifest_count = 0
    raw_occurrences = 0
    rgba_occurrences = 0
    total_bytes = 0

    for phase in phases:
        for current, dir_names, files in os.walk(
            phase, topdown=True, followlinks=False
        ):
            directory = Path(current)
            depth = len(directory.relative_to(phase).parts)
            _require(depth <= MAX_DEPTH, "manifest recursion depth exceeded")
            safe_names: list[str] = []
            if depth < MAX_DEPTH:
                for name in sorted(dir_names):
                    if name.casefold() in IGNORE_DIRS:
                        continue
                    child = directory / name
                    _require(_safe_directory(child),
                             "unsafe manifest directory or junction")
                    safe_names.append(name)
            dir_names[:] = safe_names

            # Ordinary irrelevant file names are never read or opened.
            if "manifest.json" not in files:
                continue
            target = directory / "manifest.json"
            _require(not target.is_symlink()
                     and not (getattr(target.lstat(),
                                     "st_file_attributes", 0) & 0x400),
                     "unsafe manifest file/reparse point")
            _require(target.is_file(), "manifest is not a regular file")
            size = target.stat().st_size
            _require(0 < size <= MAX_FILE_BYTES, "manifest size not bounded")
            manifest_count += 1
            total_bytes += size
            _require(manifest_count <= MAX_MANIFESTS
                     and total_bytes <= MAX_TOTAL_BYTES,
                     "legacy manifest census size exceeded")

            # Byte extraction deliberately does not parse JSON/gold fields
            # or decode case payloads; only these four metadata key names.
            blob = target.read_bytes()
            _require(len(blob) == size, "manifest changed while reading")
            old_raw = {hit.group(1).decode("ascii")
                       for hit in RAW.finditer(blob)}
            old_rgba = {hit.group(1).decode("ascii")
                        for hit in VISUAL.finditer(blob)}
            raw_occurrences += len(list(RAW.finditer(blob)))
            rgba_occurrences += len(list(VISUAL.finditer(blob)))
            if old_raw:
                raw_manifest_count += 1
            if old_rgba:
                rgba_manifest_count += 1
            raw.update(old_raw)
            rgba.update(old_rgba)

    _require(manifest_count > 0,
             "no allowlisted original historical manifest found")
    # Digest over ONLY the extracted old SHA values and their digest type,
    # not filenames, private paths, gold, timestamps, labels or predictions.
    digest = sha256("\n".join(
        ["raw:" + value for value in sorted(raw)]
        + ["rgba:" + value for value in sorted(rgba)]
    ).encode("ascii")).hexdigest()
    return {
        "status": "H185_OWNER_MANIFEST_SHA_FIELD_COVERAGE_ONLY_NOT_HISTORY_COMPLETE",
        "historical_phases_enumerated": len(phases),
        "manifest_files_scanned": manifest_count,
        "manifests_with_png_sha_fields": raw_manifest_count,
        "manifests_with_decoded_rgba_sha_fields": rgba_manifest_count,
        "png_sha_occurrences": raw_occurrences,
        "decoded_rgba_sha_occurrences": rgba_occurrences,
        "distinct_png_sha_candidates": len(raw),
        "distinct_decoded_rgba_sha_candidates": len(rgba),
        "sha_only_metadata_fingerprint_sha256": digest,
        # An owner-only caller may use these as conservative EXCLUSIONs,
        # never as an approved complete historical registry.
        "_owner_only_png_sha_candidates": frozenset(raw),
        "_owner_only_rgba_sha_candidates": frozenset(rgba),
        "complete_historical_manifest_coverage_proven": False,
        "old_image_integrity_proven": False,
        "historical_gold_accessed": False,
        "old_image_media_opened": False,
        "training_dataset_admitted": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
    }
