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
# H141 original source (h141/interactive_source.py, h141/audit.py)
# recorded TWO genuine PNG frames for each old oracle-guided diagnostic:
# at_rest_sha256 and during_drag_sha256. Those were never TRAIN/BLIND
# eligible. Preserve them as additional *old PNG exclusions*, not RGBA.
# The unrelated generic sha256 / license/model hashes are forbidden.
RAW = re.compile(
    rb'"(?P<field>image_sha256|png_sha256|at_rest_sha256|during_drag_sha256)"'
    rb'\s*:\s*"(?P<sha>[0-9a-f]{64})"'
)
DRAG_PHASE_FIELDS = frozenset({"at_rest_sha256", "during_drag_sha256"})
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
        _require(isinstance(value, (str, Path)), "historical root type invalid")
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
    drag_phase_occurrences = {"at_rest_sha256": 0, "during_drag_sha256": 0}
    drag_phase_png: set[str] = set()
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
            png_matches = list(RAW.finditer(blob))
            old_raw = {hit.group("sha").decode("ascii")
                       for hit in png_matches}
            old_rgba = {hit.group(1).decode("ascii")
                        for hit in VISUAL.finditer(blob)}
            for hit in png_matches:
                field = hit.group("field").decode("ascii")
                if field in DRAG_PHASE_FIELDS:
                    drag_phase_occurrences[field] += 1
                    drag_phase_png.add(hit.group("sha").decode("ascii"))
            raw_occurrences += len(png_matches)
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
        "h141_at_rest_png_sha_occurrences":
            drag_phase_occurrences["at_rest_sha256"],
        "h141_during_drag_png_sha_occurrences":
            drag_phase_occurrences["during_drag_sha256"],
        "distinct_old_drag_phase_png_sha_candidates": len(drag_phase_png),
        "distinct_decoded_rgba_sha_candidates": len(rgba),
        "sha_only_metadata_fingerprint_sha256": digest,
        # An owner-only caller may use these as conservative EXCLUSIONs,
        # never as an approved complete historical registry.
        "_owner_only_png_sha_candidates": frozenset(raw),
        "_owner_only_rgba_sha_candidates": frozenset(rgba),
        "complete_historical_manifest_coverage_proven": False,
        "old_image_integrity_proven": False,
        "gold_field_values_parsed": False,
        "old_image_media_opened": False,
        "training_dataset_admitted": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
    }


# Only a conservative known-hash exclusion, never a positive admission.
NEW_SHA_FIELDS = frozenset({"image_sha256", "visual_rgba_sha256"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_NEW_CASES = 100_000


def screen_unadmitted_new_case_hashes(
    new_source_hash_pairs: list[dict], old_manifest_census: dict,
) -> dict:
    """Reject known historical PNG/RGBA duplicates, do not admit cases.

    This is useful even if some old manifest SHA fields are missing, but
    unknown prior media or absent old RGBA hashes remain a hard blocker
    for any real training/DEV qualification.
    """
    _require(
        type(new_source_hash_pairs) is list
        and 0 < len(new_source_hash_pairs) <= MAX_NEW_CASES,
        "bounded new SHA-only candidates required",
    )
    _require(
        type(old_manifest_census) is dict
        and old_manifest_census.get("status")
        == "H185_OWNER_MANIFEST_SHA_FIELD_COVERAGE_ONLY_NOT_HISTORY_COMPLETE"
        and type(old_manifest_census.get("_owner_only_png_sha_candidates"))
        is frozenset
        and type(old_manifest_census.get("_owner_only_rgba_sha_candidates"))
        is frozenset
        and old_manifest_census.get("complete_historical_manifest_coverage_proven")
        is False,
        "owner manifest SHA metadata census required; not complete",
    )
    old_png = old_manifest_census["_owner_only_png_sha_candidates"]
    old_rgba = old_manifest_census["_owner_only_rgba_sha_candidates"]
    local_png: set[str] = set()
    local_rgba: set[str] = set()
    for row in new_source_hash_pairs:
        _require(type(row) is dict and set(row) == NEW_SHA_FIELDS,
                 "new SHA-pair metadata only, no source gold or model input")
        png, rgba = row["image_sha256"], row["visual_rgba_sha256"]
        _require(type(png) is str and HEX64.fullmatch(png) is not None
                 and type(rgba) is str and HEX64.fullmatch(rgba) is not None,
                 "new candidate SHA-256 invalid")
        _require(png not in old_png and rgba not in old_rgba,
                 "new scene reuses known old PNG or RGBA visual hash")
        _require(png not in local_png and rgba not in local_rgba,
                 "new candidate duplicated within a proposed dataset")
        local_png.add(png)
        local_rgba.add(rgba)
    return {
        "status": "H185_KNOWN_LEGACY_SHA_REUSE_EXCLUDED_ONLY_NOT_DATA_ADMISSION",
        "proposed_sha_only_pairs_screened": len(new_source_hash_pairs),
        "known_legacy_png_hashes_compared": len(old_png),
        "known_legacy_rgba_hashes_compared": len(old_rgba),
        "legacy_rgba_sha_coverage_missing":
            len(old_rgba) == 0,
        "historical_completeness_proven": False,
        "native_source_qa_proven": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
    }
