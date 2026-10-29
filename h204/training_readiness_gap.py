"""H204: actual owner source-manifest + partial legacy-SHA readiness, NO media.

Copy-first H198 current 20-case quarantine census and H185 historical
SHA-only extraction. This report is an explicit DENIAL of training:
historical completeness cannot be inferred from any finite SHA census.
Never reads original PNG, old OOD, sealed goal sidecars, models, or gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from h178.owner_git_inventory import _safe_directory
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields,
    screen_unadmitted_new_case_hashes,
)
from h198.owner_source_metadata_readiness import (
    COHORTS, MAX_METADATA_BYTES, _regular_small, census,
)
from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS

PHASE = re.compile(r"(?:Hala-H|private-h)([0-9]{2,3})(?:-Private)?\Z",
                   re.IGNORECASE)
MAX_LEGACY_PHASES = 40


def require(ok: bool, reason: str) -> None:
    if not ok:
        # Never print owner roots, image hashes, source instructions or gold.
        raise ValueError("H204 owner metadata gap gate: " + reason)


def _safe_legacy_phases(roots: list[Path]) -> list[Path]:
    require(type(roots) is list and 0 < len(roots) <= MAX_LEGACY_PHASES,
            "explicit bounded legacy phase roots required")
    results: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        require(isinstance(root, Path) and root.is_absolute(),
                "legacy phase must be an absolute path")
        match = PHASE.fullmatch(root.name)
        # H173 was consumed as a one-shot OOD: NEVER walk that phase.
        # H174+ sources are handled in independent, later source research.
        require(match is not None and int(match.group(1)) < 173,
                "only explicit pre-H173 legacy metadata phases permitted")
        require(_safe_directory(root),
                "historical metadata directory missing or reparse")
        key = str(root.resolve()).casefold()
        require(key not in seen, "duplicate historical phase")
        seen.add(key)
        results.append(root)
    return results


def _source_manifest_snapshot(owner_root: Path) -> dict[str, bytes]:
    """Bounded private in-memory source metadata; NEVER media or references.

    A snapshot captures one fixed name per prospective cohort. Comparing
    exact bytes before/after the existing H198 and H185 checks detects
    changed case metadata, flags and SHA pairs across audit stages.
    """
    snapshot: dict[str, bytes] = {}
    for folder_name, _count, _family, _status in COHORTS:
        path = owner_root / folder_name / "source-prospects-not-admitted.json"
        _regular_small(path, "source SHA provenance")
        raw = path.read_bytes()
        require(0 < len(raw) < MAX_METADATA_BYTES,
                "source metadata mutated or oversized during snapshot")
        snapshot[folder_name] = raw
    return snapshot


def _require_unchanged_snapshot(
    owner_root: Path, expected: dict[str, bytes],
) -> None:
    # Comparing exact bytes (not just counts) also catches changed labels,
    # split flags and source hashes that H198 might otherwise count equally.
    current = _source_manifest_snapshot(owner_root)
    require(current == expected,
            "source cohort metadata changed during SHA-only qualification")


def training_readiness_gap(
    owner_root: Path, legacy_phase_roots: list[Path],
) -> dict:
    """Read only approved manifest metadata; never admit any data/model."""
    require(isinstance(owner_root, Path) and owner_root.is_absolute(),
            "owner root must be absolute")
    # All source manifests must be exactly unchanged across the TWO
    # independent qualification passes. Never read goal/reference/media.
    source_snapshot = _source_manifest_snapshot(owner_root)
    # Validates fixed original sources, status/splits, bilingual composition,
    # new image/RGBA metadata and sealed reference EXISTENCE only.
    current = census(owner_root)
    _require_unchanged_snapshot(owner_root, source_snapshot)

    # H185's conservative known-old exclusion uses precisely the same
    # validated first-pass bytes, never a second unverified manifest version.
    pairs: list[dict[str, str]] = []
    for folder_name, _count, _family, _status in COHORTS:
        manifest = json.loads(source_snapshot[folder_name])
        for row in manifest["cases"]:
            pairs.append({
                "image_sha256": row["image_sha256"],
                "visual_rgba_sha256": row["visual_rgba_sha256"],
            })
    historical = scan_legacy_manifest_sha_fields(
        _safe_legacy_phases(legacy_phase_roots)
    )
    comparison = screen_unadmitted_new_case_hashes(pairs, historical)
    _require_unchanged_snapshot(owner_root, source_snapshot)
    require(comparison["proposed_sha_only_pairs_screened"] == 20
            and current["training_approved"] is False
            and current["historic_source_and_rgba_history_complete"] is False,
            "source-only history must remain unadmitted")
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "training SHA allowlist unexpectedly changed")

    manifest_count = historical["manifest_files_scanned"]
    old_png_count = historical["manifests_with_png_sha_fields"]
    old_rgba_count = historical["manifests_with_decoded_rgba_sha_fields"]
    require(0 <= old_png_count <= manifest_count
            and 0 <= old_rgba_count <= manifest_count,
            "historical manifest field census inconsistent")
    return {
        "status": "H204_TRAINING_RELEASE_BLOCKED_PARTIAL_SHA_ONLY_HISTORY",
        "new_unadmitted_original_gui_cases": current[
            "total_quarantined_original_preaction_png_metadata"],
        "new_source_families": current["source_implementation_families"],
        "languages": current["languages"],
        "legacy_pre_h173_phases_scanned": historical["historical_phases_enumerated"],
        "legacy_manifests_scanned": manifest_count,
        "legacy_manifests_with_png_sha": old_png_count,
        "legacy_manifests_with_rgba_sha": old_rgba_count,
        "legacy_manifests_lacking_rgba_sha": manifest_count - old_rgba_count,
        "distinct_known_legacy_png_sha_candidates":
            historical["distinct_png_sha_candidates"],
        "distinct_known_legacy_rgba_sha_candidates":
            historical["distinct_decoded_rgba_sha_candidates"],
        "known_old_png_rgba_overlap_excluded_for_all_new_candidates": True,
        "current_source_metadata_stable_across_audit": True,
        "complete_historical_source_history_proven": False,
        "complete_historical_old_rgba_proven": False,
        "old_image_integrity_proven": False,
        "sealed_dev_gold_qualification_proven": False,
        "official_train_sha_allowlist_nonempty": False,
        "train_admission_approved": False,
        "dev_admission_approved": False,
        "ood_admission_approved": False,
        "model_calls": 0,
        "gpu_optimizer_steps": 0,
        "private_media_opened": 0,
        "private_gold_opened": 0,
        "private_artifacts_uploaded": 0,
        "h173_consumed_ood_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "H204 owner SHA-only conservative training-readiness DENIAL. "
        "Never supply H173 consumed OOD or current TRAIN/DEV gold."
    ))
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--legacy-phase", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        result = training_readiness_gap(args.owner_root, args.legacy_phase)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        parser.exit(2, "H204_FAIL_CLOSED_OWNER_METADATA_NOT_QUALIFIED\n")
    print(
        f"{result['status']} "
        f"new={result['new_unadmitted_original_gui_cases']} "
        f"legacy_phases={result['legacy_pre_h173_phases_scanned']} "
        f"manifests={result['legacy_manifests_scanned']} "
        f"legacy_png={result['distinct_known_legacy_png_sha_candidates']} "
        f"legacy_rgba={result['distinct_known_legacy_rgba_sha_candidates']} "
        f"manifests_missing_rgba={result['legacy_manifests_lacking_rgba_sha']} "
        "NO_ADMISSION_NO_GOLD_NO_MEDIA_NO_OLD_OOD_NO_MODEL"
    )


if __name__ == "__main__":
    main()
