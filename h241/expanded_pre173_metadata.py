"""H241: expand owner-only pre-H173 SHA metadata exclusion, never admission.

Discover only immediate historical phase roots (strict Hala/private naming),
with phase numbers strictly below 173. Reuse H204's immutable private source
manifest snapshot and H185's SHA-field-only legacy metadata parser.
No image, old scored OOD, gold, training assets, model or network calls.
No source IDs, owner paths, source hashes or private manifests in stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from h178.owner_git_inventory import PHASE_NAME, _safe_directory
from h204.training_readiness_gap import (
    MAX_LEGACY_PHASES, training_readiness_gap,
)

NAME = PHASE_NAME  # exact H185 legacy phase names; no silent subset mismatch
POSSIBLE_OLD = re.compile(r"(?:private-h|hala-h)([0-9]{2,3})(?:-private)?\Z", re.I)
LIMIT_PARENTS = 8


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H241 historical metadata gate: " + why)


def find_pre_h173_phases(parents: list[Path]) -> list[Path]:
    """Explicit parent roots, immediate child folders only; never touch H173."""
    require(type(parents) is list and 0 < len(parents) <= LIMIT_PARENTS,
            "bounded historical parent roots required")
    found: list[Path] = []
    seen_parents: set[str] = set()
    seen_phases: set[str] = set()
    for parent in parents:
        require(isinstance(parent, Path) and parent.is_absolute()
                and _safe_directory(parent),
                "historical parent root inaccessible or reparse")
        parent_id = os.path.normcase(str(parent.resolve())).casefold()
        require(parent_id not in seen_parents, "duplicate parent root")
        seen_parents.add(parent_id)
        for child in sorted(parent.iterdir(), key=lambda p: p.name.casefold()):
            match = NAME.fullmatch(child.name)
            # PHASE_NAME has no capture groups; extract digits only AFTER
            # matching the exact private phase naming supported by H185.
            if match is None:
                continue
            number = re.search(r"(?i)h([0-9]{2,3})", child.name)
            require(number is not None, "unsupported pre-H173 phase naming")
            if int(number.group(1)) >= 173:
                # The consumed H173 directory is NEVER traversed here.
                continue
            require(_safe_directory(child),
                    "pre-H173 phase is inaccessible or reparse")
            key = os.path.normcase(str(child.resolve())).casefold()
            require(key not in seen_phases, "duplicate historical phase path")
            seen_phases.add(key)
            found.append(child)
            require(len(found) <= MAX_LEGACY_PHASES,
                    "too many older historical phases")
    require(bool(found), "no eligible pre-H173 historical phase")
    return found


def count_unscanned_old_name_variants(parents: list[Path]) -> int:
    """Count possible old phase names NOT supported by the H185 scanner.

    Only immediate entry NAMES are examined, no file metadata or payload.
    A positive count is an explicit unresolved historical coverage gap.
    """
    count = 0
    for parent in parents:
        for child in parent.iterdir():
            possible = POSSIBLE_OLD.fullmatch(child.name)
            if (possible is not None and int(possible.group(1)) < 173
                    and NAME.fullmatch(child.name) is None):
                count += 1
    return count


def audit_expanded_owner_metadata(
    owner_root: Path, parent_roots: list[Path],
) -> dict:
    """Only aggregate denial, no positive training claim or SHA values."""
    require(isinstance(owner_root, Path) and owner_root.is_absolute()
            and _safe_directory(owner_root),
            "owner candidate root must be a safe absolute directory")
    phases = find_pre_h173_phases(parent_roots)
    # H204 independently ensures each old phase strictly pre-H173,
    # snapshots 20 current unadmitted cases, and checks their SHA-only
    # metadata against all discovered older manifests.
    receipt = training_readiness_gap(owner_root, phases)
    require(receipt["legacy_pre_h173_phases_scanned"] == len(phases)
            and receipt["new_unadmitted_original_gui_cases"] == 20
            and receipt["current_source_metadata_stable_across_audit"] is True
            and receipt["known_old_png_rgba_overlap_excluded_for_all_new_candidates"] is True,
            "expanded history comparison incomplete or candidates modified")
    require(receipt["train_admission_approved"] is False
            and receipt["dev_admission_approved"] is False
            and receipt["ood_admission_approved"] is False
            and receipt["complete_historical_old_rgba_proven"] is False
            and receipt["h173_consumed_ood_accessed"] is False
            and receipt["private_media_opened"] == 0
            and receipt["private_gold_opened"] == 0
            and receipt["private_artifacts_uploaded"] == 0
            and receipt["model_calls"] == 0
            and receipt["gpu_optimizer_steps"] == 0,
            "unqualified original source admission or private media access")
    return {
        "status": "H241_EXPANDED_PRE_H173_OWNER_SHA_EXCLUSIONS_ONLY_NO_ADMISSION",
        "pre_h173_phases_scanned": len(phases),
        "potential_old_phase_name_variants_unscanned":
            count_unscanned_old_name_variants(parent_roots),
        "historical_manifests_scanned": receipt["legacy_manifests_scanned"],
        "historical_manifests_missing_rgba_sha":
            receipt["legacy_manifests_lacking_rgba_sha"],
        "distinct_prior_png_sha_exclusions":
            receipt["distinct_known_legacy_png_sha_candidates"],
        "distinct_prior_rgba_sha_exclusions":
            receipt["distinct_known_legacy_rgba_sha_candidates"],
        "new_unadmitted_cases_compared":
            receipt["new_unadmitted_original_gui_cases"],
        "known_hash_collision_rejected_or_absent": True,
        "historical_history_complete": False,
        "visual_history_complete": False,
        "training_approved": False,
        "dev_approved": False,
        "ood_approved": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "private_media_opened": 0,
        "private_gold_opened": 0,
        "h173_ood_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", required=True, type=Path)
    parser.add_argument("--historical-parent", type=Path, action="append",
                        required=True)
    args = parser.parse_args()
    try:
        result = audit_expanded_owner_metadata(
            args.owner_root, args.historical_parent)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        parser.exit(2, "H241_OWNER_METADATA_NOT_QUALIFIED\n")
    # Only aggregate counts/status; NEVER print source names/paths, images,
    # raw SHA values, old gold or model predictions.
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
