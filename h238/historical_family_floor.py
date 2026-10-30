"""H238: union of existing PUBLIC original-source historical donor identity floors.

This module only combines already-published source metadata. It neither
claims completeness nor reads owner image, gold, historical private manifest,
predictions, or model weights. A PASS is never data/model admission.
"""
from __future__ import annotations


def known_public_source_family_floor() -> frozenset[str]:
    """Fail-closed union of source families from H173/H176/H177/H184/H237.

    Imports are intentionally deferred: H176/H177 themselves depend on H174.
    Calling this inside H174.validate would otherwise create a module-init
    cycle. Keep all source family aliases normalized via the H174 canonical
    function, including case changes and repeated trailing .git suffixes.
    """
    from h174.split_firewall import (
        FROZEN_PRIOR_ORIGINAL_REPOS, canonical_repository_family,
    )
    from h176.source_family_attribution import (
        FROZEN_H173_REPOS, FROZEN_OTHER_REPOS, KNOWN_NPM_UPSTREAM_REPOS,
    )
    from h177.public_origin_pins import KNOWN_PUBLIC_REPOS
    from h184.public_prior_considered_origins import (
        BLOCKED_REPOSITORIES, validate_public_previous_ledger,
    )

    validate_public_previous_ledger()
    sources = (
        set(FROZEN_PRIOR_ORIGINAL_REPOS)
        | set(FROZEN_H173_REPOS.values())
        | set(FROZEN_OTHER_REPOS.values())
        | set(KNOWN_NPM_UPSTREAM_REPOS)
        | set(KNOWN_PUBLIC_REPOS)
        | set(BLOCKED_REPOSITORIES)
    )
    floor = frozenset(canonical_repository_family(repo) for repo in sources)
    if len(floor) < 27:
        raise ValueError("H238 fail-closed: public historical source floor contracted")
    return floor


def public_source_floor_receipt() -> dict:
    """Aggregate only. No historical owner paths, hashes or private metadata."""
    floor = known_public_source_family_floor()
    return {
        "status": "H238_PUBLIC_HISTORY_MINIMUM_ONLY_NOT_ADMISSION",
        "public_distinct_original_source_families_blocked": len(floor),
        "complete_historical_source_inventory_proven": False,
        "complete_previous_png_rgba_registry_proven": False,
        "source_license_and_transitive_media_rights_verified": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }
