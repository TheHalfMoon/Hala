"""H182 source-disjoint research role preregistration (NO dataset admission).

These SOURCE roles are prospective only. They never authorize collection
of screenshots, images or reference labels, evaluation, or model training.
The original Gantt OOD candidate MUST remain completely untouched by any
GUI build/visit, and H173 consumed original OOD is irrevocably banned.

Copy-first composition of immutable H174/H179/H181 source pin checks.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from h174.source_candidates import CANDIDATES, verify_catalog
from h179.source_novelty import screen_h174_candidates
from h181.ood_source_reservation import (
    GANTT_APP, GANTT_PIN, SOURCE_ROLE, check_reserved_source_only,
    preregistered_identity,
)

SCHEMA = "hala.h182.source-identity-role-preregistration.no-cases.v1"
TOP_FIELDS = frozenset({"schema", "roles"})
ROLE_FIELDS = frozenset({"app", "role", "source"})
SPLITS = frozenset({
    "PROSPECTIVE_TRAIN_NO_IMAGES",
    "PROSPECTIVE_DEV_NO_IMAGES",
    "RESEARCH_EXPOSED_UNASSIGNED",
    SOURCE_ROLE,
})
SPEC = (
    ("pickr", "PROSPECTIVE_TRAIN_NO_IMAGES"),
    ("interact-dropzones", "PROSPECTIVE_DEV_NO_IMAGES"),
    ("medium-zoom", "RESEARCH_EXPOSED_UNASSIGNED"),
    (GANTT_APP, SOURCE_ROLE),
)


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H182 source-only role plan: " + why)


def preregistered_plan() -> dict:
    """Public immutable Git/HTML/LICENSE source IDs; no screenshots/cases."""
    return {
        "schema": SCHEMA,
        "roles": [
            {
                "app": app,
                "role": role,
                "source": dict(
                    GANTT_PIN if app == GANTT_APP else CANDIDATES[app]
                ),
            }
            for app, role in SPEC
        ],
    }


def audit_preregistered_sources(
    plan: dict,
    historical_roots: list[str | Path],
    pinned_donor_roots: dict[str, str | Path] | None = None,
) -> dict:
    """Owner original-source preflight; *cannot* return dataset admission.

    Performs the H179 three-donor native original source *byte* checks when
    pinned donor roots are supplied, and H181 Gantt source-only reservation.
    Does not open the Gantt source tree, image files, labels or benchmark gold.
    """
    verify_catalog()
    _require(
        type(plan) is dict and set(plan) == TOP_FIELDS
        and plan.get("schema") == SCHEMA,
        "unknown top-level fields, case data or incorrect schema",
    )
    rows = plan.get("roles")
    _require(type(rows) is list and len(rows) == 4,
             "exact four original source roles required")
    _require(
        all(type(row) is dict and set(row) == ROLE_FIELDS for row in rows),
        "role metadata only; no cases, media, OOD or gold fields",
    )
    expected = preregistered_plan()
    _require(
        rows == expected["roles"],
        "immutable original source roles, order and Git/HTML/license pins required",
    )
    repos = [row["source"]["repo"].casefold() for row in rows]
    gits = [row["source"]["git_sha"] for row in rows]
    htmls = [row["source"]["html_sha256"] for row in rows]
    _require(
        len(set(repos)) == len(rows)
        and len(set(gits)) == len(rows)
        and len(set(htmls)) == len(rows),
        "TRAIN DEV research and reserved OOD identities must be disjoint",
    )
    _require(
        {row["role"] for row in rows} == SPLITS
        and rows[-1]["role"] == SOURCE_ROLE,
        "source roles must be one each and OOD source-only",
    )

    original = screen_h174_candidates(
        historical_roots, pristine_donor_roots=pinned_donor_roots
    )
    reserved = check_reserved_source_only(
        preregistered_identity(), historical_roots
    )
    _require(
        original["candidate_source_families"] == 3
        and reserved["previous_original_app_sources_in_this_branch"] == 3,
        "original source families not independently preregistered",
    )
    _require(
        reserved["owner_historical_git_checkouts_screened"]
        == original["historical_git_checkouts_screened"]
        and reserved["owner_historical_git_repositories_screened"]
        == original["historical_unique_repositories_screened"]
        and reserved["owner_historical_identity_fingerprint_sha256"]
        == original["historical_identity_fingerprint_sha256"],
        "owner historical source inventory drift between gates",
    )
    # The owner manifests and source-root census cannot establish historical
    # completeness: the returned receipt is *always* no-admission.
    digest = sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")).hexdigest()
    return {
        "status": "H182_PROSPECTIVE_SOURCE_ROLES_ONLY_NOT_DATASET_ADMISSION",
        "prospective_train_source_count": 1,
        "prospective_dev_source_count": 1,
        "exposed_research_unassigned_source_count": 1,
        "tentatively_reserved_ood_source_count": 1,
        "pinned_pristine_original_donor_files_verified":
            original["independent_local_source_byte_pins_verified"],
        "owner_old_git_checkouts_scanned":
            original["historical_git_checkouts_screened"],
        "source_role_plan_fingerprint_sha256": digest,
        "owner_historical_fingerprint_sha256":
            original["historical_identity_fingerprint_sha256"],
        "complete_history_proven": False,
        "historical_pixel_sha_coverage_proven": False,
        "original_fork_equivalence_proven_disjoint": False,
        "native_train_dev_dataset_qualified": False,
        "train_images_created": 0,
        "dev_images_created": 0,
        "ood_images_created": 0,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }
