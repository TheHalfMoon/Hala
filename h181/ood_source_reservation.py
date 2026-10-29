"""H181: one newly nominated original-app OOD SOURCE identity, never pixels.

Frappe Gantt is a fourth original public MIT Git family, distinct from
the three already source-smoked H174 donors. This step pins only public
source Git, HTML and LICENSE metadata. No renderer, screenshots, native
actions, benchmark tasks, source images, labels, predictions or model calls.

A source-only reservation is not a sealed/fresh OOD evaluation: old donor
families, forks, missing past source archives and unseen pixels need a
separate completed owner history audit before any training/evaluation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from h174.source_candidates import CANDIDATES
from h179.source_novelty import screen_pins_against_history

SCHEMA = "hala.h181.original-gantt-ood-source-identity-reservation.v1"
RESERVATION_FIELDS = frozenset({"schema", "app", "role", "source"})
SOURCE_ROLE = "OOD_SOURCE_RESERVED_NO_IMAGES"
GANTT_APP = "frappe-gantt"
SOURCE_FIELDS = frozenset({
    "repo", "git_sha", "html_path", "html_sha256",
    "license_path", "license_sha256", "license",
})
# Proven via immutable original GitHub API contents at exact 2026-10-29
# commit, independently of the live mutable master branch. Original root
# index.html is a source file ONLY; it must not be rendered in this phase.
GANTT_PIN = {
    "repo": "frappe/gantt",
    "git_sha": "5c83708d300a99d9142edea0d4d52f421e3a2e68",
    "html_path": "index.html",
    "html_sha256": "6c86478a5654910e8ba31bc6cbbe5ef3edf55eced03d2d31d0f654fc70c04065",
    "license_path": "license.txt",
    "license_sha256": "f0ee9b1340ea287eacdc522f04f85b80304c2009c237809920ff9cf7684fc4f1",
    "license": "MIT",
}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H181 source-only reservation: " + reason)


def preregistered_identity() -> dict:
    """Return only immutable source IDs, never an OOD task or scene."""
    return {
        "schema": SCHEMA,
        "app": GANTT_APP,
        "role": SOURCE_ROLE,
        "source": dict(GANTT_PIN),
    }


def check_reserved_source_only(
    candidate: dict, historical_roots: list[str | Path],
) -> dict:
    """Fail closed if any candidate pin or old local history disagrees.

    A successful source-identity check MUST NOT trigger OOD capture, create
    training files, launch a browser, or change a dataset/model state.
    """
    _require(
        type(candidate) is dict and set(candidate) == RESERVATION_FIELDS
        and candidate["schema"] == SCHEMA
        and candidate["app"] == GANTT_APP
        and candidate["role"] == SOURCE_ROLE,
        "exact OOD source-only identity schema required",
    )
    source = candidate["source"]
    _require(
        type(source) is dict and set(source) == SOURCE_FIELDS
        and source == GANTT_PIN,
        "immutable original Gantt Git/HTML/LICENSE pin mismatch",
    )
    # Distinct from the 3 sources already researched/smoked in H174/H180.
    # Their future use as TRAIN/DEV remains unapproved and never influences
    # this reserved original OOD source.
    prospective = {
        item["repo"].casefold() for item in CANDIDATES.values()
    }
    _require(
        len(prospective) == 3
        and source["repo"].casefold() not in prospective
        and source["git_sha"] not in {
            item["git_sha"] for item in CANDIDATES.values()
        }
        and source["html_sha256"] not in {
            item["html_sha256"] for item in CANDIDATES.values()
        },
        "reserved OOD source overlaps exposed donor candidate",
    )
    history = screen_pins_against_history(
        {GANTT_APP: source}, historical_roots
    )
    canonical = json.dumps(
        candidate, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf8")
    return {
        "status": "H181_ORIGINAL_GANTT_SOURCE_RESERVED_ONLY_NOT_OOD",
        "reserved_original_repo": source["repo"],
        "reserved_source_git_sha": source["git_sha"],
        "source_reservation_fingerprint_sha256": sha256(canonical).hexdigest(),
        "owner_historical_git_repositories_screened":
            history["historical_unique_repositories_screened"],
        "owner_historical_git_checkouts_screened":
            history["historical_git_checkouts_screened"],
        "previous_original_app_sources_in_this_branch":
            len(prospective),
        "source_history_completeness_proven": False,
        "source_fork_equivalence_ruled_out": False,
        "training_data_admitted": False,
        "dev_data_admitted": False,
        "ood_images_created": 0,
        "ood_model_inputs_created": 0,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }
