"""H179 candidate-source novelty screen against on-device historical Git roots.

Strictly source identity and original immutable file bytes; zero image, gold,
prediction, model or dataset calls. A source identity PASS neither verifies
historical completeness nor authorizes any TRAIN/DEV/OOD admission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h173.source_preflight import SOURCE_PINS
from h174.source_candidates import CANDIDATES, verify_catalog, verify_local_source_only
from h174.split_firewall import APP, GIT, HEX, REPO, BANNED
from h176.source_family_attribution import (
    FROZEN_H173_REPOS, FROZEN_OTHER_GIT, KNOWN_EXTRA_PREVIOUS,
    KNOWN_NPM_UPSTREAM_REPOS,
)
from h177.public_origin_pins import KNOWN_PUBLIC_REPOS
from h178.owner_git_inventory import inventory_local_original_sources
from h184.public_prior_considered_origins import (
    BLOCKED_REPOSITORIES, BLOCKED_COMMITS, validate_public_previous_ledger,
)

MAX_CANDIDATES = 100
PIN_FIELDS = frozenset({
    "repo", "git_sha", "html_path", "html_sha256",
    "license_path", "license_sha256", "license",
})
# Even if owner-local historical Git checkouts are absent/deleted, these
# published, previously consumed source roots remain disallowed.
KNOWN_BURNED_REPOS = (
    KNOWN_PUBLIC_REPOS
    | KNOWN_NPM_UPSTREAM_REPOS
    | frozenset(FROZEN_H173_REPOS.values())
    | BLOCKED_REPOSITORIES
)
KNOWN_BURNED_GIT = (
    frozenset(p["git_sha"] for p in SOURCE_PINS.values())
    | frozenset(FROZEN_OTHER_GIT.values())
    | BLOCKED_COMMITS
)
KNOWN_BURNED_HTML = frozenset(
    p["original_html_sha256"] for p in SOURCE_PINS.values()
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H179 owner novel original source: " + reason)


def screen_pins_against_history(
    pins: dict, historical_source_roots: list[str | Path],
) -> dict:
    """Screen immutable source IDs only; do not trust this as admission.

    A source can have been used historically without a surviving checkout,
    and forks may share source even when repository names differ.
    """
    validate_public_previous_ledger()
    _require(
        type(pins) is dict and 0 < len(pins) <= MAX_CANDIDATES,
        "bounded source-only catalog required",
    )
    history = inventory_local_original_sources(historical_source_roots)
    observed = history["_owner_only_repository_shas"]
    observed_repos = {repo for repo, _ in observed}
    observed_gits = {git for _, git in observed}
    seen_repos = set()
    seen_git = set()
    seen_html = set()
    for app, item in pins.items():
        _require(
            type(app) is str and APP.fullmatch(app) is not None
            and app not in (BANNED | KNOWN_EXTRA_PREVIOUS),
            "invalid or previously used app identifier",
        )
        _require(
            type(item) is dict and set(item) == PIN_FIELDS,
            "only exact pinned original source fields permitted",
        )
        repo, git, html, lic = (
            item["repo"], item["git_sha"], item["html_sha256"],
            item["license_sha256"],
        )
        _require(
            type(repo) is str and REPO.fullmatch(repo) is not None
            and all(part not in (".", "..") for part in repo.split("/")),
            "invalid Git upstream",
        )
        _require(type(git) is str and GIT.fullmatch(git) is not None,
                 "original source Git SHA missing")
        _require(type(html) is str and HEX.fullmatch(html) is not None
                 and type(lic) is str and HEX.fullmatch(lic) is not None,
                 "original source HTML/license SHA missing")
        _require(item["license"] == "MIT", "MIT donor license pin required")
        for path_key in ("html_path", "license_path"):
            path = item[path_key]
            _require(
                type(path) is str and path
                and not Path(path).is_absolute()
                and all(part not in ("", ".", "..") for part in path.replace(
                    "\\", "/"
                ).split("/")),
                "unsafe original source relative path",
            )
        canonical_repo = repo.casefold()
        _require(
            canonical_repo not in (KNOWN_BURNED_REPOS | observed_repos),
            "candidate upstream previously observed in historical app",
        )
        _require(
            git not in (KNOWN_BURNED_GIT | observed_gits)
            and html not in KNOWN_BURNED_HTML,
            "candidate original source bytes historically reused",
        )
        _require(
            canonical_repo not in seen_repos
            and git not in seen_git and html not in seen_html,
            "duplicate candidate family/Git/HTML source",
        )
        seen_repos.add(canonical_repo)
        seen_git.add(git)
        seen_html.add(html)

    candidate_ids = sorted((repo.casefold(), pins[app]["git_sha"])
                           for app, repo in ((app, pin["repo"])
                                             for app, pin in pins.items()))
    fingerprint = hashlib.sha256(json.dumps(
        candidate_ids, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "status": "H179_SOURCE_IDENTITY_SCREEN_ONLY_NOT_ADMISSION",
        "candidate_source_families": len(pins),
        "historical_git_checkouts_screened": history["source_checkouts_inspected"],
        "historical_unique_repositories_screened": history["distinct_repositories"],
        "candidate_source_fingerprint_sha256": fingerprint,
        "historical_identity_fingerprint_sha256":
            history["source_identity_fingerprint_sha256"],
        "source_history_completeness_proven": False,
        "extra_public_historic_source_families_blocked": len(BLOCKED_REPOSITORIES),
        "source_native_qa_qualified": False,
        "train_dev_ood_assigned": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }


def screen_h174_candidates(
    historical_source_roots: list[str | Path],
    pristine_donor_roots: dict[str, str | Path] | None = None,
) -> dict:
    """Evaluate fixed H174 candidates; optional three original Git byte checks.

    No code paths here generate training data, start browsers, or score OOD.
    """
    verify_catalog()
    result = screen_pins_against_history(CANDIDATES, historical_source_roots)
    _require(
        type(pristine_donor_roots) in (dict, type(None)),
        "invalid owner donor root dictionary",
    )
    if pristine_donor_roots is None:
        count = 0
    else:
        _require(
            set(pristine_donor_roots) == set(CANDIDATES),
            "all three original donor roots required together",
        )
        for app in sorted(CANDIDATES):
            proof = verify_local_source_only(app, pristine_donor_roots[app])
            _require(
                proof["source_git_sha"] == CANDIDATES[app]["git_sha"]
                and proof["original_html_sha256"]
                == CANDIDATES[app]["html_sha256"],
                "original donor byte identity mismatch",
            )
        count = len(CANDIDATES)
    result["independent_local_source_byte_pins_verified"] = count
    result["status"] = "H179_H174_PINNED_CANDIDATES_ONLY_NO_SPLIT_OR_MODEL"
    return result
