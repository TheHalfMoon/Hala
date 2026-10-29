"""H176: paired historical original-source family metadata crosscheck.

The H174 legacy five-array history identifies names, repositories and hashes,
but *does not associate them*. A completely fabricated single repository could
therefore satisfy the legacy minimum. This optional stricter audit makes every
owner-supplied app correspond to an exact repo/Git/HTML identity and checks the
H173 frozen Git/HTML pairs. Nothing here reads historical pixels, labels or
predictions, and no return value authorizes admission, training or an OOD score.

Owner-only: never check actual owner historical manifests into public GitHub.
"""
from __future__ import annotations

from hashlib import sha256
import json

from h173.source_preflight import SOURCE_PINS
from h174.source_history_gate import (
    SCHEMA as HISTORY_SCHEMA, MAX_ENTRIES, validate_with_source_history,
)
from h174.split_firewall import APP, BANNED, GIT, HEX, REPO

SCHEMA = "hala.h176.owner-historical-source-family-attribution.v1"
TOP_FIELDS = frozenset({"schema", "families"})
ROW_FIELDS = frozenset({"app", "repo", "git_sha", "html_sha256"})


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("H176 family attribution firewall: " + reason)


def audit_attribution(
    metadata: dict,
    owner_hash_registry: dict,
    source_history: dict,
    owner_attribution: dict,
) -> dict:
    """Crosscheck legacy historical sets against paired owner-side source IDs.

    This verifies only the *internal consistency* of supplied metadata, not
    its completeness or authenticity. Historic source-family completeness must
    be verified independently against every previous TRAIN/DEV/OOD source.
    """
    legacy = validate_with_source_history(
        metadata, owner_hash_registry, source_history
    )
    _require(
        type(owner_attribution) is dict
        and set(owner_attribution) == TOP_FIELDS
        and owner_attribution["schema"] == SCHEMA,
        "exact metadata-only attribution schema required",
    )
    rows = owner_attribution["families"]
    _require(
        type(rows) is list and 0 < len(rows) <= MAX_ENTRIES,
        "bounded nonempty original source family rows required",
    )

    apps: set[str] = set()
    repos: set[str] = set()
    git_shas: set[str] = set()
    html_shas: set[str] = set()
    canonical_rows: list[dict] = []
    pinned_pairs: set[str] = set()

    for row in rows:
        _require(
            type(row) is dict and set(row) == ROW_FIELDS,
            "only app/repo/Git/HTML source identities permitted",
        )
        app, repo, git_sha, html = (
            row["app"], row["repo"], row["git_sha"], row["html_sha256"]
        )
        _require(
            type(app) is str and APP.fullmatch(app) is not None,
            "malformed original source app",
        )
        _require(
            type(repo) is str and REPO.fullmatch(repo) is not None
            and all(part not in (".", "..") for part in repo.split("/")),
            "malformed original upstream repository",
        )
        _require(
            type(git_sha) is str and GIT.fullmatch(git_sha) is not None
            and type(html) is str and HEX.fullmatch(html) is not None,
            "exact source Git/HTML hashes required per historical family",
        )
        _require(app not in apps, "duplicate historical application attribution")
        # Multiple historical app IDs may share a real upstream repo; they
        # still constitute ONE source family and cannot become new donors.
        apps.add(app)
        repos.add(repo.casefold())
        git_shas.add(git_sha)
        html_shas.add(html)
        canonical_rows.append({
            "app": app, "repo": repo.casefold(),
            "git_sha": git_sha, "html_sha256": html,
        })
        if app in SOURCE_PINS:
            frozen = SOURCE_PINS[app]
            _require(
                git_sha == frozen["git_sha"]
                and html == frozen["original_html_sha256"],
                "H173 frozen original app Git/HTML pair mismatch",
            )
            pinned_pairs.add(app)

    _require(
        BANNED.issubset(apps) and set(SOURCE_PINS) == pinned_pairs,
        "missing known TRAIN/DEV/consumed OOD or H173 source pairing",
    )
    _require(
        apps == set(source_history["historical_app_ids"]),
        "historical application attribution and legacy roster disagree",
    )
    _require(
        repos == {r.casefold() for r in source_history["historical_repositories"]},
        "historical upstream repository attribution and legacy roster disagree",
    )
    _require(
        git_shas == set(source_history["source_git_shas"]),
        "historical Git SHA attribution and legacy registry disagree",
    )
    _require(
        html_shas == set(source_history["source_html_shas"]),
        "historical original HTML SHA attribution and legacy registry disagree",
    )
    canonical_rows.sort(key=lambda row: row["app"])
    fingerprint = sha256(json.dumps(
        {"schema": SCHEMA, "families": canonical_rows},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()

    return {
        "status": "H176_PAIRED_HISTORICAL_METADATA_ONLY_NOT_ADMISSION",
        "paired_source_families": len(apps),
        "distinct_historical_repositories": len(repos),
        "frozen_h173_pairs_matched": len(pinned_pairs),
        "attribution_sha256": fingerprint,
        "legacy_history_sha256": legacy["owner_source_registry_fingerprint_sha256"],
        "source_history_completeness_proven": False,
        "provenance_authenticity_proven": False,
        "model_training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }
