"""H174 owner-supplied historical SOURCE identity crosscheck. Metadata only.

The existing H174 v2 split firewall proves new TRAIN/DEV/OOD_RESERVED
families do not overlap *each other*, but cannot see arbitrary historical
repository aliases under different Git revisions. This additional gate
requires an owner-only historical identity manifest without reading ANY
previously consumed images, gold or predictions.

No input in this module is proof that the caller's history is complete.
Never use a PASS from this module alone to qualify a dataset or model.
"""
import hashlib
import json

from h174.split_firewall import (
    APP, BANNED, BANNED_GIT, BANNED_HTML, GIT, HEX, REPO, validate,
)

SCHEMA = "hala.h174.owner-prior-source-identity-history.v1"
FIELDS = frozenset({
    "schema", "historical_app_ids", "historical_repositories",
    "source_git_shas", "source_html_shas",
})
MAX_ENTRIES = 100_000

# Independently pinned consumed original H173 upstream families (source only).
# Exact public commits: robinparisi/tingle@9eadca70c72ef9adb9144cc23a4104978d3bc63b
# and ganlanyuan/tiny-slider@4d709735c417c2483e77a22d017fc1b18c04f0d4.
# This mandatory floor does NOT prove the complete older TRAIN/DEV/OOD history.
KNOWN_H173_REPOSITORIES = frozenset({
    "robinparisi/tingle", "ganlanyuan/tiny-slider",
})


def _require(condition, why):
    if not condition:
        raise ValueError("H174 historical source gate: " + why)


def _validate_array(value, label, pattern, *, nonempty=True):
    _require(type(value) is list and (bool(value) or not nonempty)
             and len(value) <= MAX_ENTRIES, label + " must be a bounded array")
    _require(all(type(item) is str and pattern.fullmatch(item) is not None
                 for item in value), label + " contains invalid identities")
    _require(len(set(value)) == len(value), label + " contains duplicates")
    return value


def validate_with_source_history(metadata, owner_hash_registry, source_history):
    """Reject historical source FAMILY reuse, without reading old media.

    The owner must assemble historical_repositories from prior immutable
    source manifests, not infer old sources from images or evaluations.
    This supplements, never weakens, the H174 v2 metadata firewall.
    """
    result = validate(metadata, owner_hash_registry)
    _require(type(source_history) is dict and set(source_history) == FIELDS,
             "exact SHA-only/source-identity fields required; no gold or pixels")
    _require(source_history["schema"] == SCHEMA, "wrong historical identity schema")

    apps = _validate_array(source_history["historical_app_ids"], "app IDs", APP)
    repos = _validate_array(
        source_history["historical_repositories"], "repositories", REPO
    )
    gits = _validate_array(source_history["source_git_shas"], "source Git SHAs", GIT)
    htmls = _validate_array(
        source_history["source_html_shas"], "original HTML SHA-256", HEX
    )
    _require(all(all(part not in (".", "..") for part in repo.split("/"))
                 for repo in repos), "invalid historical upstream repository path")

    # Case variants of one upstream repository are the SAME donor family.
    repository_ids = {repo.casefold() for repo in repos}
    _require(len(repository_ids) == len(repos),
             "historical repositories contain case-variant aliases")
    _require(KNOWN_H173_REPOSITORIES.issubset(repository_ids),
             "known consumed H173 upstream repository family missing from registry")
    # These known H173 and prior names/hashes must never disappear from
    # a claimed history. Additional historical sources remain owner supplied.
    _require(BANNED.issubset(set(apps)),
             "known historical TRAIN/DEV/OOD application missing from registry")
    _require(BANNED_GIT.issubset(set(gits)) and
             BANNED_HTML.issubset(set(htmls)),
             "known frozen H173 source SHA missing from registry")

    for source in metadata["sources"]:
        _require(source["app"] not in apps,
                 "historical source app name reused")
        _require(source["repo"].casefold() not in repository_ids,
                 "historical upstream repository family reused under new SHA")
        _require(source["git_sha"] not in gits,
                 "historical original source Git SHA reused")
        _require(source["html_sha256"] not in htmls,
                 "historical original source HTML bytes reused")

    # A reproducible *identity* of the caller's supplied metadata, NOT an
    # attestation of completeness, origin, authenticity or source license.
    canonical = {key: sorted(source_history[key])
                 for key in FIELDS if key != "schema"}
    canonical["schema"] = SCHEMA
    fingerprint = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "status": "H174_SOURCE_HISTORY_CROSSCHECK_ONLY_NOT_DATASET_ADMISSION",
        "owner_source_registry_fingerprint_sha256": fingerprint,
        "historical_repositories_compared": len(repos),
        "historical_app_ids_compared": len(apps),
        "source_history_completeness_proven": False,
        "original_app_native_qa_proven": False,
        "training_approved": False,
        "model_calls": result["model_calls"],
        "native_rewards": result["native_rewards"],
    }
