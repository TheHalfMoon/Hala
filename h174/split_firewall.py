"""H174 source-only TRAIN/DEV split firewall. No pixels, models, or owner gold.

This is a proposed metadata preflight, NOT a qualified training dataset or
new OOD score. Copy-first: historical source identity registry is reused
from h173.source_preflight; actual owner-only SHA registry is caller-supplied.
Never read historical consumed DEV/OOD media to construct that registry.
"""
import re
from h173.source_preflight import TRAIN_AND_CONSUMED_FAMILIES, SOURCE_PINS

SCHEMA = "hala.h174.new-original-app-train-dev-metadata-preflight.v2"
HISTORY_SCHEMA = "hala.h174.owner-only-prior-sha-registry.v1"
SPLITS = frozenset({"TRAIN", "DEV", "OOD_RESERVED"})
BANNED = frozenset(TRAIN_AND_CONSUMED_FAMILIES) | frozenset(SOURCE_PINS)
BANNED_GIT = frozenset(pin["git_sha"] for pin in SOURCE_PINS.values())
BANNED_HTML = frozenset(pin["original_html_sha256"] for pin in SOURCE_PINS.values())
HEX = re.compile(r"[0-9a-f]{64}\Z")
GIT = re.compile(r"[0-9a-f]{40}\Z")
APP = re.compile(r"[a-z][a-z0-9-]{1,55}\Z")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
# Source-family floor recoverable from PUBLIC H168/H169/H171/H172 evidence.
# H168 TRAIN + H169 DEV: Choices.js @e132628ac74dd427a9a454caf042c65d3065d170,
# noUiSlider @57033672d07e8bac527d7147721a4a2e07e0d699.
# H171 TRAIN + H172 DEV: flatpickr @548330e6e18871d6ca1ca78746c0e89cb12a0303,
# Awesomplete @596bc5d5916266437970ecbf65434077d3950a3d.
# Awesomplete's historic LeaVerou/awesomplete and nudeui/awesomplete are
# ONE GitHub repository (ID 30987756). BOTH aliases must be blocked.
# This floor is incomplete with respect to private unlisted historical data;
# adding these names does NOT authorize any new dataset or model training.
FROZEN_PRIOR_ORIGINAL_REPOS = frozenset({
    "choices-js/choices",
    "leongersen/nouislider",
    "flatpickr/flatpickr",
    "leaverou/awesomplete",
    "nudeui/awesomplete",
})
SOURCE_KEYS = frozenset({
    "app", "split", "repo", "git_sha", "html_sha256", "license",
})
CASE_KEYS = frozenset({
    "index", "app", "split", "image_sha256", "visual_rgba_sha256",
    "language", "action_type", "instruction",
})


def _digest(value):
    return type(value) is str and HEX.fullmatch(value) is not None


def _require(condition, message):
    if not condition:
        raise ValueError("H174 metadata firewall: " + message)


def canonical_repository_family(repo: str) -> str:
    """Resolve GitHub clone .git suffix aliases to ONE source family.

    Canonicalization is a conservative duplicate/old-source firewall,
    not provenance authenticity or an approval for TRAIN/DEV/OOD.
    Multiple .git clone suffixes cannot create distinct donors.
    """
    _require(type(repo) is str and REPO.fullmatch(repo) is not None
             and all(part not in (".", "..") for part in repo.split("/")),
             "invalid upstream repository identity")
    canonical = repo.casefold()
    while canonical.endswith(".git"):
        canonical = canonical[:-4]
    _require(canonical.split("/")[1] not in ("", ".", ".."),
             "invalid upstream repository identity after clone alias")
    return canonical


def validate(metadata, owner_hash_registry):
    """Check only NEW metadata; consumed benchmark pixels are forbidden.

    OOD_RESERVED is a SOURCE-only reservation, never images or cases.
    This check alone cannot prove upstream donor licenses, hash registry
    completeness, native browser success or new OOD generalization.
    """
    _require(type(metadata) is dict and set(metadata) ==
             {"schema", "sources", "cases"}, "unrecognized top-level fields")
    _require(metadata["schema"] == SCHEMA, "wrong candidate schema")
    _require(type(owner_hash_registry) is dict and
             set(owner_hash_registry) == {"schema", "image_shas", "visual_shas"},
             "owner historical SHA-only registry required")
    _require(owner_hash_registry["schema"] == HISTORY_SCHEMA,
             "owner history schema mismatch")
    old_images = owner_hash_registry["image_shas"]
    old_visible = owner_hash_registry["visual_shas"]
    _require(type(old_images) is list and type(old_visible) is list and
             bool(old_images) and bool(old_visible), "nonempty historical hashes required")
    _require(all(_digest(h) for h in old_images + old_visible),
             "historical hashes invalid")
    _require(len(set(old_images)) == len(old_images) and
             len(set(old_visible)) == len(old_visible),
             "duplicate historical hashes")
    sources, cases = metadata["sources"], metadata["cases"]
    _require(type(sources) is list and type(cases) is list,
             "sources/cases must be arrays")
    _require(len(sources) >= 3 and bool(cases),
             "TRAIN, DEV and reserved OOD sources required")
    indexed = {}
    git_seen, html_seen = set(BANNED_GIT), set(BANNED_HTML)
    repo_seen = set()
    by_split = {split: set() for split in SPLITS}
    for s in sources:
        _require(type(s) is dict and set(s) == SOURCE_KEYS,
                 "source must contain only allowed provenance")
        app, split = s["app"], s["split"]
        _require(type(app) is str and APP.fullmatch(app) is not None,
                 "invalid original app identifier")
        repo = s["repo"]
        _require(type(repo) is str and REPO.fullmatch(repo) is not None
                 and all(part not in (".", "..") for part in repo.split("/")),
                 "invalid upstream repository identity")
        canonical_repo = canonical_repository_family(repo)
        _require(canonical_repo not in FROZEN_PRIOR_ORIGINAL_REPOS,
                 "previous original TRAIN/DEV source family reused")
        from h238.historical_family_floor import known_public_source_family_floor
        _require(canonical_repo not in known_public_source_family_floor(),
                 "public historical original source family reused")
        _require(canonical_repo not in repo_seen,
                 "duplicate upstream repository across TRAIN DEV or OOD")
        repo_seen.add(canonical_repo)
        _require(app not in BANNED, "historical source app is forbidden")
        _require(type(split) is str and split in SPLITS,
                 "invalid source split")
        _require(app not in indexed, "same original app appears in multiple splits")
        _require(type(s["git_sha"]) is str and GIT.fullmatch(s["git_sha"]) is not None,
                 "exact original Git SHA required")
        _require(_digest(s["html_sha256"]), "source HTML SHA required")
        _require(s["git_sha"] not in git_seen and
                 s["html_sha256"] not in html_seen,
                 "historical or aliased duplicate original donor source")
        git_seen.add(s["git_sha"])
        html_seen.add(s["html_sha256"])
        _require(s["license"] in ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"),
                 "permissive license claim required")
        indexed[app] = split
        by_split[split].add(app)
    _require(all(by_split.values()), "TRAIN DEV and OOD reservation must be source-disjoint")
    raw_seen, rgba_seen, indexes = set(old_images), set(old_visible), set()
    counts = {"TRAIN": 0, "DEV": 0}
    for c in cases:
        _require(type(c) is dict and set(c) == CASE_KEYS,
                 "gold, reference coordinates or unknown model fields prohibited")
        split, app = c["split"], c["app"]
        _require(type(split) is str and type(app) is str and
                 split in counts and indexed.get(app) == split,
                 "no OOD images or cross-split source cases allowed")
        _require(type(c["index"]) is int and c["index"] >= 0 and
                 c["index"] not in indexes, "duplicate/invalid scene index")
        indexes.add(c["index"])
        raw, rgba = c["image_sha256"], c["visual_rgba_sha256"]
        _require(_digest(raw) and _digest(rgba), "PNG/RGBA SHA required")
        _require(raw not in raw_seen and rgba not in rgba_seen,
                 "historical or cross-case visual/PNG overlap")
        raw_seen.add(raw)
        rgba_seen.add(rgba)
        _require(c["language"] in ("en", "ar") and
                 c["action_type"] in ("click", "type", "drag"),
                 "unqualified language or action type")
        _require(type(c["instruction"]) is str and
                 len(c["instruction"].strip()) >= 8, "real task instruction missing")
        counts[split] += 1
    _require(all(counts.values()), "TRAIN and DEV cases both required")
    return {
        "status": "H174_METADATA_ONLY_SPLIT_FIREWALL_PASS_NOT_MODEL_QUALIFIED",
        "train_cases": counts["TRAIN"], "dev_cases": counts["DEV"],
        "reserved_ood_app_count": len(by_split["OOD_RESERVED"]),
        "history_shas_inspected_as_metadata_only": len(old_images) + len(old_visible),
        "model_calls": 0, "native_rewards": 0,
    }
