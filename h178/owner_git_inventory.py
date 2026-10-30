"""H178 local, read-only historical Git *source identity* census.

Scan only private-hNN/((upstream|vendor-apps|vendor|sources|donors))/NAME/.git
(and explicitly named Hala-HNN-Private roots). Git commands read repository
identity/config only. Never enumerate old PNG/JSON results, gold, predictions,
weights, datasets, screenshots, notebook outputs or benchmark directories.
No network, no third-party API, no writes, no model inference, no GPU.

This finds *present* historical source checkouts, NOT all once-used sources;
missing or deleted checkouts prevent a historical-completeness claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from h174.split_firewall import canonical_repository_family
from h176.source_family_attribution import KNOWN_NPM_UPSTREAM_REPOS
from h177.public_origin_pins import audit_public_origin_pins

# Explicitly shallow only; NEVER recurse through source media or private data.
SOURCE_STORAGE_NAMES = frozenset({
    "upstream", "vendor-apps", "vendor", "sources", "donors",
})
PHASE_NAME = re.compile(
    r"(?:private-h[0-9]{2,3}|Hala-H[0-9]{2,3}-Private)\Z",
    re.IGNORECASE,
)
GITHUB_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
MAX_PHASES = 100
MAX_CHECKOUTS = 400


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H178 private original-source inventory: " + why)


def _safe_directory(path: Path) -> bool:
    """Reject symlinks, Windows junctions and other reparse paths, including ancestors."""
    if not path.is_absolute():
        return False
    for node in (path, *path.parents):
        try:
            st = node.lstat()
        except (OSError, ValueError):
            return False
        if stat.S_ISLNK(st.st_mode) or (
            getattr(st, "st_file_attributes", 0) & 0x400
        ):
            return False
    return path.is_dir()


def _git(path: Path, *args: str) -> str:
    # Each command is offline/read-only and has a short timeout. Never emit
    # stderr, which might contain a user-private absolute path or URL.
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True,
            text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("local original Git identity unavailable") from None
    _require(proc.returncode == 0, "local original Git identity unavailable")
    return proc.stdout.strip()


def _github_root(origin: str) -> str:
    """Allow only canonical public GitHub remotes, not arbitrary source URLs."""
    # Deliberately reject URLs carrying credentials; neither trust nor print
    # embedded user-info, alternate hosts, HTTP URLs or Git config redirects.
    if origin.startswith("https://github.com/"):
        path = origin[len("https://github.com/"):]
    elif origin.startswith("git@github.com:"):
        path = origin[len("git@github.com:"):]
    else:
        raise ValueError("untrusted historical Git origin scheme or host")
    # H174/H226/H238 treat repeated .git clone suffixes as a single
    # GitHub family. An owner-old remote with .git.git must not evade
    # an admission candidate using the unsuffixed or once-suffixed name.
    path = path.rstrip("/")
    _require(
        GITHUB_REPO.fullmatch(path) is not None
        and all(part not in (".", "..") for part in path.split("/")),
        "untrusted historical Git repository identity",
    )
    return canonical_repository_family(path)


def inventory_local_original_sources(roots: list[str | Path]) -> dict:
    """Build source-repo identity inventory in RAM, never persist identifiers.

    A source checkout must be an immediate child of an approved historical
    phase's approved source storage folder. A .git directory is required;
    no file-backed worktrees, symlinks, junctions or arbitrary recursion.
    """
    _require(type(roots) is list and 0 < len(roots) <= MAX_PHASES,
             "bounded nonempty source storage roots required")
    phases: list[Path] = []
    seen_phase_paths: set[str] = set()
    for requested in roots:
        _require(isinstance(requested, (str, Path)), "invalid root type")
        root = Path(requested)
        _require(_safe_directory(root), "inaccessible or reparse owner root")
        if PHASE_NAME.fullmatch(root.name):
            candidates = [root]
        else:
            candidates = []
            for item in root.iterdir():
                if PHASE_NAME.fullmatch(item.name):
                    # Check the lexical entry even when a broken symlink or
                    # Windows reparse point falsely reports is_dir() == False.
                    _require(_safe_directory(item),
                             "unsafe historical phase entry")
                    candidates.append(item)
        for candidate in sorted(candidates, key=lambda p: p.name.casefold()):
            _require(_safe_directory(candidate), "unsafe historical phase folder")
            key = os.path.normcase(str(candidate.resolve()))
            _require(key not in seen_phase_paths,
                     "duplicate historical phase root")
            seen_phase_paths.add(key)
            phases.append(candidate)
    _require(0 < len(phases) <= MAX_PHASES, "no historical phase folders found")
    records: set[tuple[str, str]] = set()
    count = 0
    for phase in phases:
        for name in sorted(SOURCE_STORAGE_NAMES):
            store = phase / name
            if not os.path.lexists(store):
                continue
            _require(_safe_directory(store), "unsafe historical source store")
            for child in sorted(store.iterdir(), key=lambda p: p.name.casefold()):
                # An unresolved/broken .git checkout link must not disappear
                # from the census just because Path.is_dir() returns False.
                st = child.lstat()
                _require(
                    not stat.S_ISLNK(st.st_mode)
                    and not (getattr(st, "st_file_attributes", 0) & 0x400),
                    "unsafe original source entry",
                )
                if not child.is_dir():
                    continue
                _require(_safe_directory(child), "unsafe original checkout")
                dotgit = child / ".git"
                if not os.path.lexists(dotgit):
                    continue
                _require(_safe_directory(dotgit),
                         "original .git cannot be a symlink or worktree pointer")
                count += 1
                _require(count <= MAX_CHECKOUTS,
                         "original source checkout bound exceeded")
                origin = _github_root(_git(
                    child, "config", "--local", "--get", "remote.origin.url"
                ))
                head = _git(child, "rev-parse", "HEAD").lower()
                _require(GIT_SHA.fullmatch(head) is not None,
                         "invalid original Git SHA")
                top = _git(child, "rev-parse", "--show-toplevel")
                _require(
                    os.path.normcase(str(Path(top).resolve()))
                    == os.path.normcase(str(child.resolve())),
                    "historical repository checkout root mismatch",
                )
                records.add((origin, head))
    _require(count > 0, "no verified original Git checkout was found")
    _require(len(records) > 0, "original source identities are absent")
    # The raw set stays local to the in-memory receipt; caller must never
    # serialize this result or publish actual privately observed identities.
    digest = sha256(json.dumps(
        sorted(records), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")).hexdigest()
    return {
        "source_checkouts_inspected": count,
        "distinct_repository_revision_pairs": len(records),
        "distinct_repositories": len({repo for repo, _ in records}),
        "source_identity_fingerprint_sha256": digest,
        "_owner_only_repository_shas": frozenset(records),
        "historical_source_completeness_proven": False,
        "model_training_approved": False,
        "ood_evaluation_approved": False,
    }


def audit_source_inventory(
    candidate_metadata: dict,
    owner_hash_registry: dict,
    historical_source_registry: dict,
    paired_source_registry: dict,
    roots: list[str | Path],
) -> dict:
    """Fail closed when observed old checkout IDs are missing from history.

    This is an *additional* offline gate on top of H177, not a replacement.
    Because past checkouts may be deleted, a PASS still cannot certify
    historical completeness or authorize a model update.
    """
    public = audit_public_origin_pins(
        candidate_metadata, owner_hash_registry,
        historical_source_registry, paired_source_registry,
    )
    local = inventory_local_original_sources(roots)
    prior = {
        canonical_repository_family(name)
        for name in historical_source_registry["historical_repositories"]
    } | {canonical_repository_family(name) for name in KNOWN_NPM_UPSTREAM_REPOS}
    candidate_repos = {
        canonical_repository_family(row["repo"])
        for row in candidate_metadata["sources"]
    }
    observed = {
        repo for repo, _git_sha in local["_owner_only_repository_shas"]
    }
    _require(
        not (candidate_repos & observed),
        "new candidate uses original source checkout already seen privately",
    )
    _require(
        observed.issubset(prior),
        "owner historical registry omits a locally observed prior upstream",
    )
    return {
        "status": "H178_OWNER_GIT_ORIGINS_ONLY_NOT_DATASET_ADMISSION",
        "source_checkouts_inspected": local["source_checkouts_inspected"],
        "distinct_observed_repositories": local["distinct_repositories"],
        "source_identity_fingerprint_sha256":
            local["source_identity_fingerprint_sha256"],
        "public_prior_source_pins_checked":
            public["public_git_repository_pin_pairs_checked"],
        "source_history_completeness_proven": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }
