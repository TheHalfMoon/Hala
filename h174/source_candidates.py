"""H174 prospective original-app donor provenance; SOURCE BYTES ONLY.

These are three independently pinned MIT *candidates*, not admitted TRAIN,
DEV, or OOD datasets. No split assignment, screenshot, owner gold, model,
GPU call, native action, or consumed benchmark is used by this module.
Copied the fail-closed immutable Git/HTML/license pattern from H173.
"""
import hashlib
import re
import subprocess
from pathlib import Path

from h174.split_firewall import BANNED, BANNED_GIT, BANNED_HTML

CANDIDATES = {
    "pickr": {
        "repo": "simonwep/pickr",
        "git_sha": "e8ff88d9161ffc6e3c46a4430be8c328c42d6415",
        "html_path": "index.html",
        "html_sha256": "4cc4f0dbcb26e79bbf09202ba3896d5d66953c6d40fb6347eac2c15522712d1e",
        "license_path": "LICENSE",
        "license_sha256": "7129bc74b781d3db2979d4617940b00e75885914b8d636a3155a64e57481e5e5",
        "license": "MIT",
    },
    "interact-dropzones": {
        "repo": "taye/interact.js",
        "git_sha": "a993eb8cd5cbc65f3a19e45ee24b7fedc026174f",
        "html_path": "examples/dropzones/index.html",
        "html_sha256": "9bc4dbe2a8f4e63b2f78a589b54adcd822ba552ecf481528a3b1d65f79430049",
        "license_path": "LICENSE",
        "license_sha256": "e805e2367aed149858263a4ae0a62168146af837a9ba3095aa01d02c8559e7fe",
        "license": "MIT",
    },
    "medium-zoom": {
        "repo": "francoischalifour/medium-zoom",
        "git_sha": "dccda8ab5a1f1677cf1f141d1ead4f14780f0ca4",
        "html_path": "examples/vanilla/index.html",
        "html_sha256": "0e85b970cc9a4f489945ff4f325c7d3380855aa7433223d24f2bd3336ab1a5ed",
        "license_path": "LICENSE",
        "license_sha256": "166ba718b70c6278cce9c9758396e25e35c7f1a940d39b2c8eeaceed18cc03fb",
        "license": "MIT",
    },
}

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def verify_catalog():
    """Reject historical aliases and duplicate candidate identities."""
    git_seen, html_seen = set(BANNED_GIT), set(BANNED_HTML)
    repo_seen = set()
    for name, pin in CANDIDATES.items():
        repo = pin["repo"]
        if (type(repo) is not str or _REPO.fullmatch(repo) is None
                or any(part in (".", "..") for part in repo.split("/"))):
            raise ValueError("H174 candidate invalid upstream repository identity")
        # Different commits, case aliases or demos in the same upstream
        # repository are NOT independent original-app donor families.
        canonical_repo = repo.casefold()
        if canonical_repo in repo_seen:
            raise ValueError("H174 candidate duplicate upstream repository family")
        repo_seen.add(canonical_repo)
        if (name in BANNED or not _HEX40.fullmatch(pin["git_sha"])
                or not _HEX64.fullmatch(pin["html_sha256"])
                or not _HEX64.fullmatch(pin["license_sha256"])
                or pin["license"] != "MIT"):
            raise ValueError("H174 candidate is historical or has invalid provenance")
        if pin["git_sha"] in git_seen or pin["html_sha256"] in html_seen:
            raise ValueError("H174 candidate aliases historical or another donor source")
        git_seen.add(pin["git_sha"])
        html_seen.add(pin["html_sha256"])
    if len(CANDIDATES) < 3:
        raise ValueError("H174 source candidates require independent families")
    return len(CANDIDATES)


def _git(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("H174 candidate original Git checkout unavailable") from exc


def _reject_reparse(path):
    """Never follow Git/HTML/license symlinks or Windows junctions."""
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("H174 candidate symlink or junction prohibited")


def verify_local_source_only(name, root):
    """Verify one pristine donor checkout; NEVER capture or score its GUI."""
    verify_catalog()
    if type(name) is not str or name not in CANDIDATES:
        raise ValueError("H174 unknown candidate donor")
    pin = CANDIDATES[name]
    root = Path(root)
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("H174 candidate Git checkout missing")
    _reject_reparse(root)
    _reject_reparse(root / ".git")
    if _git(root, "rev-parse", "HEAD") != pin["git_sha"]:
        raise ValueError("H174 candidate immutable Git revision mismatch")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("H174 candidate source dirty")
    remote = _git(root, "remote", "get-url", "origin")
    allowed = {
        "https://github.com/" + pin["repo"],
        "https://github.com/" + pin["repo"] + ".git",
        "git@github.com:" + pin["repo"],
        "git@github.com:" + pin["repo"] + ".git",
    }
    if remote not in allowed:
        raise ValueError("H174 candidate original Git remote mismatch")
    for path_key, hash_key in (
        ("html_path", "html_sha256"),
        ("license_path", "license_sha256"),
    ):
        relative = Path(pin[path_key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("H174 candidate source path escapes checkout")
        path = root
        for component in relative.parts:
            path = path / component
            _reject_reparse(path)
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("H174 candidate original HTML/license path missing or outside checkout")
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin[hash_key]:
            raise ValueError("H174 candidate original HTML/license bytes mismatch")
    return {
        "status": "H174_PINNED_SOURCE_BYTES_ONLY_NO_NATIVE_QA",
        "app": name,
        "source_git_sha": pin["git_sha"],
        "original_html_sha256": pin["html_sha256"],
        "license": pin["license"],
        "split_assigned": False,
        "source_native_qa_qualified": False,
        "train_dev_images_created": 0,
        "ood_images_created": 0,
        "model_calls": 0,
    }
