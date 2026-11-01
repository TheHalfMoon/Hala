"""H339: cryptographically pin H338 first-party SignalDesk *source* custody.

Not a legal copyright decision, license grant for generated screenshots,
dataset admission, model improvement, or source-family-blind OOD evidence.
Only local Git object metadata, tiny inline HTML and repository LICENSE.
No private media, source gold, model outputs, data samples or weights.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import re
import subprocess

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h329.railkit_source_provenance import (
    SourceHTMLParser, RESOURCE_CSS, RESOURCE_JS,
    audit_git_history as audit_railkit_git_history,
)
from h338.original_large_train_candidates import (
    ROLE as SIGNAL_CANDIDATE_ROLE, SCENES, SOURCE_BLOB as SIGNAL_BLOB,
)

ROLE = "H339_BILINGUAL_ORIGINAL_SOURCE_EVIDENCE_ONLY_NOT_TRAIN_RIGHTS"
ORIGIN = "ae183d4531ccf1ada8c98098416e38e715332cd6"
ORIGIN_PARENT = "5aca7ceb8e4e5d86d736dd83683a60ef84d2d3f2"
SOURCE = "h338/signaldesk.html"
SOURCE_BLOB = "3d787aa3aafb999d342cbd3685c20d0f34d5fcd8"
LICENSE_BLOB = "d645695673349e3947e8e5ae42332d0ac3164cd7"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H339_SOURCE_CUSTODY_NOT_LEGAL_TRAIN_ADMISSION")


def audit_signaldesk_html(raw: object) -> dict:
    """Inspect pinned inline-only original synthetic source, not media."""
    need(type(raw) is bytes and 1000 <= len(raw) <= 12000
         and b"\x00" not in raw)
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == SOURCE_BLOB == SIGNAL_BLOB)
    try:
        html = raw.decode("utf-8", "strict")
        parser = SourceHTMLParser()
        parser.feed(html)
        parser.close()
        need(not parser._in_style and not parser._in_script
             and parser.tags.get("script") == 1
             and parser.tags.get("style") == 1
             and parser.tags.get("main") == 1
             and parser.tags.get("header") == 1
             and parser.tags.get("button", 0) == 0
             and parser.references == 0 and parser.externals == 0
             and not RESOURCE_CSS.search("\n".join(parser.css))
             and not RESOURCE_JS.search("\n".join(parser.scripts))
             and "configureSignalDesk" in html
             and "signalDeskState" in html
             and "SIM " in html
             and "signal" in html
             and "<title>SignalDesk" in html)
    except (UnicodeError, ValueError):
        raise ValueError(
            "H339_SOURCE_CUSTODY_NOT_LEGAL_TRAIN_ADMISSION") from None
    return {
        "pinned_original_html_git_blob": SOURCE_BLOB,
        "inline_script_count": 1, "inline_style_count": 1,
        "external_html_assets_or_links": 0,
        "external_css_resources": 0,
        "dynamic_network_loaders": 0,
        "source_author_or_title_independently_proven": False,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, check=True, text=True, timeout=25,
    ).stdout.strip()


def audit_git_history(root: Path, *, expected_head: str) -> dict:
    """Require actual origin+parent ancestry, unchanged original source/license.

    Run on an already-clean full-history checkout. No fetch performed.
    """
    need(isinstance(root, Path) and type(expected_head) is str
         and SHA40.fullmatch(expected_head) is not None)
    try:
        root = root.resolve(strict=True)
        need(root.is_dir()
             and Path(_git(root, "rev-parse", "--show-toplevel")).resolve() == root
             and _git(root, "rev-parse", "HEAD") == expected_head
             and _git(root, "status", "--porcelain",
                      "--untracked-files=normal") == ""
             and _git(root, "cat-file", "-t", ORIGIN) == "commit"
             and _git(root, "cat-file", "-t", ORIGIN_PARENT) == "commit"
             and _git(root, "rev-parse", ORIGIN + "^") == ORIGIN_PARENT)
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             ORIGIN, expected_head], capture_output=True,
            text=True, check=True, timeout=25)
        need(_git(root, "rev-parse", ORIGIN + ":" + SOURCE) == SOURCE_BLOB
             and _git(root, "rev-parse", "HEAD:" + SOURCE) == SOURCE_BLOB
             and _git(root, "rev-parse", ORIGIN + ":LICENSE") == LICENSE_BLOB
             and _git(root, "rev-parse", "HEAD:LICENSE") == LICENSE_BLOB)
        # It was introduced on this branch: reject rebadged prior code.
        missing = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e",
             ORIGIN_PARENT + ":" + SOURCE],
            capture_output=True, text=True, timeout=25)
        need(missing.returncode != 0)
        p = root / SOURCE
        license_path = root / "LICENSE"
        need(p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and license_path.is_file() and not license_path.is_symlink()
             and "Apache License" in license_path.read_text(encoding="utf-8")
             and "Version 2.0, January 2004"
                in license_path.read_text(encoding="utf-8")
             and not ADMITTED_TRAIN_MANIFEST_SHAS)
        scan = audit_signaldesk_html(p.read_bytes())
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        raise ValueError(
            "H339_SOURCE_CUSTODY_NOT_LEGAL_TRAIN_ADMISSION") from None
    return {
        "status": ROLE, **scan,
        "introduction_commit": ORIGIN,
        "previous_parent_lacks_source": True,
        "introduction_is_ancestor_of_current_commit": True,
        "unchanged_apache_2_license_git_blob": True,
        "introduction_commit_cryptographically_signed": False,
        "full_external_copyright_title_proven": False,
        "generated_screenshot_training_rights_independently_granted": False,
        "independent_source_family_rights_review_completed": False,
        "official_h199_train_admitted": False,
        "dev_or_blind_ood_qualified": False,
        "model_generations": 0, "optimizer_steps": 0,
        "adapted_model_gain_proven": False, "can_release": False,
    }


def combined_candidacy_report(
    railkit: object, signaldesk: object,
) -> dict:
    """Conservative cross-family capacity *only*, no legal approval."""
    from h329.railkit_source_provenance import ROLE as RAIL_ROLE
    need(type(railkit) is dict and type(signaldesk) is dict)
    need(railkit.get("status") == RAIL_ROLE
         and signaldesk.get("status") == ROLE
         and railkit.get("origin_commit")
             == "3303396b77dc7671cd97f425c644f71b6b0b664e"
         and railkit.get("source_pinned_git_blob") is True
         and railkit.get("source_new_in_origin_parent") is True
         and railkit.get("apache_2_license_blob_unchanged_from_origin") is True
         and signaldesk.get("introduction_commit") == ORIGIN
         and signaldesk.get("previous_parent_lacks_source") is True
         and signaldesk.get("unchanged_apache_2_license_git_blob") is True
         and signaldesk.get("pinned_original_html_git_blob") == SOURCE_BLOB
         and signaldesk.get("external_html_assets_or_links") == 0
         and signaldesk.get("dynamic_network_loaders") == 0)
    need(all(railkit.get(k) is False for k in (
        "independent_copyright_title_reviewed",
        "generated_screenshot_license_scope_independently_reviewed",
        "legal_train_use_authorized", "official_h199_train_admitted",
        "independent_dev_or_blind_ood_qualified", "can_release"))
         and all(signaldesk.get(k) is False for k in (
             "full_external_copyright_title_proven",
             "generated_screenshot_training_rights_independently_granted",
             "independent_source_family_rights_review_completed",
             "official_h199_train_admitted", "dev_or_blind_ood_qualified",
             "can_release"))
         and len(SCENES) == 128
         and sum(x[1] == "en" for x in SCENES) == 64
         and sum(x[1] == "ar" for x in SCENES) == 64
         and not ADMITTED_TRAIN_MANIFEST_SHAS)
    return {
        "status": "H339_TWO_ORIGINAL_COHORTS_PROVENANCE_ONLY",
        "candidate_source_families": 2,
        "total_candidate_scenes": 144,
        "total_candidate_action_messages": 288,
        "english_candidate_scenes": 72,
        "arabic_rtl_candidate_scenes": 72,
        "verified_git_source_introductions": 2,
        "independent_rights_reviewer_approvals": 0,
        "official_h199_train_admitted_manifests": 0,
        "can_use_as_training_data": False,
        "independent_dev_or_blind_ood_qualified": False,
        "new_optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "can_release": False,
    }


def audit_both_sources(root: Path, *, expected_head: str) -> dict:
    need(isinstance(root, Path)
         and type(expected_head) is str
         and SHA40.fullmatch(expected_head) is not None)
    # Both individually require the same exact clean checked-out HEAD.
    rail = audit_railkit_git_history(root, expected_head=expected_head)
    signal = audit_git_history(root, expected_head=expected_head)
    return combined_candidacy_report(rail, signal)
