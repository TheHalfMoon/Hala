"""H329: source-only, local-Git evidence for original RailKit TRAIN candidates.

This is NOT an independent legal-rights review, a media-license grant, or
official H199 TRAIN admission. It reads ONLY Git objects for one pinned HTML
source, LICENSE text, and the source's small bytes. Never opens screenshots,
model outputs, datasets, private roots, sealed OOD or original gold.
"""
from __future__ import annotations

from hashlib import sha1
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS

ROLE = "H329_SOURCE_ORIGIN_EVIDENCE_ONLY_NOT_RIGHTS_ADMISSION"
ORIGIN = "3303396b77dc7671cd97f425c644f71b6b0b664e"
ORIGIN_PARENT = "9c636b7c73679d8422db13a15e3fe7a298b39606"
SOURCE = "h325/railkit.html"
SOURCE_BLOB = "743b5e1b54774a1deea4dbe6bfbe0613adead7e4"
LICENSE_BLOB = "d645695673349e3947e8e5ae42332d0ac3164cd7"
LIMIT = 16000
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
RESOURCE_CSS = re.compile(r"@import\b|url\s*\(|@font-face\b", re.I)
RESOURCE_JS = re.compile(
    r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|"
    r"Worker|SharedWorker|importScripts|sendBeacon)\s*\(|"
    r"\b(?:window\.open|navigator\.serviceWorker)\b|"
    r"\bimport\s*\(", re.I
)


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H329 SOURCE_CUSTODY_OR_RIGHTS_NOT_QUALIFIED")


class SourceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: dict[str, int] = {}
        self.references = 0
        self.externals = 0
        self._in_style = False
        self._in_script = False
        self.css = []
        self.scripts = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags[tag] = self.tags.get(tag, 0) + 1
        fields = dict(attrs)
        need(len(fields) == len(attrs))
        if tag in {"iframe", "object", "embed", "audio", "video", "img",
                   "source", "link", "base", "form", "svg", "canvas"}:
            self.externals += 1
        for key, value in fields.items():
            if key in {"src", "srcset", "href", "poster", "data", "action",
                       "formaction", "integrity"} or key.startswith("on"):
                self.references += 1
            if key == "style":
                self.css.append(value or "")
        if tag == "style":
            self._in_style = True
            need(not fields)
        elif tag == "script":
            self._in_script = True
            need(not fields)

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False
        elif tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.css.append(data)
        if self._in_script:
            self.scripts.append(data)


def audit_original_html(raw: object) -> dict:
    """Reject external references and dynamic loaders, not only CSS urls."""
    need(type(raw) is bytes and 1000 <= len(raw) <= LIMIT
         and b"\x00" not in raw)
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == SOURCE_BLOB)
    try:
        html = raw.decode("utf-8", errors="strict")
        parser = SourceHTMLParser()
        parser.feed(html)
        parser.close()
        need(not parser._in_script and not parser._in_style
             and parser.tags.get("script") == 1
             and parser.tags.get("style") == 1
             and parser.tags.get("main") == 1
             and parser.tags.get("button", 0) == 0
             and parser.references == parser.externals == 0
             and not RESOURCE_CSS.search("\n".join(parser.css))
             and not RESOURCE_JS.search("\n".join(parser.scripts))
             and "configureRailKit" in html
             and "railKitState" in html
             and "ar" in html
             and "LAB-" not in html)
    except (UnicodeError, ValueError):
        raise ValueError("H329 SOURCE_CUSTODY_OR_RIGHTS_NOT_QUALIFIED") from None
    return {
        "source_pinned_git_blob": True,
        "external_html_assets_or_links": 0,
        "external_css_imports_or_urls": 0,
        "dynamic_script_network_loader_patterns": 0,
        "inline_script_count": 1,
        "inline_style_count": 1,
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True, timeout=25,
    )
    return result.stdout.strip()


def audit_git_history(root: Path, *, expected_head: str) -> dict:
    """Require pinned introducing commit, parent absence and Apache license.

    Requires a local full-enough Git object database with ORIGIN and parent
    available, and a clean checkout. Never fetches remote content itself.
    """
    need(isinstance(root, Path)
         and type(expected_head) is str
         and SHA40.fullmatch(expected_head) is not None)
    try:
        root = root.resolve(strict=True)
        need(root.is_dir()
             and _git(root, "rev-parse", "--show-toplevel") == str(root)
             and _git(root, "rev-parse", "HEAD") == expected_head
             and _git(root, "status", "--porcelain", "--untracked-files=normal")
             == ""
             and _git(root, "cat-file", "-t", ORIGIN) == "commit"
             and _git(root, "cat-file", "-t", ORIGIN_PARENT) == "commit"
             and _git(root, "rev-parse", ORIGIN + "^") == ORIGIN_PARENT)
        # A commit on a divergent branch cannot impersonate an ancestor.
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             ORIGIN, expected_head], capture_output=True,
            text=True, check=True, timeout=25)
        need(_git(root, "rev-parse", ORIGIN + ":" + SOURCE) == SOURCE_BLOB
             and _git(root, "rev-parse", "HEAD:" + SOURCE) == SOURCE_BLOB
             and _git(root, "rev-parse", ORIGIN + ":LICENSE") == LICENSE_BLOB
             and _git(root, "rev-parse", "HEAD:LICENSE") == LICENSE_BLOB)
        # Parent must not have RailKit: avoids repackaged older project.
        missing = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e",
             ORIGIN_PARENT + ":" + SOURCE],
            capture_output=True, text=True, timeout=25)
        need(missing.returncode != 0)
        path = root / SOURCE
        license_path = root / "LICENSE"
        need(path.is_file() and not path.is_symlink()
             and not path.parent.is_symlink()
             and license_path.is_file() and not license_path.is_symlink()
             and "Apache License" in license_path.read_text(encoding="utf-8")
             and "Version 2.0, January 2004" in
                 license_path.read_text(encoding="utf-8"))
        scan = audit_original_html(path.read_bytes())
        need(not ADMITTED_TRAIN_MANIFEST_SHAS)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        raise ValueError("H329 SOURCE_CUSTODY_OR_RIGHTS_NOT_QUALIFIED") from None
    return {
        "status": ROLE,
        **scan,
        "origin_commit": ORIGIN,
        "source_new_in_origin_parent": True,
        "origin_is_ancestor_of_checked_head": True,
        "apache_2_license_blob_unchanged_from_origin": True,
        "verified_github_commit_cryptographic_signature": False,
        "independent_copyright_title_reviewed": False,
        "generated_screenshot_license_scope_independently_reviewed": False,
        "complete_external_source_history_proven": False,
        "legal_train_use_authorized": False,
        "official_h199_train_admitted": False,
        "independent_dev_or_blind_ood_qualified": False,
        "model_generations": 0,
        "optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "can_release": False,
    }
