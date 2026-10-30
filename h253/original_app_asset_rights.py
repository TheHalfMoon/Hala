"""Source-only visual asset dependency audit. NEVER grants dataset admission.

Reads public original application HTML/CSS and Git metadata only. Does not open
image/font bytes, screenshots, private cohorts, model weights, or any OOD data.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)
CSS_IMPORT = re.compile(r"@import\s+(?!url\()['\"]([^'\"]+)['\"]", re.I)
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".avif", ".woff", ".woff2", ".ttf", ".otf"}
LINK_RELS = {"stylesheet", "icon", "shortcut", "preconnect", "dns-prefetch", "preload", "modulepreload"}


class ResourceHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.resources: list[tuple[str, str]] = []
        self.inline_css: list[str] = []
        self._in_style = False
        self.inline_svg_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        if tag == "svg":
            self.inline_svg_count += 1
        if tag == "style":
            self._in_style = True
        if d.get("style"):
            self.inline_css.append(d["style"] or "")
        if tag == "link" and d.get("href"):
            rel = (d.get("rel") or "").lower().split()
            if set(rel) & LINK_RELS:
                kind = "hint" if set(rel) & {"preconnect", "dns-prefetch"} else "link"
                self.resources.append((kind, d["href"] or ""))
        for key in ("src", "poster", "data"):
            if tag in {"script", "img", "source", "video", "audio", "iframe", "embed", "object"} and d.get(key):
                self.resources.append((tag, d[key] or ""))
        if d.get("srcset"):
            for item in (d["srcset"] or "").split(","):
                candidate = item.strip().split()
                if candidate:
                    self.resources.append(("srcset", candidate[0]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.inline_css.append(data)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return result.stdout.strip()


def _css_resources(text: str) -> list[tuple[str, str]]:
    found = [("css-url", m.group(2)) for m in CSS_URL.finditer(text)]
    found += [("css-import", m.group(1)) for m in CSS_IMPORT.finditer(text)]
    return found


def audit(root: Path, expected_sha: str, entries: list[str]) -> dict:
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("Expected immutable full Git SHA required")
    if _git(root, "rev-parse", "HEAD") != expected_sha:
        raise ValueError("Original Git HEAD differs from pinned source")
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Original source checkout is not clean")
    if not (root / "LICENSE").is_file() or "MIT License" not in (root / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("Original root MIT LICENSE evidence missing")

    local_media: set[str] = set()
    remote: set[str] = set()
    hints: set[str] = set()
    missing: set[str] = set()
    unsafe: set[str] = set()
    css_visited: set[Path] = set()
    svg_count = 0
    html_count = 0

    def record(origin: Path, kind: str, value: str) -> None:
        value = value.strip()
        if not value or value.startswith("#"):
            return
        if value.startswith(("data:", "blob:", "javascript:")):
            unsafe.add(kind + ":embedded-or-opaque")
            return
        if value.startswith("//") or urlsplit(value).scheme in {"https", "http"}:
            (hints if kind == "hint" else remote).add(value)
            return
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            unsafe.add(kind + ":unsupported-scheme")
            return
        relative = unquote(parsed.path)
        if not relative:
            return
        candidate = (origin.parent / relative).resolve()
        try:
            rel = candidate.relative_to(root).as_posix()
        except ValueError:
            unsafe.add(kind + ":path-escape")
            return
        if not candidate.is_file():
            missing.add(rel)
            return
        if candidate.suffix.lower() in MEDIA_SUFFIXES or kind in {"img", "srcset", "video", "audio"}:
            local_media.add(rel)
        if candidate.suffix.lower() == ".css" and candidate not in css_visited:
            css_visited.add(candidate)
            for css_kind, css_value in _css_resources(candidate.read_text(encoding="utf-8")):
                record(candidate, css_kind, css_value)

    for entry in entries:
        html_path = (root / entry).resolve()
        if not html_path.is_file() or not html_path.is_relative_to(root) or html_path.suffix.lower() != ".html":
            raise ValueError("HTML entry missing, invalid or outside original checkout")
        html_count += 1
        parser = ResourceHTML()
        parser.feed(html_path.read_text(encoding="utf-8"))
        svg_count += parser.inline_svg_count
        for kind, value in parser.resources:
            record(html_path, kind, value)
        for inline_css in parser.inline_css:
            for kind, value in _css_resources(inline_css):
                record(html_path, kind, value)

    return {
        "status": "INCOMPLETE_NO_ADMISSION",
        "pinned_source_verified": True,
        "entry_count": html_count,
        "referenced_local_media": sorted(local_media),
        "remote_runtime_dependencies": sorted(remote),
        "remote_hints_not_proof_of_load": sorted(hints),
        "missing_local_resources": sorted(missing),
        "unsafe_or_embedded_resources": sorted(unsafe),
        "inline_svg_elements": svg_count,
        "visual_media_rights_proven": False,
        "actual_browser_resources_loaded_proven": False,
        "historical_source_and_rgba_completeness_proven": False,
        "official_train_dev_ood_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--entry", action="append", required=True)
    args = parser.parse_args()
    try:
        report = audit(args.repo, args.sha, args.entry)
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        print(json.dumps({"status": "INCOMPLETE_NO_ADMISSION", "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
