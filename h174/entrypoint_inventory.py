"""H174 original HTML entrypoint dependency inventory, never native QA.

Only scans NEW pinned candidate HTML and local paths. No browser, screenshots,
historical evaluations, gold, model inference, or network fetches.
"""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from h174.source_candidates import (
    CANDIDATES, _git, _reject_reparse, verify_local_source_only,
)


class _EntryReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.base_hrefs = []

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "base" and "href" in fields:
            self.base_hrefs.append(fields["href"])
        if tag == "script" and fields.get("src"):
            self.refs.append(("script", fields["src"]))
        if tag == "link" and "stylesheet" in (fields.get("rel") or "").lower().split():
            if fields.get("href"):
                self.refs.append(("stylesheet", fields["href"]))


def inventory_entrypoint(name, checkout):
    """Conservative source-only inventory, NOT executable-app admission.

    Missing assets, external resources and non-browser CSS source formats
    are reported separately. A zero count does not prove native readiness:
    module imports, build steps, browser actions and replay remain untested.
    """
    verify_local_source_only(name, checkout)
    root = Path(checkout)
    html = root / CANDIDATES[name]["html_path"]
    parser = _EntryReferences()
    parser.feed(html.read_text(encoding="utf-8"))
    # The first HTML <base href> changes browser URL resolution. Refuse
    # to misreport an apparently local decoy as the original script/CSS.
    if parser.base_hrefs:
        raise ValueError("H174 entrypoint base href changes browser URL resolution")
    missing, external, nonbrowser, present, untracked = [], [], [], [], []
    for kind, reference in parser.refs:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or reference.startswith("//"):
            external.append(reference)
            continue
        decoded = unquote(parsed.path).replace("\\", "/")
        if not decoded or any(part == ".." for part in decoded.split("/")):
            raise ValueError("H174 entrypoint unsafe local asset reference")
        if decoded.startswith("/"):
            target = root / decoded.lstrip("/")
        else:
            target = html.parent / decoded
        _reject_reparse(target)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("H174 entrypoint asset escapes pinned checkout")
        if not target.is_file():
            missing.append(reference)
            continue
        # A clean checkout may still contain ignored/generated files. A
        # present local asset is source-backed only when tracked by Git.
        relative = target.relative_to(root).as_posix()
        try:
            tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
        except ValueError:
            tracked = None
        if tracked != relative:
            untracked.append(reference)
        elif kind == "stylesheet" and target.suffix.lower() in (".scss", ".sass", ".less"):
            nonbrowser.append(reference)
        else:
            present.append(reference)
    return {
        "status": "H174_STATIC_HTML_REFERENCES_ONLY_NOT_NATIVE_QA",
        "app": name,
        "local_present": sorted(present),
        "local_missing": sorted(missing),
        "local_untracked": sorted(untracked),
        "external_references": sorted(external),
        "nonbrowser_stylesheets": sorted(nonbrowser),
        "source_native_qa_qualified": False,
        "split_assigned": False,
        "model_calls": 0,
    }
