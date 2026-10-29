"""Read-only H174 tracked browser-module edge inventory; NOT native QA.

This deliberately conservative scanner is not a JavaScript parser. An unknown
import/export syntax is reported as an unresolved edge, never ignored to
claim browser readiness. No source writes, browser, network or model calls.
"""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from h174.entrypoint_inventory import _EntryReferences, inventory_entrypoint
from h174.source_candidates import CANDIDATES, _git, _reject_reparse

_JS_STATIC = re.compile(
    r"""(?:^|[;\n])\s*(?:import\s+(?:(?:[^;\n]*?)\s+from\s+)?|export\s+[^;\n]*?\s+from\s+)['"]([^'"\n]+)['"]""",
    re.M,
)
_JS_DYNAMIC = re.compile(r"""\bimport\s*\(\s*(['"])([^'"\n]+)\1\s*\)""")
_CSS_IMPORT = re.compile(
    r"""@import\s+(?:url\(\s*)?['"]?([^'")\s;]+)""", re.I
)
_JS_TOKEN = re.compile(r"\b(?:import\s*(?:\(|[\w{*'\"])|export\s*\{)")
_MAX_FILES = 64
_MAX_BYTES = 512 * 1024


def _dedup(items):
    return sorted(set(items))


def _resolve(root, base, ref):
    # URI parsing runs before and after percent decoding, so an encoded
    # external scheme or traversal can never be treated as a local file.
    for _ in range(3):
        decoded = unquote(ref)
        if decoded == ref:
            break
        ref = decoded
    parsed = urlsplit(ref)
    if parsed.scheme or parsed.netloc or ref.startswith("//"):
        return "external", ref
    name = parsed.path.replace("\\", "/")
    if not name or name.startswith("//") or (
        "%2f" in name.casefold() or "%5c" in name.casefold()
    ):
        raise ValueError("H174 module edge has unsafe encoded/traversal path")
    if name.startswith("../"):
        raise ValueError("H174 module edge parent traversal forbidden")
    if not (name.startswith("/") or name.startswith("./")):
        return "bare_package", ref
    relative = name[1:] if name.startswith("/") else name[2:]
    if not relative or any(part in ("..", ".", "") for part in relative.split("/")):
        raise ValueError("H174 module edge has unsafe encoded/traversal path")
    target = root / relative if name.startswith("/") else base / relative
    _reject_reparse(target)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("H174 module edge escapes original checkout")
    if not target.is_file():
        return "missing", ref
    relative = target.relative_to(root).as_posix()
    try:
        tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
    except ValueError:
        tracked = None
    if tracked != relative:
        return "untracked", ref
    return "tracked", target


def inventory_module_edges(name, checkout):
    """Bounded source-only import graph, not a proof of JS completeness."""
    entry = inventory_entrypoint(name, checkout)
    root = Path(checkout)
    html = root / CANDIDATES[name]["html_path"]
    parser = _EntryReferences()
    parser.feed(html.read_text(encoding="utf-8"))
    edges = {k: [] for k in (
        "tracked_edges", "missing_local", "untracked_local",
        "bare_packages", "external_edges", "unparsed_edges",
    )}
    queue, seen = [], set()
    for kind, reference in parser.refs:
        if kind not in ("script", "stylesheet"):
            continue
        # HTML references are already provenance-audited by the parent.
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or reference.startswith("//"):
            edges["external_edges"].append(reference)
            continue
        local = unquote(parsed.path).replace("\\", "/")
        target = root / local.lstrip("/") if local.startswith("/") else html.parent / local
        if target.is_file() and target.relative_to(root).as_posix() in {
            _git(root, "ls-files", "--", target.relative_to(root).as_posix())
        }:
            queue.append(target)
    while queue:
        target = queue.pop()
        relative = target.relative_to(root).as_posix()
        if relative in seen:
            continue
        if len(seen) >= _MAX_FILES:
            raise ValueError("H174 module graph exceeds bounded source inspection")
        seen.add(relative)
        if target.stat().st_size > _MAX_BYTES:
            raise ValueError("H174 module source exceeds bounded byte limit")
        code = target.read_text(encoding="utf-8")
        refs = []
        if target.suffix.lower() in (".js", ".mjs", ".ts", ".jsx", ".tsx"):
            refs.extend(_JS_STATIC.findall(code))
            refs.extend(m.group(2) for m in _JS_DYNAMIC.finditer(code))
            # Conservative: flag imports which are not recognized by this
            # limited scanner; never certify the graph as complete JS parsing.
            matches = len(refs)
            if len(_JS_TOKEN.findall(code)) > matches:
                edges["unparsed_edges"].append(relative)
        elif target.suffix.lower() == ".css":
            refs.extend(_CSS_IMPORT.findall(code))
        for ref in refs:
            # CSS @import "theme.css" is path-relative even without "./";
            # JavaScript uses a different bare-package resolution contract.
            lookup = ref
            if (target.suffix.lower() == ".css" and
                not ref.startswith(("/", ".")) and
                not urlsplit(ref).scheme and not ref.startswith("//")):
                lookup = "./" + ref
            kind, resolved = _resolve(root, target.parent, lookup)
            label = relative + " -> " + ref
            mapping = {
                "external": "external_edges", "bare_package": "bare_packages",
                "missing": "missing_local", "untracked": "untracked_local",
                "tracked": "tracked_edges",
            }
            edges[mapping[kind]].append(label)
            if kind == "tracked":
                queue.append(resolved)
    output = {
        "status": "H174_BOUNDED_STATIC_MODULE_EDGES_ONLY_NOT_NATIVE_QA",
        "app": name, "inspected_tracked_files": len(seen),
        "direct_html_missing": entry["local_missing"],
        "direct_html_untracked": entry["local_untracked"],
        "nonbrowser_stylesheets": entry["nonbrowser_stylesheets"],
        **{key: _dedup(items) for key, items in edges.items()},
        "requires_build_or_resolution": any((
            entry["local_missing"], entry["local_untracked"],
            entry["external_references"], entry["nonbrowser_stylesheets"],
            edges["missing_local"], edges["untracked_local"],
            edges["bare_packages"], edges["external_edges"],
            edges["unparsed_edges"],
        )),
        "native_browser_qualified": False,
        "source_families_assigned_to_splits": False,
        "model_calls": 0,
    }
    return output
