"""H231: isolated private native GUI replay for *new* original donor candidates.

This is an oracle/reachability data-acquisition preflight, never Hala model
accuracy or qualified TRAIN/DEV/OOD admission. Owner historical media is forbidden.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from urllib.request import url2pathname

SCHEMA = "hala.h231.private-native-replay.v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"[a-z][a-z0-9-]{1,45}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
CASE_FIELDS = {"id", "lang", "action", "selector", "destination",
               "text", "expect_selector", "expect_attribute", "expect_value"}
META_FIELDS = {"schema", "repo", "git_sha", "license", "license_sha256",
               "html", "html_sha256", "cases"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("H231 fail-closed: " + message)


def _read_json(path: Path) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("H231 fail-closed: duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def _file(root: Path, raw: str, suffix: str = "") -> Path:
    _require(type(raw) is str and bool(raw) and
             "\\" not in raw and not raw.startswith("/") and
             all(part not in ("", ".", "..") for part in raw.split("/")),
             "unsafe relative path")
    path = root.joinpath(*raw.split("/"))
    _require(path.is_file() and not path.is_symlink() and
             path.resolve().is_relative_to(root) and
             (not suffix or path.suffix.lower() == suffix),
             "missing/escaping/symlinked source file")
    return path


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args],
                            capture_output=True, text=True, timeout=15,
                            check=False)
    _require(result.returncode == 0, "unverified local Git source")
    return result.stdout.strip()


def _origin_family(root: Path) -> str:
    """Require the claimed donor to match its local Git origin identity.

    This is a fail-closed source-consistency check, NOT independent proof that
    GitHub created or authenticated the checkout, or that old aliases are absent.
    """
    raw = _run_git(root, "remote", "get-url", "origin")
    if raw.startswith("git@github.com:"):
        slug = raw[len("git@github.com:"):]
    else:
        parts = urlsplit(raw)
        _require(parts.scheme in ("https", "ssh") and
                 parts.netloc in ("github.com", "git@github.com") and
                 not parts.query and not parts.fragment and
                 parts.path.startswith("/"),
                 "Git origin must be an exact GitHub repository URL")
        slug = parts.path[1:]
    from h174.split_firewall import canonical_repository_family
    return canonical_repository_family(slug)


def preflight(manifest: dict, root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    _require(type(manifest) is dict and set(manifest) == META_FIELDS and
             manifest["schema"] == SCHEMA, "exact manifest schema required")
    repo, commit = manifest["repo"], manifest["git_sha"]
    _require(type(repo) is str and REPOSITORY.fullmatch(repo) is not None and
             type(commit) is str and HEX40.fullmatch(commit) is not None,
             "original repository and SHA required")
    _require(_run_git(root, "rev-parse", "HEAD") == commit,
             "local checkout differs from pinned source commit")
    _require(_run_git(root, "status", "--porcelain", "--untracked-files=all") == "",
             "dirty or untracked donor source")
    _require(not output.is_relative_to(root), "private evidence must be outside source root")
    _require(not any((parent / ".git").exists() for parent in (output, *output.parents)),
             "private evidence must be outside every Git checkout")
    _require(not output.exists(), "output directory is create-only")
    # Explicit minimum source-history exclusions: NOT a complete history proof.
    from h176.source_family_attribution import (
        FROZEN_H173_REPOS, FROZEN_OTHER_REPOS, KNOWN_NPM_UPSTREAM_REPOS)
    from h174.split_firewall import canonical_repository_family
    family = canonical_repository_family(repo)
    old = (set(FROZEN_H173_REPOS.values()) |
           set(FROZEN_OTHER_REPOS.values()) | set(KNOWN_NPM_UPSTREAM_REPOS))
    _require(family not in old, "known historical original source reused")
    _require(_origin_family(root) == family, "source repository must match origin")
    license_path = _file(root, manifest["license"])
    html_path = _file(root, manifest["html"], ".html")
    for field, path in (("license_sha256", license_path), ("html_sha256", html_path)):
        expected = manifest[field]
        _require(type(expected) is str and HEX64.fullmatch(expected) is not None and
                 sha256(path.read_bytes()).hexdigest() == expected,
                 field + " does not match original local bytes")
    cases = manifest["cases"]
    _require(type(cases) is list and 1 <= len(cases) <= 16, "1..16 cases required")
    ids = set()
    for case in cases:
        _require(type(case) is dict and set(case) == CASE_FIELDS,
                 "exact case schema required")
        case_id = case["id"]
        _require(type(case_id) is str and CASE_ID.fullmatch(case_id) is not None
                 and case_id not in ids, "invalid/duplicate case ID")
        ids.add(case_id)
        _require(case["lang"] in ("en", "ar") and
                 case["action"] in ("click", "type", "drag"),
                 "unknown language or action")
        for key in ("selector", "expect_selector", "expect_attribute", "expect_value"):
            _require(type(case[key]) is str and 1 <= len(case[key]) <= 200,
                     "selector/expectation must be bounded text")
        _require(case["expect_attribute"].startswith("data-"),
                 "only DOM test data attributes are permitted as outcome proof")
        _require(type(case["destination"]) is str and
                 type(case["text"]) is str, "destination/text must be strings")
        if case["action"] == "drag":
            _require(bool(case["destination"]) and not case["text"],
                     "drag needs destination, no text")
        elif case["action"] == "type":
            _require(not case["destination"] and 1 <= len(case["text"]) <= 200,
                     "type requires bounded text")
        else:
            _require(not case["destination"] and not case["text"],
                     "click must have no destination or text")
    return {"root": root, "output": output, "html": html_path, "repo": family,
            "commit": commit, "cases": cases}


def _rgba_sha(png: bytes) -> str:
    from PIL import Image
    with Image.open(BytesIO(png)) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        return sha256(width.to_bytes(4, "big") +
                      height.to_bytes(4, "big") + rgba.tobytes()).hexdigest()


def _allowed_url(url: str, root: Path) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return False
    try:
        target = Path(url2pathname(parsed.path)).resolve(strict=True)
        return target.is_file() and not target.is_symlink() and target.is_relative_to(root)
    except (ValueError, OSError):
        return False


def replay(manifest_path: Path, source_root: Path, output_dir: Path) -> dict:
    manifest = _read_json(manifest_path)
    checked = preflight(manifest, source_root, output_dir)
    from playwright.sync_api import sync_playwright
    report = {
        "schema": SCHEMA,
        "status": "H231_OWNER_ONLY_ORACLE_NATIVE_REPLAY_NOT_ADMISSION",
        "source_repository": checked["repo"],
        "source_git_sha": checked["commit"],
        "root_license_claim_only": True,
        "complete_license_audit": False,
        "complete_historical_source_and_rgba_proven": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "cases": [],
    }
    # Files are never included in the repo or sent over network.
    output = checked["output"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        with sync_playwright() as playwright:
            from h130.browser_env import _chromium_binary
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            opts = {"headless": True}
            existing = _chromium_binary()
            if existing:
                opts["executable_path"] = existing
            elif edge.is_file():
                opts["executable_path"] = str(edge)
            browser = playwright.chromium.launch(**opts)
            try:
                ctx = browser.new_context(
                    viewport={"width": 1100, "height": 760},
                    device_scale_factor=1, service_workers="block")
                attempted_network = []
                def isolate(route):
                    url = route.request.url
                    if _allowed_url(url, checked["root"]):
                        route.continue_()
                    else:
                        attempted_network.append(urlsplit(url).scheme)
                        route.abort()
                ctx.route("**/*", isolate)
                page = ctx.new_page()
                for case in checked["cases"]:
                    page.goto(checked["html"].as_uri(), wait_until="load")
                    before = page.screenshot(type="png")
                    target = page.locator(case["selector"])
                    if case["action"] == "click":
                        target.click(timeout=3000)
                    elif case["action"] == "type":
                        target.fill(case["text"], timeout=3000)
                    else:
                        target.drag_to(page.locator(case["destination"]), timeout=3000)
                    state = page.locator(case["expect_selector"]).get_attribute(
                        case["expect_attribute"], timeout=3000)
                    after = page.screenshot(type="png")
                    good = state == case["expect_value"]
                    for suffix, png in (("before", before), ("after", after)):
                        (output / (case["id"] + "-" + suffix + ".png")).write_bytes(png)
                    report["cases"].append({
                        "id": case["id"], "language": case["lang"],
                        "action": case["action"],
                        "native_outcome_matches_oracle": good,
                        "before_png_sha256": sha256(before).hexdigest(),
                        "before_rgba_sha256": _rgba_sha(before),
                        "after_png_sha256": sha256(after).hexdigest(),
                        "after_rgba_sha256": _rgba_sha(after),
                    })
                report["network_or_unscoped_file_attempts"] = len(attempted_network)
                _require(not attempted_network, "network or out-of-root file access attempted")
                report["all_native_oracle_outcomes_passed"] = all(
                    c["native_outcome_matches_oracle"] for c in report["cases"])
                _require(report["all_native_oracle_outcomes_passed"],
                         "native oracle outcome mismatch")
                ctx.close()
            finally:
                browser.close()
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    except BaseException:
        # Fail closed: do not leave incomplete evidence falsely looking passed.
        (output / "INCOMPLETE_NO_ADMISSION").write_text(
            "Capture interrupted or failed; never use this output for training.\n",
            encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.manifest, args.source_root, args.output)
    print(json.dumps({"status": result["status"],
                      "cases": len(result["cases"]),
                      "oracle_pass": result["all_native_oracle_outcomes_passed"],
                      "training_approved": False}, sort_keys=True))


if __name__ == "__main__":
    main()
