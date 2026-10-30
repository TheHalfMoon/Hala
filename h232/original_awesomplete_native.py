"""H232 original MIT Awesomplete source-native Windows Edge QA; research only.

Original Git/HTML/JS/license unchanged. Author QA invokes the original public
Awesomplete constructor on its original upstream test fixture. Never TRAIN/DEV/OOD.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from h231.native_replay import _allowed_url, _rgba_sha

REPO = "nudeui/awesomplete"  # canonical GitHub repository (id 30987756); LeaVerou URL redirects
COMMIT = "596bc5d5916266437970ecbf65434077d3950a3d"
PINNED = {
    "LICENSE": "3358ef750899fd62ec44637f2a23719a89bb7a8a0b4225d72488d2483359cf1d",
    "test/fixtures/options.html": "907c1a85936ac6f0a0eaaa7ad4867d44fbd7c3d2dcc179dfccd405f87b098c07",
    "awesomplete.js": "5f5b922b7945bf4a58487080ef075bf7e2734c953063886276cd386d760803f8",
}

def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H232 fail-closed: " + why)

def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       text=True, timeout=15, check=False)
    require(p.returncode == 0, "git command failed")
    return p.stdout.strip()

def preflight(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    require(Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
            "source must be original Git root")
    require(git(root, "rev-parse", "HEAD") == COMMIT, "wrong original Git SHA")
    require(git(root, "status", "--porcelain", "--untracked-files=all") == "",
            "dirty original source")
    require(git(root, "remote", "get-url", "origin").lower() in
            ("https://github.com/leaverou/awesomplete.git",
             "https://github.com/leaverou/awesomplete",
             "https://github.com/nudeui/awesomplete.git",
             "https://github.com/nudeui/awesomplete"), "wrong original upstream")
    require(not output.exists() and not output.is_relative_to(root),
            "evidence must be create-only and outside original source")
    require(not any((p / ".git").exists() for p in (output, *output.parents)),
            "evidence must be outside all Git checkouts")
    from h174.split_firewall import canonical_repository_family
    from h176.source_family_attribution import (
        FROZEN_H173_REPOS, FROZEN_OTHER_REPOS, KNOWN_NPM_UPSTREAM_REPOS)
    old = set(FROZEN_H173_REPOS.values()) | set(FROZEN_OTHER_REPOS.values()) | set(KNOWN_NPM_UPSTREAM_REPOS)
    require(all(canonical_repository_family(alias) not in old for alias in
                (REPO, "LeaVerou/awesomplete")), "known historical source reused")
    for name, digest in PINNED.items():
        path = root.joinpath(*name.split("/"))
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root)
                and sha256(path.read_bytes()).hexdigest() == digest,
                "original license/HTML/JS byte mismatch: " + name)
    require(b"MIT License" in (root / "LICENSE").read_bytes(),
            "original root MIT license missing")
    return {"root": root, "output": output,
            "html": root / "test" / "fixtures" / "options.html",
            "script": root / "awesomplete.js"}

def replay(root: Path, output: Path) -> dict:
    checked = preflight(root, output)
    from playwright.sync_api import sync_playwright
    report = {
        "status": "H232_ORIGINAL_SOURCE_NATIVE_RESEARCH_ONLY_NOT_ADMISSION",
        "source": REPO, "git_sha": COMMIT, "source_asset_sha256": PINNED,
        "contexts": [], "historical_source_and_rgba_complete": False,
        "license_and_media_audit_complete": False, "training_approved": False,
        "dev_approved": False, "ood_approved": False,
        "model_calls": 0, "optimizer_steps": 0}
    output.mkdir(parents=True, exist_ok=False)
    try:
        with sync_playwright() as p:
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            require(edge.is_file(), "owner Windows Edge missing")
            browser = p.chromium.launch(executable_path=str(edge), headless=True)
            try:
                for i in range(2):
                    ctx = browser.new_context(viewport={"width": 1100, "height": 760},
                                              device_scale_factor=1, service_workers="block")
                    attempted = []
                    def isolate(route):
                        if _allowed_url(route.request.url, checked["root"]):
                            route.continue_()
                        else:
                            attempted.append(urlsplit(route.request.url).scheme)
                            route.abort()
                    ctx.route("**/*", isolate)
                    page = ctx.new_page()
                    page.goto(checked["html"].as_uri(), wait_until="load")
                    page.add_script_tag(path=str(checked["script"]))
                    page.evaluate("""() => new Awesomplete(
                        document.querySelector('#with-data-list-inline'))""")
                    entry = page.locator("#with-data-list-inline")
                    before = page.screenshot(type="png")
                    entry.fill("Da")
                    require(entry.get_attribute("aria-expanded") == "true",
                            "original EN autocomplete did not open")
                    page.locator("ul[role=listbox] li").filter(has_text="Data").first.click()
                    require(entry.input_value() == "Data" and
                            entry.get_attribute("aria-expanded") == "false",
                            "original EN suggestion selection failed")
                    en_after = page.screenshot(type="png")
                    entry.fill("مرحبا")
                    require(entry.input_value() == "مرحبا" and
                            entry.get_attribute("aria-expanded") == "false",
                            "Arabic text input or unmatched-suggestion state failed")
                    ar_after = page.screenshot(type="png")
                    require(not attempted, "network/out-of-root request attempted")
                    require(_rgba_sha(before) != _rgba_sha(en_after),
                            "native visual state did not change")
                    for name, png in (("before", before), ("en-selected", en_after),
                                      ("ar-input", ar_after)):
                        (output / f"context-{i+1}-{name}.png").write_bytes(png)
                    report["contexts"].append({
                        "context": i+1, "original_english_suggestion_selected": True,
                        "arabic_text_entry_only": True, "network_attempts": len(attempted),
                        "before_png_sha256": sha256(before).hexdigest(),
                        "before_rgba_sha256": _rgba_sha(before),
                        "en_png_sha256": sha256(en_after).hexdigest(),
                        "en_rgba_sha256": _rgba_sha(en_after),
                        "ar_png_sha256": sha256(ar_after).hexdigest(),
                        "ar_rgba_sha256": _rgba_sha(ar_after)})
                    ctx.close()
            finally:
                browser.close()
        require(len(report["contexts"]) == 2, "two independent contexts required")
        (output / "receipt.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
        return report
    except BaseException:
        (output / "INCOMPLETE_NO_ADMISSION").write_text(
            "Incomplete original source QA; never TRAIN/DEV/OOD.\n", encoding="utf-8")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(args.source_root, args.output)
    print(json.dumps({"status": result["status"],
                      "contexts": len(result["contexts"]), "training_approved": False}))

if __name__ == "__main__":
    main()
