"""H236: genuine original MIT Dragula JS, native offline mouse drag QA.

Source-only research; strictly NOT original app TRAIN, DEV, OOD or model gain.
All owner-local captures stay outside every Git tree, never uploaded.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from h231.native_replay import _rgba_sha
from h174.split_firewall import (
    canonical_repository_family, FROZEN_PRIOR_ORIGINAL_REPOS,
)
from h176.source_family_attribution import (
    FROZEN_H173_REPOS, FROZEN_OTHER_REPOS, KNOWN_NPM_UPSTREAM_REPOS,
)

REPO = "bevacqua/dragula"
COMMIT = "09ab978dd550490f1f8e456551a1a39ed0d7f984"
PINNED = {
    "license": "13c2292db05054bffd25180c9e696cf798a033b0fcdc0806b7222aedfd3c2cf8",
    "dist/dragula.min.js": "7a08f9c71b63e340ecd0651b39b559ee6d8b10eb27beef62e4e82d99dd48fa39",
    "index.html": "ac94a69fbf7b7833bc0b4b13daef74eeb6695c474508b6fa24d0b2c0cfb86795",
}

# Author-created QA harness, NOT an unchanged original demo or admitted dataset.
FIXTURE = """<!doctype html><meta charset='utf-8'>
<title>H236 source-native offline drag QA only</title>
<style>
body{font:21px Arial;margin:0;background:#fafafa}
main{padding:20px}
section{display:inline-block;vertical-align:top;width:350px;min-height:350px;
border:2px solid black;margin:20px;padding:14px;background:white}
.card{padding:24px;border:1px solid #888;margin:15px;background:#e6eeff;cursor:grab}
.gu-mirror{position:fixed!important;margin:0!important;z-index:9999}
.gu-hide{display:none!important}.gu-unselectable{user-select:none!important}
.gu-transit{opacity:.2}
</style>
<main>
<section id='left' aria-label='Source'><div id='card-one' class='card'>CARD 1</div>
<div id='card-two' class='card'>CARD 2</div></section>
<section id='right' aria-label='Destination'><div id='target' class='card'>TARGET</div></section>
</main>"""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H236 fail-closed: " + reason)


def git(root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *argv], text=True,
        capture_output=True, timeout=15, check=False,
    )
    require(completed.returncode == 0, "Git command failed")
    return completed.stdout.strip()


def preflight(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    require(Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
            "must use exact original Git root")
    require(git(root, "rev-parse", "HEAD") == COMMIT,
            "original Git SHA mismatch")
    require(git(root, "status", "--porcelain", "--untracked-files=all") == "",
            "original checkout must be clean")
    origin = git(root, "remote", "get-url", "origin")
    require(origin.lower() in (
        "https://github.com/bevacqua/dragula",
        "https://github.com/bevacqua/dragula.git",
        "git@github.com:bevacqua/dragula.git",
        "git@github.com:bevacqua/dragula",
    ), "original remote identity mismatch")
    old = (
        set(FROZEN_H173_REPOS.values())
        | set(FROZEN_OTHER_REPOS.values())
        | set(KNOWN_NPM_UPSTREAM_REPOS)
        | set(FROZEN_PRIOR_ORIGINAL_REPOS)  # H168/H169/H171/H172 historical source floor
    )
    from h238.historical_family_floor import known_public_source_family_floor
    require(canonical_repository_family(REPO) not in (old | known_public_source_family_floor()),
            "historical known source family reuse")
    require(not output.exists(), "owner output must be create-only")
    require(not output.is_relative_to(root), "output must leave original source")
    require(not any((p / ".git").exists() for p in (output, *output.parents)),
            "output must be outside all Git worktrees")
    for relative, digest in PINNED.items():
        p = root.joinpath(*relative.split("/"))
        require(p.is_file() and not p.is_symlink() and
                p.resolve().is_relative_to(root) and
                sha256(p.read_bytes()).hexdigest() == digest,
                "original public asset SHA mismatch: " + relative)
    require(b"MIT" in (root / "license").read_bytes(),
            "root MIT license missing")
    return {
        "original_source": root,
        "asset": root / "dist" / "dragula.min.js",
        "output": output,
    }


def replay(root: Path, output: Path) -> dict:
    checked = preflight(root, output)
    from playwright.sync_api import sync_playwright

    output = checked["output"]
    result = {
        "status": "H236_ORIGINAL_DRAGULA_NATIVE_QA_NOT_ADMISSION",
        "source_repository": REPO,
        "source_git_sha": COMMIT,
        "source_sha256": dict(PINNED),
        "full_history_and_media_rights_audited": False,
        "historical_rgba_history_complete": False,
        "official_train_dev_admitted": False,
        "new_ood_approved": False,
        "model_calls": 0,
        "optimizer_updates": 0,
        "contexts": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        with sync_playwright() as pw:
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            kwargs = {"headless": True}
            if edge.is_file():
                kwargs["executable_path"] = str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("H236 needs already installed Chromium/Edge")
            browser = pw.chromium.launch(**kwargs)
            try:
                for i, lang in enumerate(("en", "ar"), start=1):
                    ctx = browser.new_context(
                        viewport={"width": 960, "height": 720},
                        device_scale_factor=1, service_workers="block",
                    )
                    attempted = []
                    ctx.route("**/*", lambda route: (
                        attempted.append(route.request.url), route.abort()))
                    page = ctx.new_page()
                    page.set_content(FIXTURE, wait_until="load")
                    page.add_script_tag(path=str(checked["asset"]))
                    # Invoke the authentic original JS public constructor;
                    # no DOM identity appears in any model or training input.
                    page.evaluate("""() => window.dragula([
                        document.querySelector('#left'),
                        document.querySelector('#right')
                    ])""")
                    before = page.screenshot(type="png")
                    card = page.locator("#card-one")
                    right = page.locator("#right")
                    card.drag_to(right, timeout=7000)
                    require(page.locator("#right > #card-one").count() == 1,
                            "native original Dragula did not move source card")
                    require(page.locator("#left > #card-one").count() == 0,
                            "source retained moved card")
                    require(len(attempted) == 0,
                            "unexpected network or file request")
                    after = page.screenshot(type="png")
                    require(_rgba_sha(before) != _rgba_sha(after),
                            "decoded screenshot did not change")
                    for key, png in (("before", before), ("after", after)):
                        (output / f"case-{i}-{key}.png").write_bytes(png)
                    result["contexts"].append({
                        "context": i,
                        "instruction_language": lang,
                        "original_native_dom_drop_verified": True,
                        "network_attempts": 0,
                        "before_png_sha256": sha256(before).hexdigest(),
                        "before_rgba_sha256": _rgba_sha(before),
                        "after_png_sha256": sha256(after).hexdigest(),
                        "after_rgba_sha256": _rgba_sha(after),
                    })
                    ctx.close()
            finally:
                browser.close()
        require(len(result["contexts"]) == 2, "exactly two independent contexts")
        (output / "receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except BaseException:
        (output / "INCOMPLETE_NO_ADMISSION").write_text(
            "H236 capture incomplete; never TRAIN/DEV/OOD.\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    x = replay(args.source_root, args.output)
    print(json.dumps({
        "status": x["status"], "contexts": len(x["contexts"]),
        "model_calls": 0, "admitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
