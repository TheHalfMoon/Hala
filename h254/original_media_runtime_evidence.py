"""H254: public original-app runtime media evidence, never dataset admission."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

from h243.browser_offline import require_browser_offline
from h253.original_app_asset_rights import audit


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True, timeout=20).stdout.strip()


def within_root(url: str, root: Path) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return False
    try:
        path = Path(url2pathname(unquote(parsed.path))).resolve()
        return path.is_relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False


def history_status(shallow: bool, commits: list[str]) -> str:
    if shallow or not commits:
        return "INCOMPLETE_GIT_HISTORY"
    return "HISTORY_VISIBLE_RIGHTS_UNPROVEN"


def summarize(static: dict, *, shallow: bool, history: dict,
              images: list[dict], blocked: int, remote_responses: int,
              file_escapes: int, websocket_attempts: int) -> dict:
    if static["status"] != "INCOMPLETE_NO_ADMISSION" or not static["pinned_source_verified"]:
        raise ValueError("Source preflight failed")
    if remote_responses or file_escapes or websocket_attempts:
        raise ValueError("Forbidden browser activity")
    if any(not isinstance(x.get("loaded"), bool) or not isinstance(x.get("local"), bool)
           for x in images):
        raise ValueError("Malformed image observation")
    return {
        "status": "INCOMPLETE_NO_ADMISSION",
        "original_source_pinned": True,
        "browser_offline_enabled": True,
        "static_local_media_references": len(static["referenced_local_media"]),
        "static_external_runtime_urls": len(static["remote_runtime_dependencies"]),
        "static_inline_svg_elements": static["inline_svg_elements"],
        "browser_image_elements": len(images),
        "browser_loaded_local_images": sum(x["loaded"] and x["local"] for x in images),
        "browser_failed_or_nonlocal_images": sum(not (x["loaded"] and x["local"]) for x in images),
        "blocked_external_http_requests_observed": blocked,
        "external_http_responses_observed": remote_responses,
        "out_of_scope_file_requests_observed": file_escapes,
        "websocket_attempts_observed": websocket_attempts,
        "git_checkout_shallow": shallow,
        "referenced_media_history_status": {
            path: history_status(shallow, history.get(path, []))
            for path in static["referenced_local_media"]
        },
        "asset_authorship_and_transitive_rights_proven": False,
        "complete_prior_source_and_rgba_history_proven": False,
        "train_dev_ood_admitted": False,
        "model_calls": 0, "optimizer_steps": 0,
    }


def observe(root: Path, sha: str, entry: str) -> dict:
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Full pinned Git SHA required")
    static = audit(root, sha, [entry])
    shallow = git(root, "rev-parse", "--is-shallow-repository") == "true"
    history = {path: git(root, "log", "--follow", "--format=%H", "--", path).splitlines()
               for path in static["referenced_local_media"]}

    from playwright.sync_api import sync_playwright

    blocked, responses, escapes, websockets = [], [], [], []
    with sync_playwright() as pw:
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        if not edge.is_file():
            raise RuntimeError("Installed Edge required")
        browser = pw.chromium.launch(executable_path=str(edge), headless=True)
        try:
            context = browser.new_context(viewport={"width": 960, "height": 720},
                                          device_scale_factor=1, service_workers="block")
            require_browser_offline(context)
            context.on("response", lambda response: responses.append(1)
                       if urlsplit(response.url).scheme in {"http", "https"} else None)
            context.on("websocket", lambda socket: websockets.append(1))

            def guard(route):
                url = route.request.url
                scheme = urlsplit(url).scheme
                if scheme in {"http", "https", "ws", "wss"}:
                    blocked.append(1)
                    route.abort()
                elif scheme == "file" and within_root(url, root):
                    route.continue_()
                else:
                    escapes.append(1)
                    route.abort()

            context.route("**/*", guard)
            page = context.new_page()
            page.goto((root / entry).as_uri(), wait_until="load", timeout=20000)
            page.wait_for_timeout(300)
            images = page.evaluate("""() => Array.from(document.images, el => ({
                loaded: el.complete && el.naturalWidth > 0 && el.naturalHeight > 0,
                local: el.currentSrc.startsWith('file://')
            }))""")
            context.close()
        finally:
            browser.close()
    return summarize(static, shallow=shallow, history=history, images=images,
                     blocked=len(blocked), remote_responses=len(responses),
                     file_escapes=len(escapes), websocket_attempts=len(websockets))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--entry", required=True)
    args = p.parse_args()
    try:
        result = observe(args.repo, args.sha, args.entry)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "INCOMPLETE_NO_ADMISSION", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
