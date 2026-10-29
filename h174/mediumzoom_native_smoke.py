"""H174 original medium-zoom native Linux GUI smoke; no datasets or old OOD.

Never publish images or train models. Source must be independently verified
at the immutable original upstream commit before building in a scratch tree.
"""
from __future__ import annotations

import argparse
import platform
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
)
from h174.source_candidates import _reject_reparse


def verify_mediumzoom_dist(dist_dir: Path) -> Path:
    """Check only scratch build integrity, NEVER claim original file provenance."""
    root = Path(dist_dir)
    _reject_reparse(root)
    resolved = root.resolve(strict=True)
    _require(resolved.is_dir() and not root.is_symlink(),
             "H174 medium-zoom scratch build directory missing")
    index = resolved / "index.html"
    _reject_reparse(index)
    _require(index.is_file() and not index.is_symlink(),
             "H174 medium-zoom generated HTML missing")
    html = index.read_text(encoding="utf-8")
    _require("<title>Medium Zoom | Vanilla JavaScript demo</title>" in html,
             "H174 medium-zoom derived original title mismatch")
    _require("assets/" in html, "H174 medium-zoom Vite assets missing")
    _require(not (resolved / "node_modules").exists(),
             "H174 medium-zoom server root includes dependencies")
    _require(len(tuple(resolved.rglob("*.jpg"))) >= 2,
             "H174 medium-zoom original photo outputs missing")
    # CI uses a public zero-cost ephemeral runner, not owner data.
    return resolved


def run_native_mediumzoom_smoke(dist_dir: Path, channel: str) -> tuple[str, int]:
    """Original photo open; independent fresh-context decoded RGBA replay."""
    root = verify_mediumzoom_dist(dist_dir)
    from PIL import Image
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(root))
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel, headless=True)
            try:
                digests = []
                for _ in range(2):
                    context = create_isolated_smoke_context(browser)
                    try:
                        context.route(
                            "**/*",
                            lambda route: route.continue_()
                            if allow_loopback_request(
                                route.request.url, server.server_port
                            ) else route.abort(),
                        )
                        page = context.new_page()
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        resp = page.goto(
                            f"http://127.0.0.1:{server.server_port}/",
                            wait_until="networkidle",
                        )
                        _require(resp is not None and resp.status == 200,
                                 "H174 medium-zoom original demo HTTP failed")
                        images = page.locator("article.container img")
                        _require(images.count() == 2,
                                 "H174 medium-zoom original images missing")
                        images.first.wait_for(state="visible", timeout=8000)
                        _require(images.first.evaluate(
                            "(img) => img.complete && img.naturalWidth > 0"
                        ), "H174 medium-zoom original photo not decoded")
                        images.first.click()
                        page.locator(".medium-zoom-overlay").wait_for(
                            state="visible", timeout=8000
                        )
                        opened = page.locator(".medium-zoom-image--opened")
                        opened.wait_for(state="visible", timeout=8000)
                        _require(opened.count() == 1,
                                 "H174 medium-zoom original photo not zoomed")
                        page.mouse.move(0, 0)
                        page.wait_for_timeout(650)
                        rgba = Image.open(
                            BytesIO(page.screenshot(animations="disabled"))
                        ).convert("RGBA")
                        _require(rgba.size == (960, 720),
                                 "H174 medium-zoom RGB viewport mismatch")
                        digests.append(sha256(rgba.tobytes()).hexdigest())
                        _require(not errors,
                                 "H174 medium-zoom JS errors: " + repr(errors))
                    finally:
                        context.close()
                _require(len(digests) == 2 and digests[0] == digests[1],
                         "H174 medium-zoom independent original RGBA replay mismatch")
                return digests[0], 2
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="H174 medium-zoom original GUI only")
    p.add_argument("--dist-dir", type=Path, required=True)
    p.add_argument("--browser-channel", default="chromium")
    args = p.parse_args()
    digest, replay = run_native_mediumzoom_smoke(
        args.dist_dir, args.browser_channel
    )
    print(f"H174_MEDIUMZOOM_ORIGINAL_NATIVE_PASS platform={platform.system().lower()} "
          f"rgba_sha256={digest} independent_contexts={replay}")
    print("NO_TRAIN_DEV_OOD_ADMISSION_NO_MODEL_CALLS_NO_MEDIA_UPLOAD")


if __name__ == "__main__":
    main()
