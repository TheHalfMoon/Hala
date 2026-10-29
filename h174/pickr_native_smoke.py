"""H174 Pickr owner-only scratch-build native smoke. Never creates datasets.

The source donor must be separately verified at an immutable Git SHA.
Only generated scratch build assets are served; screenshots stay in memory.
"""
from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

from h174.source_candidates import _reject_reparse


class _Entrypoint(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.styles: list[str] = []
        self.has_base = False
        self.inline_scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fields = dict(attrs)
        if tag == "base":
            self.has_base = True
        if tag == "script":
            if fields.get("src"):
                self.scripts.append(fields["src"])
            else:
                self.inline_scripts += 1
        if tag == "link" and "stylesheet" in (fields.get("rel") or "").lower().split() and fields.get("href"):
            self.styles.append(fields["href"])


def allow_loopback_request(url: str, port: int) -> bool:
    """Reject non-loopback requests, credentials, alternate ports and schemes."""
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                and parsed.port == port and parsed.username is None
                and parsed.password is None)
    except ValueError:
        return False


def _require(condition: bool, message: str) -> None:
    """Fail closed even when Python assertions are disabled with -O."""
    if not condition:
        raise ValueError(message)


def verify_derived_assets(dist_dir: Path) -> dict[str, str]:
    """Check generated JS/CSS assets exist inside the scratch build, not donor Git."""
    _reject_reparse(dist_dir)
    root = dist_dir.resolve(strict=True)
    _require(root.is_dir() and not dist_dir.is_symlink(), "Invalid derived build root")
    html = root / "index.html"
    _reject_reparse(html)
    _require(html.is_file() and not html.is_symlink(), "Missing built index.html")
    parser = _Entrypoint()
    parser.feed(html.read_text(encoding="utf-8"))
    parser.close()
    _require(not parser.has_base, "HTML base URL not permitted")
    _require(parser.inline_scripts == 0, "Unexpected inline script in derived HTML")
    _require(len(parser.scripts) == 1, "Expected exactly one built script")
    local_styles = [s for s in parser.styles if not urlsplit(s).scheme]
    _require(len(local_styles) == 1, "Expected exactly one built stylesheet")
    assets: dict[str, str] = {}
    for kind, ref in [("js", parser.scripts[0]), ("css", local_styles[0])]:
        _require(ref.startswith("./assets/") and "%" not in ref and chr(92) not in ref, "Unsafe generated URL")
        parsed = urlsplit(ref)
        _require(not (parsed.query or parsed.fragment or parsed.scheme or parsed.netloc), "Unsafe URL parts")
        parts = Path(ref[2:]).parts
        _require(".." not in parts, "Path traversal")
        target = root.joinpath(*parts)
        _reject_reparse(target)
        resolved = target.resolve(strict=True)
        _require(resolved.is_relative_to(root) and target.is_file() and not target.is_symlink(), "Asset escaped root")
        _require(resolved.stat().st_size > 0, "Empty generated asset")
        assets[kind] = sha256(resolved.read_bytes()).hexdigest()
    return assets


def run_native_smoke(dist_dir: Path, browser_channel: str) -> tuple[str, dict[str, str]]:
    """Real original GUI click+swatch; independent fresh contexts and exact RGBA."""
    assets = verify_derived_assets(dist_dir)
    # Lazy imports: CI synthetic tests never launch a browser or read media.
    from PIL import Image
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(dist_dir.resolve(strict=True)))
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=browser_channel, headless=True)
            try:
                digests: list[str] = []
                for _ in range(2):
                    context = browser.new_context(
                        viewport={"width": 960, "height": 720},
                        device_scale_factor=1,
                        locale="en-US", reduced_motion="reduce",
                    )
                    try:
                        context.route(
                            "**/*",
                            lambda route: route.continue_()
                            if allow_loopback_request(route.request.url, server.server_port)
                            else route.abort(),
                        )
                        page = context.new_page()
                        errors: list[str] = []
                        page.on("pageerror", lambda err: errors.append(str(err)))
                        response = page.goto(
                            f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle"
                        )
                        _require(response is not None and response.status == 200, "Original demo response not OK")
                        _require(page.get_by_role("button", name="classic").count() == 1, "Original classic picker missing")
                        button = page.locator(".pcr-button")
                        _require(button.count() >= 1, "Original picker missing")
                        button.first.click()
                        page.locator(".pcr-app.visible").wait_for(timeout=5000)
                        swatches = page.locator(".pcr-app.visible .pcr-swatches button")
                        _require(swatches.count() >= 1, "Original swatches missing")
                        result = page.locator(".pcr-app.visible .pcr-result")
                        _require(result.count() == 1, "Original result input missing")
                        before = result.input_value()
                        swatches.first.click()
                        _require(result.input_value() != before, "Native swatch did not change color")
                        page.mouse.move(0, 0)
                        page.wait_for_timeout(350)
                        rgba = Image.open(BytesIO(page.screenshot(animations="disabled"))).convert("RGBA")
                        _require(rgba.size == (960, 720), "Original screenshot dimension mismatch")
                        digests.append(sha256(rgba.tobytes()).hexdigest())
                        _require(not errors, f"JS page errors: {errors}")
                    finally:
                        context.close()
                _require(digests[0] == digests[1], "Independent RGBA replay mismatch"
                return digests[0], assets
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="H174 Pickr scratch-build smoke; NOT dataset QA")
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--browser-channel", default="msedge")
    args = parser.parse_args()
    rgba_sha, assets = run_native_smoke(args.dist_dir, args.browser_channel)
    print(f"PICKR_NATIVE_2X_RGBA_PASS rgba_sha256={rgba_sha} assets_sha256={assets}")
    print("SOURCE_PIN_AND_BUILD_RECEIPTS_REQUIRED; NO TRAIN/DEV/OOD ADMISSION")


if __name__ == "__main__":
    main()
