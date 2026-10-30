"""Opt-in synthetic loopback-only browser isolation probe (no owner data).

Run: python -m h243.loopback_probe --native
Uses an ephemeral 127.0.0.1 TCP server and an authored file:// page.
Tests page + Dedicated Worker WebSocket attempts in separate contexts.
Does NOT establish OS-level isolation, WebRTC/UDP isolation, or data admission.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import socketserver
import tempfile
import threading

from h243.browser_offline import require_browser_offline


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.server.accepted += 1
        try:
            self.connection.settimeout(0.5)
            if self.rfile.readline():
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        except (OSError, TimeoutError):
            pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.accepted = 0


def _trial(browser, scratch: Path, offline: bool) -> int:
    with _Server() as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"ws://127.0.0.1:{server.server_address[1]}/synthetic"
        html = """<!doctype html><meta charset="utf-8"><script>
const url = __URL__;
try { new WebSocket(url); } catch(e) {}
try {
  const js = 'try { new WebSocket(' + JSON.stringify(url) + '); } catch(e) {}';
  new Worker(URL.createObjectURL(new Blob([js], {type:'text/javascript'})));
} catch(e) {}
</script>""".replace("__URL__", repr(url))
        page_path = scratch / ("offline.html" if offline else "baseline.html")
        page_path.write_text(html, encoding="utf-8")
        context = browser.new_context(service_workers="block")
        try:
            if offline:
                require_browser_offline(context)
            page = context.new_page()
            page.goto(page_path.as_uri(), wait_until="load", timeout=10000)
            page.wait_for_timeout(1500)
            return server.accepted
        finally:
            context.close()
            server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if not args.native:
        parser.error("explicit --native opt-in required")
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory(prefix="hala-h243-loopback-") as folder:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(channel="msedge", headless=True)
            except Exception:
                browser = pw.chromium.launch(headless=True)
            try:
                baseline = _trial(browser, Path(folder), offline=False)
                guarded = _trial(browser, Path(folder), offline=True)
            finally:
                browser.close()
    print(f"H243_LOOPBACK_NATIVE baseline_accepts={baseline} offline_accepts={guarded}")
    if baseline < 2 or guarded != 0:
        print("H243_LOOPBACK_NATIVE_FAIL")
        return 1
    print("H243_LOOPBACK_NATIVE_PASS research_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
