"""H174 source-native interact.js dropzone smoke, never OOD/data admission.

This runs ONLY against pinned upstream's original Vite-served dropzones demo
on a free public Linux runner. Browser frames remain in RAM. No GPU/model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
)


def run_native_dropzone_smoke(port: int, channel: str) -> str:
    _require(type(port) is int and 1 <= port <= 65535,
             "H174 interactjs invalid localhost port")
    from PIL import Image
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        try:
            hashes = []
            for _ in range(2):
                context = create_isolated_smoke_context(browser)
                try:
                    context.route(
                        "**/*",
                        lambda route: route.continue_()
                        if allow_loopback_request(route.request.url, port)
                        else route.abort(),
                    )
                    page = context.new_page()
                    errors = []
                    page.on("pageerror", lambda err: errors.append(str(err)))
                    resp = page.goto(
                        f"http://127.0.0.1:{port}/dropzones/index.html",
                        wait_until="networkidle", timeout=20000,
                    )
                    _require(resp is not None and resp.status == 200,
                             "H174 original interactjs HTML failed")
                    drag = page.locator("#drag3")
                    drop = page.locator("#drop1")
                    _require(drag.count() == 1 and drop.count() == 1,
                             "H174 original interactjs drag/drop controls missing")
                    _require(drag.inner_text().strip().startswith("Drag me"),
                             "H174 original interactjs initial drag state missing")
                    source = drag.bounding_box()
                    target = drop.bounding_box()
                    _require(source is not None and target is not None,
                             "H174 original drag or drop box not visible")
                    sx = source["x"] + source["width"]/2
                    sy = source["y"] + source["height"]/2
                    tx = target["x"] + target["width"]/2
                    ty = target["y"] + target["height"]/2
                    page.mouse.move(sx, sy)
                    page.mouse.down()
                    page.mouse.move(tx, ty, steps=30)
                    page.mouse.up()
                    page.wait_for_function(
                        "() => document.querySelector('#drag3')?.textContent?.trim() === 'Dropped'",
                        timeout=8000,
                    )
                    _require(drag.inner_text().strip() == "Dropped",
                             "H174 original native drop event not delivered")
                    page.mouse.move(0, 0)
                    page.wait_for_timeout(400)
                    rgba = Image.open(BytesIO(
                        page.screenshot(animations="disabled")
                    )).convert("RGBA")
                    _require(rgba.size == (960, 720),
                             "H174 native drop original viewport changed")
                    _require(not errors,
                             "H174 original interactjs page errors: " + repr(errors))
                    hashes.append(sha256(rgba.tobytes()).hexdigest())
                finally:
                    context.close()
            _require(len(hashes) == 2 and hashes[0] == hashes[1],
                     "H174 original interactjs independent RGBA replay mismatch")
            return hashes[0]
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="H174 original dropzone smoke only")
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    digest = run_native_dropzone_smoke(args.port, args.browser_channel)
    print(f"H174_INTERACT_ORIGINAL_NATIVE_DROP_PASS rgba_sha256={digest} "
          "independent_contexts=2")
    print("NO_TRAIN_DEV_OOD_ADMISSION_NO_MODEL_CALLS_NO_MEDIA_UPLOAD")


if __name__ == "__main__":
    main()
