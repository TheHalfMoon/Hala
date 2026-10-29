"""H203: independently reject a stale screenshot on the pinned ORIGINAL SNAP GUI.

Real browser negative control only. The test harness (not an AI model)
uses original DOM source to seed a visible slider change and measure reward.
No real model, training, owner screenshot, historical OOD or media persistence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json

from h174.pickr_native_smoke import allow_loopback_request
from h193.interact_snap_preaction import SPECS, REPEATS
from h201.original_native_action_executor import (
    WIDTH, HEIGHT, execute_attested_native_action,
    exact_original_loopback_url, require,
)


def _original_stale_case(browser, port: int, spec: dict) -> dict:
    context = browser.new_context(
        viewport={"width": WIDTH, "height": HEIGHT},
        device_scale_factor=1,
        reduced_motion="reduce",
        service_workers="block",
        accept_downloads=False,
    )
    try:
        context.route(
            "**/*", lambda route: route.continue_()
            if allow_loopback_request(route.request.url, port)
            else route.abort()
        )
        context.route_web_socket("**/*", lambda ws: ws.close())
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        expected_url = exact_original_loopback_url(port)
        response = page.goto(expected_url, wait_until="networkidle", timeout=18000)
        require(response is not None and response.status == 200,
                "pinned original source not loaded")
        target = page.locator("#" + spec["target"])
        target.wait_for(state="visible", timeout=12000)
        grid = page.locator("#grid")
        require(grid.count() == 1 and target.input_value() == "300",
                "original source slider or grid missing")
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        before = page.screenshot(animations="disabled")
        old_grid_digest = sha256(
            grid.evaluate("(canvas) => canvas.toDataURL('image/png')").encode("ascii")
        ).hexdigest()

        # Test-only ORIGINAL native keyboard input changes the on-screen UI
        # AFTER this screenshot was captured. It is not a model prediction.
        target.focus()
        target.press("ArrowRight")
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        after = page.screenshot(animations="disabled")
        new_grid_digest = sha256(
            grid.evaluate("(canvas) => canvas.toDataURL('image/png')").encode("ascii")
        ).hexdigest()
        require(target.input_value() == "301"
                and sha256(before).digest() != sha256(after).digest()
                and old_grid_digest != new_grid_digest,
                "real ORIGINAL GUI was not changed before stale check")
        box = target.bounding_box()
        require(box is not None, "original target no longer visible")
        click = [
            float(round(box["x"] + box["width"] * 0.5 + 12.0, 4)),
            float(round(box["y"] + box["height"] * 0.5, 4)),
        ]
        raw = json.dumps({
            "actions": [{"kind": "key_press",
                         "point": click, "key": "ArrowRight"}]
        }, separators=(",", ":"))
        rejected = False
        try:
            execute_attested_native_action(page, raw, expected_url, before)
        except ValueError as exc:
            rejected = "preaction screenshot is stale or changed" in str(exc)
        require(rejected, "stale screenshot action was not refused")
        require(target.input_value() == "301" and not errors,
                "stale action dispatched an extra keyboard event or JS failed")
        return {
            "task_id": spec["id"],
            "language": spec["language"],
            "changed_original_grid": True,
            "stale_action_refused": True,
            "model_calls": 0,
            "media_saved": 0,
        }
    finally:
        context.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H203 pinned original stale-image negative browser proof")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--browser-channel", choices=("msedge", "chromium"),
                        default="msedge")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser_channel, headless=True)
        try:
            results = tuple(
                _original_stale_case(browser, args.port, spec)
                for spec in SPECS for _ in range(REPEATS)
            )
        finally:
            browser.close()
    require(len(results) == len(SPECS) * REPEATS
            and all(r["stale_action_refused"] and r["changed_original_grid"]
                    for r in results),
            "not all original stale PNG negative controls passed")
    print(
        "H203_ORIGINAL_NATIVE_STALE_PREACTION_REJECTION_PASS "
        f"original_tasks={len(SPECS)} fresh_contexts={len(results)} "
        f"blocked_stale_actions={len(results)} "
        "NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD"
    )


if __name__ == "__main__":
    main()
