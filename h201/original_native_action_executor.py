"""H201: execute a strict H200 model-action JSON as native GUI pointer events.

Copy H199/H200 exact native action grammar and H193 pinned original
interact.js native slider reward. This is an *executor smoke*, with
hardcoded non-model research actions only. No training, scores, private
owner source images, gold, model weights, network egress or OOD.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from h199.native_action_sft import parse_actions, WIDTH, HEIGHT
from h193.interact_snap_preaction import SPECS, REPEATS, _rgba_digest

ORIGINAL_ROUTE = "/snap/index.html"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H201 original native action execution: " + why)


def exact_original_loopback_url(port: int) -> str:
    require(type(port) is int and 1 <= port <= 65535,
            "exact loopback original source port required")
    return f"http://127.0.0.1:{port}{ORIGINAL_ROUTE}"


def execute_one_native_action(page: Any, raw: str, expected_url: str) -> dict:
    """Only native browser pointer/keyboard, no DOM/evaluate/selector access.

    Reject unregistered page destinations before parsing/executing actions.
    This executor deliberately scopes to the pinned original SNAP demo
    until an audited app allowlist and real permission UI exist.
    """
    require(type(expected_url) is str
            and expected_url.startswith("http://127.0.0.1:")
            and expected_url.endswith(ORIGINAL_ROUTE)
            and expected_url.count("/") == 4,
            "only registered original static source URL may receive actions")
    require(page.url == expected_url, "browser left the original app page")
    require(page.viewport_size == {"width": WIDTH, "height": HEIGHT},
            "native screenshot coordinate viewport not 960x720")
    # H200 rejects duplicate/unicode-escaped keys and NaN/Infinity, not
    # just model actions that happen to parse as valid Python objects.
    action = parse_actions(raw)["actions"][0]
    kind = action["kind"]
    if kind == "abstain":
        return {"status": "ABSTAIN_NO_BROWSER_INPUT", "native_events": 0,
                "kind": "abstain"}
    if kind == "click":
        x, y = action["point"]
        page.mouse.click(x, y)
        count = 1
    elif kind == "drag":
        start = action["start"]
        end = action["end"]
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*end, steps=12)
        page.mouse.up()
        count = 4
    elif kind == "key_press":
        # A real browser click, not a hidden DOM focus() or JavaScript
        # value setter, focuses the model-visible control first.
        x, y = action["point"]
        page.mouse.click(x, y)
        page.keyboard.press(action["key"])
        count = 2
    else:
        raise ValueError("H201 impossible action kind")
    return {"status": "NATIVE_GUI_INPUT_DISPATCHED",
            "native_events": count, "kind": kind}


@dataclass(frozen=True)
class SnapNativeActionReceipt:
    task_id: str
    language: str
    before_input: str
    after_input: str
    other_input: str
    before_png_sha256: str
    before_rgba_sha256: str
    after_rgba_sha256: str
    before_grid_sha256: str
    after_grid_sha256: str
    target_center: tuple[float, float]
    native_events: int


def validate_original_execution(rows: tuple[SnapNativeActionReceipt, ...]) -> dict:
    require(type(rows) is tuple and len(rows) == len(SPECS)*REPEATS,
            "missing fresh original independent task replay")
    groups: dict[str, list[SnapNativeActionReceipt]] = {}
    for r in rows:
        require(type(r) is SnapNativeActionReceipt, "untrusted native receipt")
        spec = next((s for s in SPECS if s["id"] == r.task_id), None)
        require(spec is not None and r.language == spec["language"],
                "original source task identity changed")
        require(r.before_input == "300" and r.after_input == "301"
                and r.other_input == "499" and r.native_events == 2,
                "original slider input reward not achieved by executor")
        require(r.before_rgba_sha256 != r.after_rgba_sha256
                and r.before_grid_sha256 != r.after_grid_sha256,
                "original visible scene or native grid canvas did not redraw")
        require(all(type(v) is str and len(v) == 64 for v in (
            r.before_png_sha256, r.before_rgba_sha256,
            r.after_rgba_sha256, r.before_grid_sha256,
            r.after_grid_sha256,
        )), "invalid visual source identity")
        require(type(r.target_center) is tuple and
                len(r.target_center) == 2 and
                all(type(x) is float for x in r.target_center),
                "native model pixel target is not bounded")
        require(0 <= r.target_center[0] < WIDTH
                and 0 <= r.target_center[1] < HEIGHT,
                "native target absent from model screenshot viewport")
        groups.setdefault(r.task_id, []).append(r)
    require(set(groups) == {x["id"] for x in SPECS},
            "missing one original bilingual source task")
    for group in groups.values():
        require(len(group) == REPEATS and group[0] == group[1],
                "same original preaction input/reward differs across fresh contexts")
    first = [groups[s["id"]][0] for s in SPECS]
    require(len({x.before_png_sha256 for x in first}) == 2
            and len({x.before_rgba_sha256 for x in first}) == 2,
            "different seeded source tasks share identical GUI images")
    return {
        "status": "H201_ORIGINAL_NATIVE_JSON_ACTION_EXECUTOR_REWARD_PASS",
        "original_source_tasks": len(SPECS),
        "fresh_contexts": len(rows),
        "real_browser_input_events": sum(r.native_events for r in rows),
        "unique_preaction_source_images": len(first),
        "changed_real_original_grid_canvases": len(rows),
        "source_data_admitted": 0,
        "media_saved": 0,
        "model_calls": 0,
        "gpu_steps": 0,
        "old_ood_read": False,
    }


def _one_snap_original(browser: Any, port: int, spec: dict) -> SnapNativeActionReceipt:
    from h174.pickr_native_smoke import allow_loopback_request
    # Inherit the H196 source browser egress model: block external HTTP
    # and WS before opening any page; disable service workers/downloads.
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
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        resp = page.goto(exact_original_loopback_url(port),
                         wait_until="networkidle", timeout=18000)
        require(resp is not None and resp.status == 200,
                "original pinned SNAP demo source not loaded")
        seed = page.locator("#" + spec["seed"])
        target = page.locator("#" + spec["target"])
        grid = page.locator("#grid")
        seed.wait_for(state="visible", timeout=12000)
        target.wait_for(state="visible", timeout=12000)
        require(grid.count() == 1 and seed.input_value() == "300"
                and target.input_value() == "300",
                "original source controls missing or seed drifted")
        # This is scenario SETUP, not claimed model-inferred actions.
        seed.focus()
        seed.press("End")
        seed.press("ArrowLeft")
        require(seed.input_value() == "499" and target.input_value() == "300",
                "original native source scenario preparation did not succeed")
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        box = target.bounding_box()
        require(box is not None, "original source target slider not visible")
        # A range input's bounding-box midpoint is NOT necessarily the
        # visible thumb: the pinned original min=20/max=500, value=300,
        # and real mouse-click-at-center changes the value to 260.
        # Original Edge research measured a thumb hit 7–17px right of
        # midpoint that PRESERVES 300 before ArrowRight yields 301.
        # This is a source-native FIXTURE coordinate, not a learned
        # model prediction nor data automatically admitted to training.
        center = (
            float(round(box["x"] + box["width"]*0.5 + 12.0, 4)),
            float(round(box["y"] + box["height"]*0.5, 4)),
        )
        require(0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT,
                "source target not screenshot-visible")
        before_png = page.screenshot(animations="disabled")
        before_rgba = _rgba_digest(before_png)
        grid_before = sha256(grid.evaluate(
            "(canvas) => canvas.toDataURL('image/png')"
        ).encode("ascii")).hexdigest()
        # This hardcoded known source target action is a genuine dispatch
        # integration test, NOT a Hala learned model prediction.
        raw = json.dumps({"actions": [{
            "kind": "key_press",
            "point": [center[0], center[1]],
            "key": "ArrowRight",
        }]}, separators=(",", ":"))
        execution = execute_one_native_action(
            page, raw, exact_original_loopback_url(port),
        )
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        after_png = page.screenshot(animations="disabled")
        after_rgba = _rgba_digest(after_png)
        grid_after = sha256(grid.evaluate(
            "(canvas) => canvas.toDataURL('image/png')"
        ).encode("ascii")).hexdigest()
        require(execution["native_events"] == 2
                and target.input_value() == "301"
                and seed.input_value() == "499"
                and before_rgba != after_rgba
                and grid_before != grid_after
                and not errors,
                "H199 JSON→real browser pointer/keyboard did not reward original app")
        return SnapNativeActionReceipt(
            task_id=spec["id"], language=spec["language"],
            before_input="300", after_input=target.input_value(),
            other_input=seed.input_value(),
            before_png_sha256=sha256(before_png).hexdigest(),
            before_rgba_sha256=before_rgba,
            after_rgba_sha256=after_rgba,
            before_grid_sha256=grid_before,
            after_grid_sha256=grid_after,
            target_center=center,
            native_events=execution["native_events"],
        )
    finally:
        context.close()


def real_original_native_source_smoke(port: int, channel: str) -> dict:
    from playwright.sync_api import sync_playwright

    exact_original_loopback_url(port)
    require(channel in {"chromium", "msedge"},
            "source replay only on free Chromium or installed Edge")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        try:
            rows = tuple(_one_snap_original(browser, port, spec)
                         for spec in SPECS for _ in range(REPEATS))
        finally:
            browser.close()
    return validate_original_execution(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H201 actual original GUI JSON executor smoke; never model/data"
    )
    parser.add_argument("--port", type=int, default=8201)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    result = real_original_native_source_smoke(args.port, args.browser_channel)
    print(
        f"{result['status']} "
        f"original_tasks={result['original_source_tasks']} "
        f"contexts={result['fresh_contexts']} "
        f"native_events={result['real_browser_input_events']} "
        f"redrawn_canvases={result['changed_real_original_grid_canvases']} "
        "NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD"
    )


if __name__ == "__main__":
    main()
