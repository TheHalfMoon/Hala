"""H193 original interact.js SNAP sliders: PRE-action native keyboard UI proof.

Copy-first H174 isolated Playwright + H189 source GUI replay pattern.
Uses unchanged pinned MIT taye/interact.js@a993eb8c... original
examples/snap, already exposed DEV upstream. Two independent native
grid-spacing tasks, each first seeded via the OTHER original slider to
create a different genuine visual starting screen, then screenshot
BEFORE changing the target slider using a real ArrowRight key action.

Requires native source app input value changes AND screenshot pixel
change from its own original canvas rendering. No media/gold persisted,
no model training, no OOD and no source family reclassification.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from typing import Callable

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
)
from h185.manifest_sha_only import screen_unadmitted_new_case_hashes

WIDTH, HEIGHT = 960, 720
REPEATS = 2
SPECS = (
    {
        "id": "interact-original-grid-x-slider",
        "language": "en",
        "seed": "grid-y", "target": "grid-x",
        "instruction": "Increase the grid's horizontal spacing by one step.",
    },
    {
        "id": "interact-original-grid-y-slider",
        "language": "ar",
        "seed": "grid-x", "target": "grid-y",
        "instruction": "زد التباعد الرأسي للشبكة خطوة واحدة.",
    },
)
SOURCE_FIELDS = frozenset({"id", "language", "seed", "target", "instruction"})


@dataclass(frozen=True)
class OriginalSnapPreaction:
    id: str
    language: str
    seeded_slider: str
    target_slider: str
    before_target_value: int
    after_target_value: int
    seeded_slider_value: int
    native_slider_center: tuple[float, float]
    preaction_png_sha256: str
    preaction_rgba_sha256: str
    postaction_rgba_sha256: str
    preaction_original_grid_canvas_sha256: str
    postaction_original_grid_canvas_sha256: str


def validate_specs(specs: tuple[dict, ...] = SPECS) -> None:
    _require(type(specs) is tuple and specs == SPECS and len(specs) == 2
             and all(type(s) is dict and set(s) == SOURCE_FIELDS for s in specs),
             "H193 exact original interactive snap slider scenarios required")
    _require({s["seed"] for s in specs} == {"grid-x", "grid-y"}
             and {s["target"] for s in specs} == {"grid-x", "grid-y"}
             and {s["language"] for s in specs} == {"en", "ar"}
             and all(s["seed"] != s["target"] for s in specs),
             "H193 source-native original grid spacing tasks not independent")


def _sha(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def validate_replay(
    rows: tuple[OriginalSnapPreaction, ...],
    old_sha_metadata: dict | None = None,
) -> dict:
    """Strict native receipts. Real JS actions must be run in browser."""
    validate_specs()
    _require(type(rows) is tuple and len(rows) == len(SPECS)*REPEATS,
             "H193 missing original source GUI replay")
    groups: dict[str, list[OriginalSnapPreaction]] = {}
    for row in rows:
        _require(type(row) is OriginalSnapPreaction,
                 "H193 untrusted original source native receipt")
        spec = next((s for s in SPECS if row.id == s["id"]), None)
        _require(spec is not None and row.language == spec["language"]
                 and row.seeded_slider == spec["seed"]
                 and row.target_slider == spec["target"],
                 "H193 original source slider/seed/language drift")
        _require(type(row.before_target_value) is int
                 and type(row.after_target_value) is int
                 and type(row.seeded_slider_value) is int
                 and row.before_target_value == 300
                 and row.after_target_value == 301
                 and row.seeded_slider_value == 499,
                 "H193 source-native target/seed slider reward invalid")
        point = row.native_slider_center
        _require(type(point) is tuple and len(point) == 2
                 and all(type(x) is float and isfinite(x)
                         for x in point)
                 and 0 <= point[0] < WIDTH
                 and 0 <= point[1] < HEIGHT,
                 "H193 source-native target slider not in GUI viewport")
        _require(_sha(row.preaction_png_sha256)
                 and _sha(row.preaction_rgba_sha256)
                 and _sha(row.postaction_rgba_sha256)
                 and _sha(row.preaction_original_grid_canvas_sha256)
                 and _sha(row.postaction_original_grid_canvas_sha256)
                 and row.preaction_rgba_sha256 != row.postaction_rgba_sha256
                 and row.preaction_original_grid_canvas_sha256
                     != row.postaction_original_grid_canvas_sha256,
                 "H193 original slider did not change actual source-rendered grid canvas")
        groups.setdefault(row.id, []).append(row)
    _require(set(groups) == {s["id"] for s in SPECS},
             "H193 one original application source task missing")
    for repeats in groups.values():
        _require(len(repeats) == REPEATS and repeats[0] == repeats[1],
                 "H193 fresh original browser replay PNG/RGBA/native geometry drift")
    unique = [groups[s["id"]][0] for s in SPECS]
    _require(len({r.preaction_png_sha256 for r in unique}) == 2
             and len({r.preaction_rgba_sha256 for r in unique}) == 2,
             "H193 both original seed visual states have duplicate pixels")
    if old_sha_metadata is not None:
        screen_unadmitted_new_case_hashes([
            {
                "image_sha256": r.preaction_png_sha256,
                "visual_rgba_sha256": r.preaction_rgba_sha256,
            } for r in unique
        ], old_sha_metadata)
    return {
        "status": "H193_ORIGINAL_INTERACT_SNAP_PREACTION_SLIDER_NATIVE_ONLY",
        "original_visual_source_tasks": 2,
        "fresh_browser_contexts_per_task": REPEATS,
        "real_native_seed_and_target_keyboard_actions": len(rows)*3,
        "distinct_preaction_rgba": 2,
        "original_source_family_role": "PROSPECTIVE_DEV_NOT_ADMITTED",
        "owner_old_sha_checked": old_sha_metadata is not None,
        "saved_images": 0,
        "saved_private_gold": 0,
        "train_or_dev_cases_admitted": 0,
        "old_ood_read": False,
        "model_calls": 0,
        "gpu_steps": 0,
        "complete_prior_raster_history_proven": False,
    }


def _rgba_digest(png: bytes) -> str:
    from PIL import Image

    with Image.open(BytesIO(png)) as im:
        _require(im.format == "PNG" and im.size == (WIDTH, HEIGHT),
                 "H193 genuine original GUI screenshot size/format mismatch")
        return sha256(im.convert("RGBA").tobytes()).hexdigest()


def _one_native_slider(
    browser, port: int, spec: dict,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> OriginalSnapPreaction:
    context = create_isolated_smoke_context(browser)
    try:
        context.route(
            "**/*",
            lambda route: route.continue_()
            if allow_loopback_request(route.request.url, port)
            else route.abort(),
        )
        page = context.new_page()
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        resp = page.goto(
            f"http://127.0.0.1:{port}/snap/index.html",
            wait_until="networkidle", timeout=18000,
        )
        _require(resp is not None and resp.status == 200,
                 "H193 original snap GUI did not load")
        seed = page.locator("#" + spec["seed"])
        target = page.locator("#" + spec["target"])
        seed.wait_for(state="visible", timeout=12000)
        target.wait_for(state="visible", timeout=12000)
        _require(page.locator("#grid").count() == 1
                 and page.locator("#drag").count() == 1
                 and seed.input_value() == "300"
                 and target.input_value() == "300",
                 "H193 original source grid controls/runtime not initialized")
        # Two ACTUAL keyboard events on the other native original slider:
        # first End, then ArrowLeft -> 499. No JS/DOM value patching.
        seed.focus()
        seed.press("End")
        seed.press("ArrowLeft")
        _require(seed.input_value() == "499"
                 and target.input_value() == "300",
                 "H193 original source-native preaction seed did not succeed")
        page.mouse.move(0, 0)
        page.wait_for_timeout(400)
        bbox = target.bounding_box()
        _require(bbox is not None, "H193 original target slider not visible")
        center = (
            float(round(bbox["x"] + bbox["width"]/2, 4)),
            float(round(bbox["y"] + bbox["height"]/2, 4)),
        )
        _require(0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT,
                 "H193 target range control outside model-visible screenshot")
        before_png = page.screenshot(animations="disabled")
        before_rgba = _rgba_digest(before_png)
        original_grid = page.locator("#grid")
        grid_before = sha256(
            original_grid.evaluate("(canvas) => canvas.toDataURL('image/png')").encode("ascii")
        ).hexdigest()
        # Source-native task via real browser keyboard INPUT, never a
        # fake DOM value assignment or direct canvas drawing.
        target.focus()
        target.press("ArrowRight")
        _require(target.input_value() == "301"
                 and seed.input_value() == "499",
                 "H193 real original snap slider target did not update")
        page.mouse.move(0, 0)
        page.wait_for_timeout(400)
        after_png = page.screenshot(animations="disabled")
        after_rgba = _rgba_digest(after_png)
        grid_after = sha256(
            original_grid.evaluate("(canvas) => canvas.toDataURL('image/png')").encode("ascii")
        ).hexdigest()
        _require(before_rgba != after_rgba
                 and grid_before != grid_after and not js_errors,
                 "H193 original grid canvas did not repaint or browser emitted JS error")
        # H194 optional owner-PRIVATE sink receives the ORIGINAL preaction
        # PNG and key/target geometry only AFTER genuine source reward
        # passed. Public GitHub CI never enables the sink.
        if capture_preaction is not None:
            capture_preaction(
                dict(spec), bytes(before_png),
                {
                    "action_type": "key_press", "key": "ArrowRight",
                    "slider": spec["target"],
                    "target_center": [center[0], center[1]],
                },
            )
        return OriginalSnapPreaction(
            id=spec["id"], language=spec["language"],
            seeded_slider=spec["seed"], target_slider=spec["target"],
            before_target_value=300, after_target_value=301,
            seeded_slider_value=499, native_slider_center=center,
            preaction_png_sha256=sha256(before_png).hexdigest(),
            preaction_rgba_sha256=before_rgba,
            postaction_rgba_sha256=after_rgba,
            preaction_original_grid_canvas_sha256=grid_before,
            postaction_original_grid_canvas_sha256=grid_after,
        )
    finally:
        context.close()


def run_original_snap(
    port: int, channel: str,
    owner_old_sha_metadata: dict | None = None,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> dict:
    validate_specs()
    _require(type(port) is int and 1 <= port <= 65535,
             "H193 only original pinned source served on local loopback")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel, headless=True)
        try:
            rows = tuple(
                _one_native_slider(browser, port, spec, capture_preaction)
                for spec in SPECS for _ in range(REPEATS)
            )
        finally:
            browser.close()
    return validate_replay(rows, owner_old_sha_metadata)


def main() -> None:
    cli = argparse.ArgumentParser(
        description="H193 original interact.js snap native grid slider PRE-action, research only"
    )
    cli.add_argument("--port", type=int, default=8193)
    cli.add_argument("--browser-channel", default="chromium")
    args = cli.parse_args()
    proof = run_original_snap(args.port, args.browser_channel)
    print(
        f"{proof['status']} "
        f"original_tasks={proof['original_visual_source_tasks']} "
        f"native_key_actions={proof['real_native_seed_and_target_keyboard_actions']} "
        f"distinct_pre_rgba={proof['distinct_preaction_rgba']} "
        "NO_MEDIA_NO_TRAIN_DEV_ADMISSION_NO_OOD_NO_MODEL"
    )


if __name__ == "__main__":
    main()
