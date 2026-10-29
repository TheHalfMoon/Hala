"""H187 original Pickr *pre-action* visual grounding source candidates.

Unlike H180's post-action screenshot, H187 captures the actual scene a
GUI-use model would see BEFORE executing the target color selection.
Each task starts from a different real original color-swatch-selected
state. Every source-native target action must be successful, and 2 fresh
contexts must reproduce exact PNG *and decoded RGBA*. Screenshots,
reference target geometry, instruction strings and PNG bytes remain in
RAM ONLY. No TRAIN/DEV/one-shot-OOD case admission and no model calls.

Source app verified by existing immutable H174 original source/build
provenance: simonwep/pickr@e8ff88d9161ffc6e3c46a4430be8c328c42d6415.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Callable

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
    verify_derived_assets,
)
from h180.pickr_native_multiscene import SCENES as H180_ORIGINAL_SWATCH_SCENES
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)

WIDTH, HEIGHT = 960, 720
REPEATS = 2
# The seed swatch is deliberately the OTHER original native swatch so
# that two task screenshots depict genuinely different app states.
SPECS = tuple(
    {
        "id": old["id"], "language": old["language"],
        "target_index": old["swatch_index"],
        "seed_index": 1 - old["swatch_index"],
        "instruction": old["instruction"],
    } for old in H180_ORIGINAL_SWATCH_SCENES
)
SPEC_FIELDS = frozenset({
    "id", "language", "target_index", "seed_index", "instruction",
})


@dataclass(frozen=True)
class PreactionNativeReceipt:
    id: str
    language: str
    seed_index: int
    target_index: int
    before_native_color: str
    after_native_color: str
    before_png_sha256: str
    before_rgba_sha256: str


def validate_specs(specs: tuple[dict, ...] = SPECS) -> None:
    _require(type(specs) is tuple and len(specs) == 2,
             "H187 exact two new source preaction tasks required")
    _require(all(type(s) is dict and set(s) == SPEC_FIELDS for s in specs),
             "H187 source task fields drift")
    _require(specs == SPECS,
             "H187 immutable original source tasks, language and native seed required")
    _require(all(s["seed_index"] != s["target_index"]
                 and s["seed_index"] in (0, 1)
                 and s["target_index"] in (0, 1)
                 for s in specs),
             "H187 identical native baseline and target swatch forbidden")


def validate_replayed_preaction_scenes(
    receipts: tuple[PreactionNativeReceipt, ...],
    prior_sha_census: dict | None = None,
) -> dict:
    """Pure metadata proof; ONLY real Playwright run may qualify native GUI."""
    validate_specs()
    _require(type(receipts) is tuple and len(receipts) == len(SPECS)*REPEATS,
             "H187 incomplete source preaction scene replay")
    by_id: dict[str, list[PreactionNativeReceipt]] = {}
    for row in receipts:
        _require(type(row) is PreactionNativeReceipt,
                 "H187 strict source-native preaction receipt only")
        spec = next((s for s in SPECS if s["id"] == row.id), None)
        _require(spec is not None and row.language == spec["language"]
                 and row.seed_index == spec["seed_index"]
                 and row.target_index == spec["target_index"],
                 "H187 original app source task attribution changed")
        _require(type(row.before_native_color) is str
                 and type(row.after_native_color) is str
                 and bool(row.before_native_color.strip())
                 and row.before_native_color != row.after_native_color,
                 "H187 original source-native action did not change selected color")
        _require(all(
            type(d) is str and len(d) == 64
            and all(c in "0123456789abcdef" for c in d)
            for d in (row.before_png_sha256, row.before_rgba_sha256)
        ), "H187 invalid source-original screenshot SHA")
        by_id.setdefault(row.id, []).append(row)
    _require(set(by_id) == {s["id"] for s in SPECS},
             "H187 one of two native tasks missing")
    for rows in by_id.values():
        _require(len(rows) == REPEATS and rows[0] == rows[1],
                 "H187 two independent preaction GUI contexts disagree")
    distinct_png = {rows[0].before_png_sha256 for rows in by_id.values()}
    distinct_rgba = {rows[0].before_rgba_sha256 for rows in by_id.values()}
    _require(len(distinct_png) == len(SPECS)
             and len(distinct_rgba) == len(SPECS),
             "H187 native seed did not yield distinct preaction visual states")
    # Optional extra owner-only SHA history collision screen. It can
    # rule out KNOWN old SHA reuse, never prove complete historical coverage.
    if prior_sha_census is not None:
        screen_unadmitted_new_case_hashes(
            [
                {
                    "image_sha256": rows[0].before_png_sha256,
                    "visual_rgba_sha256": rows[0].before_rgba_sha256,
                }
                for rows in by_id.values()
            ],
            prior_sha_census,
        )
    return {
        "status": "H187_REAL_PREACTION_PICKR_SOURCE_REPLAY_ONLY_NOT_TRAIN_DATA",
        "original_preaction_scenes": len(SPECS),
        "original_native_action_successes": len(receipts),
        "independent_browser_contexts_per_scene": REPEATS,
        "distinct_preaction_pngs": len(distinct_png),
        "distinct_preaction_decoded_rgba": len(distinct_rgba),
        "known_owner_history_shas_screened": prior_sha_census is not None,
        "historical_source_and_visual_completeness_proven": False,
        "train_dev_images_persisted": 0,
        "train_dev_examples_admitted": 0,
        "new_ood_images_persisted": 0,
        "model_calls": 0,
        "model_training_approved": False,
    }


def _one_browser_preaction(
    browser, port: int, spec: dict,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> PreactionNativeReceipt:
    from PIL import Image

    context = create_isolated_smoke_context(browser)
    try:
        context.route(
            "**/*",
            lambda route: route.continue_()
            if allow_loopback_request(route.request.url, port)
            else route.abort(),
        )
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        response = page.goto(
            f"http://127.0.0.1:{port}/", wait_until="networkidle",
            timeout=15000,
        )
        _require(response is not None and response.status == 200,
                 "H187 original Pickr source demo did not load")
        page.locator(".pcr-button").first.click()
        popup = page.locator(".pcr-app.visible")
        popup.wait_for(state="visible", timeout=8000)
        swatches = popup.locator(".pcr-swatches button")
        result = popup.locator(".pcr-result")
        _require(swatches.count() >= 2 and result.count() == 1,
                 "H187 original Pickr source-native swatches/result missing")
        # Build a preaction visible state using an ORIGINAL native action,
        # not a fake overlay/pixel manipulation/DOM style patch.
        swatches.nth(spec["seed_index"]).click()
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        selected_before = result.input_value()
        _require(bool(selected_before), "H187 original seed not selected")
        # Capture only the visible original, BEFORE selecting the goal.
        # Coordinates are owner-gold, never present in model input.
        goal_box = swatches.nth(spec["target_index"]).bounding_box()
        _require(goal_box is not None, "H188 original source target geometry missing")
        goal = {
            "x": round(goal_box["x"] + goal_box["width"]/2, 4),
            "y": round(goal_box["y"] + goal_box["height"]/2, 4),
        }
        _require(0 <= goal["x"] < WIDTH and 0 <= goal["y"] < HEIGHT,
                 "H188 original source goal is off viewport")
        png = page.screenshot(animations="disabled")
        with Image.open(BytesIO(png)) as raw:
            image = raw.convert("RGBA")
        _require(image.size == (WIDTH, HEIGHT),
                 "H187 original GUI viewport unexpectedly changed")
        # Only AFTER capturing the source-model input scene do we perform
        # the reference target action and check its actual native result.
        swatches.nth(spec["target_index"]).click()
        selected_after = result.input_value()
        _require(selected_before != selected_after,
                 "H187 original native color target did not change")
        _require(not errors, "H187 original app emitted browser JS errors")
        # An H188 owner-private sink MAY retain preaction image+private goal
        # in RAM, but this callback cannot write anything during public CI.
        if capture_preaction is not None:
            capture_preaction(spec, png, goal)
        return PreactionNativeReceipt(
            id=spec["id"], language=spec["language"],
            seed_index=spec["seed_index"],
            target_index=spec["target_index"],
            before_native_color=selected_before,
            after_native_color=selected_after,
            before_png_sha256=sha256(png).hexdigest(),
            before_rgba_sha256=sha256(image.tobytes()).hexdigest(),
        )
    finally:
        context.close()


def run_source_only(
    dist_dir: Path, channel: str, historical_root: Path | None = None,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> dict:
    """Real source GUI proof, no photo persists; optionally reject old SHA."""
    validate_specs()
    verify_derived_assets(dist_dir)
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(Quiet, directory=str(dist_dir.resolve(strict=True))),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel, headless=True)
            try:
                results = tuple(
                    _one_browser_preaction(
                        browser, server.server_port, spec, capture_preaction
                    )
                    for spec in SPECS for _ in range(REPEATS)
                )
            finally:
                browser.close()
        old = (
            scan_legacy_manifest_sha_fields([historical_root])
            if historical_root is not None else None
        )
        return validate_replayed_preaction_scenes(results, old)
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="H187 source-original Pickr PREACTION preview ONLY; no dataset"
    )
    p.add_argument("--dist-dir", required=True, type=Path)
    p.add_argument("--browser-channel", default="chromium")
    p.add_argument("--historical-root", type=Path)
    a = p.parse_args()
    proof = run_source_only(a.dist_dir, a.browser_channel, a.historical_root)
    print(
        f"{proof['status']} "
        f"native_actions={proof['original_native_action_successes']} "
        f"preaction_scenes={proof['original_preaction_scenes']} "
        f"distinct_pre_rgba={proof['distinct_preaction_decoded_rgba']} "
        f"historical_sha_screened={int(proof['known_owner_history_shas_screened'])} "
        "NO_TRAINING_DATA_PERSISTED_NO_OOD_NO_MODEL"
    )


if __name__ == "__main__":
    main()
