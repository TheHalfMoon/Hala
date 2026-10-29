"""H191: original Pickr three-theme × four genuine native seed colors.

12 genuinely different MODEL-VISIBLE PRE-action Pickr source scenes;
each image created by exact original app's own theme + color controls.
Two independent isolated Chromium contexts per scene, no media persisted.
No TRAIN/DEV/old-OOD admission or model run. Source upstream immutable:
simonwep/pickr@e8ff88d9161ffc6e3c46a4430be8c328c42d6415.
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
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)

THEMES = ("classic", "monolith", "nano")
SEEDS = (2, 3, 4, 5)
REPEATS = 2
WIDTH, HEIGHT = 960, 720
CASE_COUNT = len(THEMES) * len(SEEDS)
_FIELDS = frozenset({
    "id", "theme", "seed_index", "target_index", "language", "instruction",
})
# Every theme has >=7 original source swatches: www/index.js at pinned Git.
# Each task has a different *visible source color state* and original UI.
# Target selection alternates first/second; bilingual instructions are
# native action goals, not extra text drawn into the screenshots.
SPECS = tuple({
    "id": f"pickr-{theme}-seed-{seed}",
    "theme": theme,
    "seed_index": seed,
    "target_index": seed % 2,
    "language": "en" if seed % 2 == 0 else "ar",
    "instruction": (
        f"In the {theme} color picker, select the first color swatch."
        if seed % 2 == 0 else
        f"في منتقي الألوان {theme}، اختر عينة اللون الثانية."
    ),
} for theme in THEMES for seed in SEEDS)


@dataclass(frozen=True)
class OriginalThemePreaction:
    id: str
    theme: str
    seed_index: int
    target_index: int
    language: str
    before_native_color: str
    after_native_color: str
    original_target_center: tuple[float, float]
    preaction_png_sha256: str
    preaction_rgba_sha256: str


def validate_specs(specs: tuple[dict, ...] = SPECS) -> None:
    _require(type(specs) is tuple and len(specs) == CASE_COUNT,
             "H191 requires all original-theme preaction source scenarios")
    _require(all(type(s) is dict and set(s) == _FIELDS
                 for s in specs) and specs == SPECS,
             "H191 original theme/seed/target task specification drift")
    _require(len({s["id"] for s in specs}) == CASE_COUNT,
             "H191 repeated native scene ID")
    _require({s["theme"] for s in specs} == set(THEMES)
             and {s["seed_index"] for s in specs} == set(SEEDS)
             and {s["language"] for s in specs} == {"en", "ar"},
             "H191 original source themes/seeds/language coverage missing")
    _require(all(s["target_index"] != s["seed_index"]
                 and s["target_index"] in (0, 1) for s in specs),
             "H191 target may not be same original seed color")


def _sha_ok(s: str) -> bool:
    return type(s) is str and len(s) == 64 and all(
        c in "0123456789abcdef" for c in s
    )


def validate_replay(
    results: tuple[OriginalThemePreaction, ...],
    history_sha_metadata: dict | None = None,
) -> dict:
    validate_specs()
    _require(type(results) is tuple and len(results) == CASE_COUNT * REPEATS,
             "H191 missing fresh original-theme replay scene")
    rows: dict[str, list[OriginalThemePreaction]] = {}
    for row in results:
        _require(type(row) is OriginalThemePreaction,
                 "H191 strict original native scene receipt only")
        spec = next((s for s in SPECS if s["id"] == row.id), None)
        _require(spec is not None and row.theme == spec["theme"]
                 and row.seed_index == spec["seed_index"]
                 and row.target_index == spec["target_index"]
                 and row.language == spec["language"],
                 "H191 original app theme/seed/target identity changed")
        _require(type(row.before_native_color) is str
                 and type(row.after_native_color) is str
                 and bool(row.before_native_color)
                 and row.before_native_color != row.after_native_color,
                 "H191 original native color target did not change")
        _require(type(row.original_target_center) is tuple
                 and len(row.original_target_center) == 2
                 and all(type(v) is float and 0 <= v < bound
                         for v, bound in zip(
                             row.original_target_center, (WIDTH, HEIGHT)
                         )),
                 "H191 real source target geometry invalid")
        _require(_sha_ok(row.preaction_png_sha256)
                 and _sha_ok(row.preaction_rgba_sha256),
                 "H191 original source PNG or decoded RGBA SHA malformed")
        rows.setdefault(row.id, []).append(row)
    _require(set(rows) == {spec["id"] for spec in SPECS},
             "H191 original theme/seed task missing")
    for repeated in rows.values():
        _require(len(repeated) == REPEATS and repeated[0] == repeated[1],
                 "H191 independent context preaction PNG/RGBA/geometry drift")
    first = [rows[spec["id"]][0] for spec in SPECS]
    _require(len({r.preaction_png_sha256 for r in first}) == CASE_COUNT
             and len({r.preaction_rgba_sha256 for r in first}) == CASE_COUNT,
             "H191 different original theme/seed UI states have duplicate pixels")
    if history_sha_metadata is not None:
        screen_unadmitted_new_case_hashes([
            {
                "image_sha256": r.preaction_png_sha256,
                "visual_rgba_sha256": r.preaction_rgba_sha256,
            } for r in first
        ], history_sha_metadata)
    return {
        "status": "H191_REAL_PICKR_THREE_ORIGINAL_THEMES_TWELVE_PREACTION_REPLAY",
        "original_theme_count": len(THEMES),
        "genuine_visible_seed_colors_per_theme": len(SEEDS),
        "unique_original_preaction_scenes": CASE_COUNT,
        "fresh_contexts_per_scene": REPEATS,
        "original_native_seed_and_target_actions": len(results) * 2,
        "distinct_png_preaction": CASE_COUNT,
        "distinct_rgba_preaction": CASE_COUNT,
        "owner_old_image_sha_screened": history_sha_metadata is not None,
        "official_train_cases_admitted": 0,
        "official_dev_cases_admitted": 0,
        "old_ood_accessed": False,
        "model_calls": 0,
        "gpu_optimizer_steps": 0,
        "historical_source_complete": False,
        "old_decoded_rgba_history_complete": False,
    }


def _one_context(
    browser, port: int, spec: dict,
    on_real_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> OriginalThemePreaction:
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
                 "H191 original immutable Pickr demo unreachable")
        # Original website exposes these three buttons in www/index.js,
        # never a synthetic in-page element or overlay.
        theme = page.get_by_role("button", name=spec["theme"], exact=True)
        _require(theme.count() == 1,
                 "H191 original theme selector not present")
        theme.click()
        button = page.locator(".pcr-button").first
        button.wait_for(state="visible", timeout=8000)
        button.click()
        popup = page.locator(".pcr-app.visible")
        popup.wait_for(state="visible", timeout=8000)
        swatches = popup.locator(".pcr-swatches button")
        result = popup.locator(".pcr-result")
        _require(swatches.count() >= 7 and result.count() == 1,
                 "H191 source-native colored swatches not available")
        swatches.nth(spec["seed_index"]).click()
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        before = result.input_value()
        _require(bool(before), "H191 source-native seed color absent")
        box = swatches.nth(spec["target_index"]).bounding_box()
        _require(box is not None, "H191 target swatch native box not visible")
        target = (
            float(round(box["x"] + box["width"] / 2, 4)),
            float(round(box["y"] + box["height"] / 2, 4)),
        )
        _require(0 <= target[0] < WIDTH and 0 <= target[1] < HEIGHT,
                 "H191 actual original target not inside preaction viewport")
        png = page.screenshot(animations="disabled")
        with Image.open(BytesIO(png)) as image:
            _require(image.format == "PNG" and image.size == (WIDTH, HEIGHT),
                     "H191 real preaction browser PNG size changed")
            rgba_sha = sha256(image.convert("RGBA").tobytes()).hexdigest()
        # Source-native reward AFTER capturing actual input scene.
        swatches.nth(spec["target_index"]).click()
        after = result.input_value()
        _require(bool(after) and after != before and not errors,
                 "H191 original app target failed or JS page error")
        if on_real_preaction is not None:
            on_real_preaction(
                dict(spec), bytes(png),
                {"x": target[0], "y": target[1]},
            )
        return OriginalThemePreaction(
            id=spec["id"], theme=spec["theme"],
            seed_index=spec["seed_index"],
            target_index=spec["target_index"],
            language=spec["language"],
            before_native_color=before, after_native_color=after,
            original_target_center=target,
            preaction_png_sha256=sha256(png).hexdigest(),
            preaction_rgba_sha256=rgba_sha,
        )
    finally:
        context.close()


def run_original_theme_batch(
    dist_dir: Path, browser_channel: str,
    history_root: Path | None = None,
    capture: Callable[[dict, bytes, dict], None] | None = None,
) -> dict:
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
            browser = p.chromium.launch(
                channel=browser_channel, headless=True,
            )
            try:
                rows = tuple(
                    _one_context(browser, server.server_port, spec, capture)
                    for spec in SPECS for _ in range(REPEATS)
                )
            finally:
                browser.close()
        previous = (
            scan_legacy_manifest_sha_fields([history_root])
            if history_root is not None else None
        )
        return validate_replay(rows, previous)
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H191 3 ORIGINAL Pickr themes × 4 PRE-action color seeds; NO TRAIN"
    )
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--browser-channel", default="chromium")
    parser.add_argument("--historical-root", type=Path)
    args = parser.parse_args()
    receipt = run_original_theme_batch(
        args.dist_dir, args.browser_channel, args.historical_root,
    )
    print(
        f"{receipt['status']} "
        f"preaction_scenes={receipt['unique_original_preaction_scenes']} "
        f"native_seed_target_actions={receipt['original_native_seed_and_target_actions']} "
        f"distinct_rgba={receipt['distinct_rgba_preaction']} "
        f"old_sha_screened={int(receipt['owner_old_image_sha_screened'])} "
        "NO_DATA_ADMISSION_NO_MODEL_NO_OOD_NO_MEDIA_PERSISTED"
    )


if __name__ == "__main__":
    main()
