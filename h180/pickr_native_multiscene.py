"""H180 source-only original Pickr TWO semantic swatches × TWO contexts.

This is an ephemeral public-donor Playwright demonstration, not TRAIN/DEV or
OOD. No screenshots, prompts, labels, coordinates or user data are persisted,
and no model or paid compute is touched. Reuse the existing H174 exact donor
build validator, isolated browser/network routing and original GUI selectors.
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

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
    verify_derived_assets,
)

SCENES = (
    {"id": "pickr-swatch-0", "language": "en", "swatch_index": 0,
     "instruction": "Select the first available color swatch in the classic picker."},
    {"id": "pickr-swatch-1", "language": "ar", "swatch_index": 1,
     "instruction": "اختر عينة اللون الثانية في منتقي الألوان الكلاسيكي."},
)
WIDTH, HEIGHT = 960, 720
REPEATS = 2
SCENE_FIELDS = frozenset({
    "id", "language", "swatch_index", "instruction",
})


@dataclass(frozen=True)
class NativeScene:
    id: str
    language: str
    swatch_index: int
    before: str
    after: str
    rgba_sha256: str
    png_sha256: str


def validate_specs(specs: tuple[dict, ...] = SCENES) -> None:
    _require(type(specs) is tuple and len(specs) == 2,
             "H180 requires exactly two predeclared original source scenes")
    ids = set()
    swatches = set()
    languages = set()
    for spec in specs:
        _require(type(spec) is dict and set(spec) == SCENE_FIELDS,
                 "H180 scene schema drift")
        _require(type(spec["id"]) is str and spec["id"].startswith("pickr-swatch-"),
                 "H180 scene identity mismatch")
        _require(type(spec["swatch_index"]) is int
                 and spec["swatch_index"] in (0, 1),
                 "H180 only original known first/second swatches allowed")
        _require(spec["id"] == f"pickr-swatch-{spec['swatch_index']}",
                 "H180 source scene id and swatch disagree")
        _require(type(spec["language"]) is str
                 and spec["language"] in ("ar", "en"),
                 "H180 bilingual source research requirement")
        _require(type(spec["instruction"]) is str
                 and len(spec["instruction"].strip()) >= 15,
                 "H180 native source task instruction missing")
        _require(spec["id"] not in ids
                 and spec["swatch_index"] not in swatches
                 and spec["language"] not in languages,
                 "H180 duplicate scene, swatch or language")
        ids.add(spec["id"])
        swatches.add(spec["swatch_index"])
        languages.add(spec["language"])
    _require(swatches == {0, 1} and languages == {"en", "ar"},
             "H180 bilingual original swatch fixture lost")


def verify_native_scenes(rows: tuple[NativeScene, ...]) -> dict:
    """Pure owner-data-free verification of real source response and replay.

    No supplied synthetic hash rows can certify *real* browser execution:
    only the browser runner is permitted to call this as its final check.
    """
    validate_specs()
    _require(type(rows) is tuple and len(rows) == len(SCENES)*REPEATS,
             "H180 exact two-source-scene replay missing")
    by_id: dict[str, list[NativeScene]] = {}
    for row in rows:
        _require(type(row) is NativeScene,
                 "H180 exact named original scene receipt required")
        _require(row.id in {s["id"] for s in SCENES},
                 "H180 unknown original task")
        expected = next(s for s in SCENES if s["id"] == row.id)
        _require(row.language == expected["language"]
                 and row.swatch_index == expected["swatch_index"],
                 "H180 original scene identity changed")
        _require(type(row.before) is str and type(row.after) is str
                 and bool(row.before) and bool(row.after)
                 and row.before != row.after,
                 "H180 original swatch did not change native result value")
        for digest in (row.rgba_sha256, row.png_sha256):
            _require(type(digest) is str and len(digest) == 64
                     and all(c in "0123456789abcdef" for c in digest),
                     "H180 invalid source screenshot fingerprint")
        by_id.setdefault(row.id, []).append(row)
    _require(set(by_id) == {s["id"] for s in SCENES},
             "H180 original task missing")
    for runs in by_id.values():
        _require(len(runs) == REPEATS
                 and runs[0].before == runs[1].before
                 and runs[0].after == runs[1].after
                 and runs[0].rgba_sha256 == runs[1].rgba_sha256
                 and runs[0].png_sha256 == runs[1].png_sha256,
                 "H180 fresh-context independent PNG/RGBA replay differs")
    distinct_rgba = {runs[0].rgba_sha256 for runs in by_id.values()}
    distinct_png = {runs[0].png_sha256 for runs in by_id.values()}
    distinct_effects = {runs[0].after for runs in by_id.values()}
    _require(len(distinct_rgba) == len(SCENES)
             and len(distinct_png) == len(SCENES)
             and len(distinct_effects) == len(SCENES),
             "H180 two original scenes are visually or semantically duplicate")
    return {
        "status": "H180_ORIGINAL_PICKR_TWO_SCENES_NATIVE_ONLY_NOT_ADMISSION",
        "source_native_semantic_successes": len(rows),
        "independent_contexts_per_scene": REPEATS,
        "distinct_native_rgba_scenes": len(distinct_rgba),
        "distinct_png_scenes": len(distinct_png),
        "languages": len({row.language for row in rows}),
        "old_ood_media_used": False,
        "train_dev_ood_admitted": False,
        "model_calls": 0,
        "native_rewards_model": 0,
    }


def _browser_one_scene(browser, port: int, spec: dict) -> NativeScene:
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
                 "H180 original built Pickr page unavailable")
        _require(page.get_by_role("button", name="classic").count() == 1,
                 "H180 original classic picker missing")
        popup = page.locator(".pcr-app.visible")
        page.locator(".pcr-button").first.click()
        popup.wait_for(state="visible", timeout=5000)
        result = popup.locator(".pcr-result")
        swatches = popup.locator(".pcr-swatches button")
        _require(result.count() == 1
                 and swatches.count() >= len(SCENES),
                 "H180 original swatches/result missing")
        before = result.input_value()
        swatches.nth(spec["swatch_index"]).click()
        after = result.input_value()
        _require(after != before, "H180 original color selection was not native")
        page.mouse.move(0, 0)
        page.wait_for_timeout(350)
        png = page.screenshot(animations="disabled")
        image = Image.open(BytesIO(png)).convert("RGBA")
        _require(image.size == (WIDTH, HEIGHT),
                 "H180 original viewport unexpected")
        _require(not errors, "H180 original browser reported JS errors")
        return NativeScene(
            id=spec["id"], language=spec["language"],
            swatch_index=spec["swatch_index"], before=before, after=after,
            rgba_sha256=sha256(image.tobytes()).hexdigest(),
            png_sha256=sha256(png).hexdigest(),
        )
    finally:
        context.close()


def run_native_multiscene(dist_dir: Path, channel: str) -> dict:
    """Real GUI actions and four fresh browser contexts; screenshots in RAM."""
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
                rows = tuple(
                    _browser_one_scene(browser, server.server_port, spec)
                    for spec in SCENES for _ in range(REPEATS)
                )
                return verify_native_scenes(rows)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H180 original Pickr two colors x 2 contexts; no datasets"
    )
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    result = run_native_multiscene(args.dist_dir, args.browser_channel)
    print(
        f"{result['status']} "
        f"native_actions={result['source_native_semantic_successes']} "
        f"distinct_rgba={result['distinct_native_rgba_scenes']} "
        f"languages={result['languages']} "
        "NO_DATASET_ADMISSION_NO_MODEL_CALLS_NO_UPLOAD"
    )


if __name__ == "__main__":
    main()
