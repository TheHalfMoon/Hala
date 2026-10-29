"""H192 original medium-zoom source-native CLOSE action from PRE-action images.

A third independent original UI implementation: real medium-zoom MIT
example, two distinct expanded photographs with the click-to-close task
visible BEFORE the target click. Original photo opening is a SEED action;
closing enlarged original photo is a real native target outcome. No data,
screenshots, model weights, gold, history OOD or files are persisted.

This original medium-zoom source was already research-exposed in H174
and is RESEARCH_EXPOSED_UNASSIGNED in H182. It is not an eligible
held-out OOD, TRAIN, or DEV family without a new independent registry.
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
)
from h174.mediumzoom_native_smoke import verify_mediumzoom_dist
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)

WIDTH, HEIGHT = 960, 720
REPEATS = 2
SCENES = (
    {"id": "original-zoom-close-first", "photo_index": 0, "language": "en",
     "instruction": "Close the enlarged first photograph."},
    {"id": "original-zoom-close-second", "photo_index": 1, "language": "ar",
     "instruction": "أغلق الصورة الثانية المكبّرة."},
)
SCENE_FIELDS = frozenset({"id", "photo_index", "language", "instruction"})


@dataclass(frozen=True)
class OriginalZoomPreaction:
    id: str
    photo_index: int
    language: str
    original_photo_opened: bool
    original_photo_closed: bool
    private_native_close_center: tuple[float, float]
    preaction_png_sha256: str
    preaction_rgba_sha256: str


def validate_original_specs(specs: tuple[dict, ...] = SCENES) -> None:
    _require(type(specs) is tuple and len(specs) == 2
             and specs == SCENES
             and all(type(s) is dict and set(s) == SCENE_FIELDS for s in specs),
             "H192 exact original already-exposed two-photo source fixture required")
    _require({s["photo_index"] for s in specs} == {0, 1}
             and {s["language"] for s in specs} == {"en", "ar"},
             "H192 original two photographs bilingual task coverage missing")


def _sha_ok(s: str) -> bool:
    return type(s) is str and len(s) == 64 and all(
        ch in "0123456789abcdef" for ch in s
    )


def validate_preaction_replay(
    receipts: tuple[OriginalZoomPreaction, ...],
    owner_old_sha_metadata: dict | None = None,
) -> dict:
    """Only original real browser runner qualifies native actions."""
    validate_original_specs()
    _require(type(receipts) is tuple and len(receipts) == len(SCENES)*REPEATS,
             "H192 incomplete original native two-photo context replay")
    grouped: dict[str, list[OriginalZoomPreaction]] = {}
    for row in receipts:
        _require(type(row) is OriginalZoomPreaction,
                 "H192 original source-native record required, not fake gold")
        expected = next((s for s in SCENES if s["id"] == row.id), None)
        _require(expected is not None and row.language == expected["language"]
                 and row.photo_index == expected["photo_index"],
                 "H192 source original photo/language identity changed")
        _require(row.original_photo_opened is True
                 and row.original_photo_closed is True,
                 "H192 real original zoom-open seed/close reward not proven")
        _require(type(row.private_native_close_center) is tuple
                 and len(row.private_native_close_center) == 2
                 and all(type(v) is float and 0 <= v < limit
                         for v, limit in zip(
                             row.private_native_close_center, (WIDTH, HEIGHT)
                         )),
                 "H192 original zoom close target is not real visible geometry")
        _require(_sha_ok(row.preaction_png_sha256)
                 and _sha_ok(row.preaction_rgba_sha256),
                 "H192 original PRE-action PNG/RGBA digest malformed")
        grouped.setdefault(row.id, []).append(row)
    _require(set(grouped) == {s["id"] for s in SCENES},
             "H192 original two-photo source cases missing")
    for rows in grouped.values():
        _require(len(rows) == REPEATS and rows[0] == rows[1],
                 "H192 original zoom PRE-action independent context differs")
    distinct = [grouped[s["id"]][0] for s in SCENES]
    _require(len({r.preaction_png_sha256 for r in distinct}) == 2
             and len({r.preaction_rgba_sha256 for r in distinct}) == 2,
             "H192 different original zoom photographs have duplicate pixels")
    if owner_old_sha_metadata is not None:
        screen_unadmitted_new_case_hashes([
            {"image_sha256": r.preaction_png_sha256,
             "visual_rgba_sha256": r.preaction_rgba_sha256}
            for r in distinct
        ], owner_old_sha_metadata)
    return {
        "status": "H192_ORIGINAL_MEDIUMZOOM_PREACTION_CLOSE_NATIVE_ONLY",
        "independent_original_photo_scenes": 2,
        "fresh_browser_contexts_per_photo": REPEATS,
        "native_open_close_action_successes": len(receipts)*2,
        "distinct_preaction_decoded_rgba": 2,
        "original_upstream_was_already_research_exposed": True,
        "source_family_role": "RESEARCH_EXPOSED_UNASSIGNED",
        "old_known_png_shas_screened": owner_old_sha_metadata is not None,
        "images_persisted": 0,
        "reference_gold_persisted": 0,
        "train_dev_examples_admitted": 0,
        "ood_images_created": 0,
        "model_calls": 0,
        "gpu_steps": 0,
        "complete_historical_visual_registry_proven": False,
    }


def _one_native_close(browser, port: int, spec: dict) -> OriginalZoomPreaction:
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
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        result = page.goto(
            f"http://127.0.0.1:{port}/",
            wait_until="networkidle", timeout=15000,
        )
        _require(result is not None and result.status == 200,
                 "H192 original medium-zoom HTML not loaded")
        photos = page.locator("article.container img")
        _require(photos.count() == 2,
                 "H192 original two source photographs missing")
        photo = photos.nth(spec["photo_index"])
        photo.wait_for(state="visible", timeout=10000)
        _require(photo.evaluate("(p) => p.complete && p.naturalWidth > 0"),
                 "H192 original source image not decoded")
        photo.click()
        opened = page.locator(".medium-zoom-image--opened")
        opened.wait_for(state="visible", timeout=10000)
        overlay = page.locator(".medium-zoom-overlay")
        overlay.wait_for(state="visible", timeout=10000)
        _require(opened.count() == 1 and not js_errors,
                 "H192 original source zoom-open seed failed")
        page.mouse.move(0, 0)
        page.wait_for_timeout(650)
        box = opened.bounding_box()
        _require(box is not None,
                 "H192 original native zoomed image close target missing")
        center = (
            float(round(box["x"] + box["width"] / 2, 4)),
            float(round(box["y"] + box["height"] / 2, 4)),
        )
        _require(0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT,
                 "H192 original close target center outside viewport")
        preaction_png = page.screenshot(animations="disabled")
        with Image.open(BytesIO(preaction_png)) as image:
            _require(image.format == "PNG"
                     and image.size == (WIDTH, HEIGHT),
                     "H192 genuine native PRE-action screenshot invalid")
            preaction_rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        # Only after image acquisition: original clicked zoomed photo
        # should close its own native overlay. No script injecting rewards.
        opened.click(timeout=10000)
        overlay.wait_for(state="hidden", timeout=10000)
        _require(opened.count() == 0
                 and not js_errors,
                 "H192 original source zoom-close result not observed")
        return OriginalZoomPreaction(
            id=spec["id"], photo_index=spec["photo_index"],
            language=spec["language"],
            original_photo_opened=True,
            original_photo_closed=True,
            private_native_close_center=center,
            preaction_png_sha256=sha256(preaction_png).hexdigest(),
            preaction_rgba_sha256=preaction_rgba,
        )
    finally:
        context.close()


def run_original_zoom_close(
    dist_dir: Path, channel: str,
    owner_history_root: Path | None = None,
) -> dict:
    validate_original_specs()
    root = verify_mediumzoom_dist(dist_dir)
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(Quiet, directory=str(root)),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel=channel, headless=True,
            )
            try:
                rows = tuple(
                    _one_native_close(browser, server.server_port, s)
                    for s in SCENES for _ in range(REPEATS)
                )
            finally:
                browser.close()
        history = (
            scan_legacy_manifest_sha_fields([owner_history_root])
            if owner_history_root is not None else None
        )
        return validate_preaction_replay(rows, history)
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    cli = argparse.ArgumentParser(
        description="H192 actual original medium-zoom close gesture; source only"
    )
    cli.add_argument("--dist-dir", required=True, type=Path)
    cli.add_argument("--browser-channel", default="chromium")
    cli.add_argument("--historical-root", type=Path)
    args = cli.parse_args()
    result = run_original_zoom_close(
        args.dist_dir, args.browser_channel, args.historical_root,
    )
    print(
        f"{result['status']} "
        f"original_photos={result['independent_original_photo_scenes']} "
        f"real_source_actions={result['native_open_close_action_successes']} "
        f"distinct_pre_rgba={result['distinct_preaction_decoded_rgba']} "
        f"owner_sha_screened={int(result['old_known_png_shas_screened'])} "
        "RESEARCH_EXPOSED_UNASSIGNED_NO_DATASET_NO_OOD_NO_MODEL_NO_MEDIA"
    )


if __name__ == "__main__":
    main()
