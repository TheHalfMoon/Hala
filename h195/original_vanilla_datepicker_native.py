"""H195 original vanilla JS datepicker source-native GUI action, research only.

A fourth actual source implementation is preflighted against old ORIGINAL
source identities, NOT admitted as TRAIN/DEV/OOD. Gantt OOD never opened.
Copy-first H174 pinned MIT source and H193 bilingual PRE-action replay.
No external browser requests, no images/targets persisted, no model/GPU.

Original source: mymth/vanillajs-datepicker at exact frozen commit.
The original demo expects highlight.js exclusively for code formatting;
QA uses an isolated no-op highlighting helper (not an app/date/reward
shim), because all outside-CDN requests are blocked.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import subprocess
from threading import Thread
from typing import Callable

from h174.source_candidates import _reject_reparse
from h179.source_novelty import screen_pins_against_history
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)

PIN = {
    "repo": "mymth/vanillajs-datepicker",
    "git_sha": "358dfaaa3e2b8772e313d71191720f5415c9275c",
    "html_path": "demo/index.html",
    "html_sha256": "f2eeaa6393ed504765fa124991d57ae9bb8aeed5057865517b898e2af5aa247a",
    "license_path": "LICENSE",
    "license_sha256": "fc128e54c03aa96a27de02d0bb9c468a6869faca42c62942abcdffa25eaa3810",
    "license": "MIT",
}
SOURCE_KEYS = "vanilla-datepicker"
WIDTH, HEIGHT, REPEATS = 960, 720, 2
SCENES = (
    {"id": "datepicker-current-month-day15", "language": "en",
     "next_month_clicks": 0, "day": 15,
     "expected_value": "10/15/2026",
     "instruction": "Choose October 15, 2026 in the open calendar."},
    {"id": "datepicker-next-month-day22", "language": "ar",
     "next_month_clicks": 1, "day": 22,
     "expected_value": "11/22/2026",
     "instruction": "اختر يوم 22 نوفمبر 2026 في التقويم المفتوح."},
)
_SCENE_FIELDS = frozenset({
    "id", "language", "next_month_clicks", "day",
    "expected_value", "instruction",
})
_FROZEN_TIME = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H195 original datepicker PRE-action: " + why)


def verify_original_source(
    donor: Path, historical_root: Path | None = None,
) -> dict:
    _require(type(donor) is Path and donor.is_absolute()
             and donor.is_dir(),
             "original Git clone root required")
    _reject_reparse(donor)
    _reject_reparse(donor / ".git")
    _require((donor / ".git").is_dir(), "original .git directory missing")
    def git(*cmd: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(donor), *cmd],
                text=True, stderr=subprocess.DEVNULL,
                timeout=20,
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise ValueError("source Git command unavailable") from e
    _require(git("rev-parse", "HEAD") == PIN["git_sha"],
             "original datepicker Git pin drift")
    _require(not git("status", "--porcelain", "--untracked-files=all"),
             "original datepicker donor is dirty")
    _require(Path(git("rev-parse", "--show-toplevel")).resolve() == donor.resolve(),
             "original Git root path drift")
    _require(git("remote", "get-url", "origin") in {
        "https://github.com/mymth/vanillajs-datepicker.git",
        "https://github.com/mymth/vanillajs-datepicker",
        "git@github.com:mymth/vanillajs-datepicker.git",
    }, "original upstream remote unexpected")
    for rel, expected in (
        (PIN["html_path"], PIN["html_sha256"]),
        (PIN["license_path"], PIN["license_sha256"]),
    ):
        f = donor / rel
        _reject_reparse(f)
        _require(f.is_file()
                 and git("ls-files", "--error-unmatch", "--", rel) == rel
                 and sha256(f.read_bytes()).hexdigest() == expected,
                 "pinned original HTML/MIT LICENSE identity drift")
    for rel in (
        "demo/live-demo.js", "dist/js/datepicker-full.js",
        "dist/css/datepicker-bulma.css",
    ):
        f = donor / rel
        _reject_reparse(f)
        _require(f.is_file()
                 and git("ls-files", "--error-unmatch", "--", rel) == rel
                 and git("rev-parse", f"HEAD:{rel}") ==
                 subprocess.check_output(
                     ["git", "hash-object", str(f)],
                     text=True, stderr=subprocess.DEVNULL, timeout=20,
                 ).strip(),
                 "original demo JS, compiled datepicker or CSS provenance drift")
    source = screen_pins_against_history(
        {SOURCE_KEYS: PIN},
        [historical_root] if historical_root is not None else [],
    )
    _require(source["candidate_source_families"] == 1
             and source["training_approved"] is False
             and source["source_history_completeness_proven"] is False,
             "source novelty research may not admit training")
    return {
        "status": "H195_FOURTH_PINNED_ORIGINAL_DATEPICKER_SOURCE_ONLY",
        "historical_git_checkouts_screened":
            source["historical_git_checkouts_screened"],
        "complete_history_proven": False,
        "train_dev_ood_admitted": False,
    }


def validate_original_specs(specs: tuple[dict, ...] = SCENES) -> None:
    _require(type(specs) is tuple and specs == SCENES
             and len(specs) == 2
             and all(type(s) is dict and set(s) == _SCENE_FIELDS
                     for s in specs)
             and {s["language"] for s in specs} == {"en", "ar"}
             and {s["next_month_clicks"] for s in specs} == {0, 1},
             "two original date selection cases must be exact/bilingual")


@dataclass(frozen=True)
class DatepickerNativeReceipt:
    id: str
    language: str
    month_step: int
    native_calendar_opened: bool
    native_selected_value: str
    native_day_center: tuple[float, float]
    preaction_png_sha256: str
    preaction_rgba_sha256: str
    postaction_rgba_sha256: str


def _digest_valid(s: str) -> bool:
    return type(s) is str and len(s) == 64 and all(
        c in "0123456789abcdef" for c in s
    )


def validate_replay(
    rows: tuple[DatepickerNativeReceipt, ...],
    old_sha_metadata: dict | None = None,
) -> dict:
    validate_original_specs()
    _require(type(rows) is tuple and len(rows) == len(SCENES)*REPEATS,
             "independent original browser context missing")
    grouped: dict[str, list[DatepickerNativeReceipt]] = {}
    for row in rows:
        _require(type(row) is DatepickerNativeReceipt,
                 "unsupported source receipt type")
        spec = next((s for s in SCENES if row.id == s["id"]), None)
        _require(spec is not None
                 and row.language == spec["language"]
                 and row.month_step == spec["next_month_clicks"],
                 "original app/language/month source task drift")
        _require(row.native_calendar_opened is True
                 and row.native_selected_value == spec["expected_value"],
                 "original native date selection reward not observed")
        point = row.native_day_center
        _require(type(point) is tuple and len(point) == 2
                 and all(type(v) is float and 0 <= v < dim
                         for v, dim in zip(point, (WIDTH, HEIGHT))),
                 "source native day was not visible inside model image")
        _require(_digest_valid(row.preaction_png_sha256)
                 and _digest_valid(row.preaction_rgba_sha256)
                 and _digest_valid(row.postaction_rgba_sha256)
                 and row.preaction_rgba_sha256 != row.postaction_rgba_sha256,
                 "original date selection did not produce changed GUI raster")
        grouped.setdefault(row.id, []).append(row)
    _require(set(grouped) == {s["id"] for s in SCENES},
             "not all original datepicker cases seen")
    for repeat in grouped.values():
        _require(len(repeat) == REPEATS and repeat[0] == repeat[1],
                 "preaction screenshot/native geometry differs across fresh contexts")
    first = [grouped[s["id"]][0] for s in SCENES]
    _require(len({x.preaction_png_sha256 for x in first}) == 2
             and len({x.preaction_rgba_sha256 for x in first}) == 2,
             "two independent month screenshots are pixel duplicates")
    if old_sha_metadata is not None:
        screen_unadmitted_new_case_hashes([
            {"image_sha256": x.preaction_png_sha256,
             "visual_rgba_sha256": x.preaction_rgba_sha256}
            for x in first
        ], old_sha_metadata)
    return {
        "status": "H195_ORIGINAL_DATEPICKER_TWO_PREACTION_DATE_SELECTIONS_PASS",
        "independent_original_date_selection_tasks": 2,
        "fresh_browser_contexts_per_task": REPEATS,
        "native_original_open_month_and_day_actions": 10,
        "unique_preaction_png": 2,
        "unique_preaction_decoded_rgba": 2,
        "known_old_sha_screened": old_sha_metadata is not None,
        "source_family": "mymth/vanillajs-datepicker",
        "source_role": "RESEARCH_UNASSIGNED_NOT_ADMITTED",
        "train_dev_ood_admitted": 0,
        "persisted_images": 0,
        "persisted_private_gold": 0,
        "model_calls": 0,
        "gpu_steps": 0,
        "old_ood_accessed": False,
        "historical_source_complete": False,
    }


def _png_rgba(png: bytes) -> str:
    from PIL import Image

    with Image.open(BytesIO(png)) as im:
        _require(im.format == "PNG"
                 and im.size == (WIDTH, HEIGHT),
                 "actual 960x720 original source screenshot required")
        return sha256(im.convert("RGBA").tobytes()).hexdigest()


def _one_context(browser, port: int, spec: dict) -> DatepickerNativeReceipt:
    import re

    ctx = browser.new_context(
        viewport={"width": WIDTH, "height": HEIGHT},
        device_scale_factor=1,
        reduced_motion="reduce",
        timezone_id="Asia/Riyadh",
        locale="en-US",
    )
    try:
        # Pinned original demo imports highlight.js via a public CDN for
        # code syntax display only. Browser strictly blocks ALL external
        # resources. A no-op syntax highlighter allows the untouched
        # original app's initialization to proceed and does not touch
        # Datepicker, DOM values, events, or reward logic.
        ctx.add_init_script("window.hljs={highlightAll:function(){}}")
        ctx.route(
            "**/*",
            lambda route: route.continue_()
            if route.request.url.startswith(f"http://127.0.0.1:{port}/")
            else route.abort(),
        )
        page = ctx.new_page()
        page.clock.install(time=_FROZEN_TIME)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        resp = page.goto(
            f"http://127.0.0.1:{port}/demo/index.html",
            wait_until="networkidle", timeout=18000,
        )
        _require(resp is not None and resp.status == 200
                 and page.evaluate("typeof Datepicker") == "function",
                 "original demo/local datepicker runtime missing")
        field = page.locator("#sandbox input.date")
        _require(field.count() == 1 and field.input_value() == "",
                 "original datepicker input state changed")
        field.click()
        picker = page.locator(".datepicker-picker:visible")
        picker.wait_for(state="visible", timeout=10000)
        _require(picker.count() == 1, "original datepicker did not open")
        if spec["next_month_clicks"]:
            original_next = picker.locator(".datepicker-controls .next-btn")
            _require(original_next.count() >= 1,
                     "native next-month button unavailable")
            original_next.first.click()
        day = picker.locator(".datepicker-cell.day:not(.prev):not(.next)").filter(
            has_text=re.compile(rf"^{spec['day']}$")
        )
        _require(day.count() == 1 and day.is_visible(),
                 "native date cell absent from requested real month")
        box = day.bounding_box()
        _require(box is not None, "native calendar day target invisible")
        center = (
            float(round(box["x"] + box["width"]/2, 4)),
            float(round(box["y"] + box["height"]/2, 4)),
        )
        _require(0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT,
                 "date cell outside model-visible preaction screenshot")
        page.mouse.move(0, 0)
        page.wait_for_timeout(100)
        before = page.screenshot(animations="disabled")
        before_rgba = _png_rgba(before)
        # Only AFTER source screenshot: real browser native day click.
        day.click()
        page.wait_for_timeout(150)
        after = page.screenshot(animations="disabled")
        after_rgba = _png_rgba(after)
        _require(field.input_value() == spec["expected_value"]
                 and before_rgba != after_rgba and not errors,
                 "original browser date choice reward or visual change failed")
        return DatepickerNativeReceipt(
            id=spec["id"], language=spec["language"],
            month_step=spec["next_month_clicks"],
            native_calendar_opened=True,
            native_selected_value=field.input_value(),
            native_day_center=center,
            preaction_png_sha256=sha256(before).hexdigest(),
            preaction_rgba_sha256=before_rgba,
            postaction_rgba_sha256=after_rgba,
        )
    finally:
        ctx.close()


def run_original_datepicker(
    donor_root: Path, channel: str,
    historical_root: Path | None = None,
) -> dict:
    validate_original_specs()
    provenance = verify_original_source(donor_root, historical_root)
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(Quiet, directory=str(donor_root.resolve(strict=True))),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel=channel, headless=True
            )
            try:
                receipts = tuple(
                    _one_context(browser, server.server_port, spec)
                    for spec in SCENES for _ in range(REPEATS)
                )
            finally:
                browser.close()
        history = (
            scan_legacy_manifest_sha_fields([historical_root])
            if historical_root is not None else None
        )
        result = validate_replay(receipts, history)
        result["old_git_checkouts_screened"] = provenance[
            "historical_git_checkouts_screened"
        ]
        return result
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H195 original vanilla datepicker bilingual date selection, no data"
    )
    parser.add_argument("--donor-root", required=True, type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    result = run_original_datepicker(
        args.donor_root, args.browser_channel, args.historical_root,
    )
    print(
        f"{result['status']} "
        f"native_actions={result['native_original_open_month_and_day_actions']} "
        f"preaction_scenes={result['unique_preaction_decoded_rgba']} "
        f"old_git_checkouts={result['old_git_checkouts_screened']} "
        f"old_sha_screened={int(result['known_old_sha_screened'])} "
        "NO_DATASET_ADMISSION_NO_MODEL_NO_OOD_NO_MEDIA"
    )


if __name__ == "__main__":
    main()
