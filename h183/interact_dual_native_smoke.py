"""H183: original interact.js two native accepted drag/drop scenes.

No TRAIN/DEV/OOD admission; source-only GUI research on public original
immutable interact.js donor. Images decoded and hashed in RAM only.
Copy-first H174 original browser isolation, network allowlist and Vite build.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
)

SCHEMA = "hala.h183.original-interact-drag-replay.source-only.v1"
SCENES = (
    {
        "id": "interact-drag1-to-drop1",
        "language": "en",
        "drag": "drag1",
        "drop": "drop1",
        "instruction": "Drag the first blue element into the first drop zone.",
    },
    {
        "id": "interact-drag2-to-drop2",
        "language": "ar",
        "drag": "drag2",
        "drop": "drop2",
        "instruction": "اسحب العنصر الأزرق الثاني إلى منطقة الإفلات الثانية.",
    },
)
REPEATS = 2
FIELDS = frozenset({"id", "language", "drag", "drop", "instruction"})
WIDTH, HEIGHT = 960, 720


@dataclass(frozen=True)
class NativeDropScene:
    id: str
    language: str
    drag: str
    drop: str
    before: str
    after: str
    unrelated_drag_unchanged: bool
    rgba_sha256: str
    png_sha256: str


def validate_scenes(specs: tuple[dict, ...] = SCENES) -> None:
    _require(type(specs) is tuple and len(specs) == 2,
             "H183 requires two source-only preregistered original scenes")
    names = set()
    endpoints = set()
    languages = set()
    for idx, scene in enumerate(specs):
        _require(type(scene) is dict and set(scene) == FIELDS,
                 "H183 exact metadata fields only")
        expected = SCENES[idx]
        _require(scene == expected,
                 "H183 pinned original drag/drop roles or instruction changed")
        _require(scene["language"] in ("en", "ar")
                 and type(scene["instruction"]) is str
                 and len(scene["instruction"].strip()) >= 20,
                 "H183 valid bilingual source instructions required")
        _require(scene["id"] not in names
                 and (scene["drag"], scene["drop"]) not in endpoints
                 and scene["language"] not in languages,
                 "H183 duplicate source-native scene identity")
        names.add(scene["id"])
        endpoints.add((scene["drag"], scene["drop"]))
        languages.add(scene["language"])
    _require(languages == {"ar", "en"},
             "H183 both source instructions are necessary")


def _sha_ok(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def verify_native_replay(rows: tuple[NativeDropScene, ...]) -> dict:
    """Pure receipt schema and independence guard; not browser proof alone."""
    validate_scenes()
    _require(type(rows) is tuple and len(rows) == len(SCENES)*REPEATS,
             "H183 missing original scene/repeat")
    runs: dict[str, list[NativeDropScene]] = {}
    for row in rows:
        _require(type(row) is NativeDropScene,
                 "H183 strict native drop receipt only")
        spec = next((s for s in SCENES if s["id"] == row.id), None)
        _require(spec is not None, "H183 unknown source-native drop")
        _require(
            row.language == spec["language"]
            and row.drag == spec["drag"] and row.drop == spec["drop"],
            "H183 native task identity mismatch",
        )
        _require(
            type(row.before) is str and row.before.startswith("Drag me")
            and row.after == "Dropped"
            and row.unrelated_drag_unchanged is True,
            "H183 original native drop action or non-target was not verified",
        )
        _require(
            _sha_ok(row.rgba_sha256) and _sha_ok(row.png_sha256),
            "H183 original screenshot hash malformed",
        )
        runs.setdefault(row.id, []).append(row)
    _require(set(runs) == {s["id"] for s in SCENES},
             "H183 original source scene missing")
    for repeats in runs.values():
        _require(
            len(repeats) == REPEATS and repeats[0] == repeats[1],
            "H183 independent native source PNG/RGBA replay mismatch",
        )
    _require(
        len({pair[0].rgba_sha256 for pair in runs.values()}) == len(SCENES)
        and len({pair[0].png_sha256 for pair in runs.values()}) == len(SCENES),
        "H183 original drag tasks have identical visual outcomes",
    )
    return {
        "status": "H183_INTERACT_TWO_ORIGINAL_NATIVE_DRAGS_SOURCE_ONLY",
        "native_source_drag_successes": len(rows),
        "distinct_original_rgba_scenes": len(SCENES),
        "different_original_source_drags": len(SCENES),
        "languages": len({row.language for row in rows}),
        "independent_contexts_per_scene": REPEATS,
        "train_dev_ood_data_admitted": False,
        "old_ood_accessed": False,
        "new_ood_screenshots": 0,
        "model_calls": 0,
        "model_rewards": 0,
    }


def _one_scene(browser, port: int, spec: dict) -> NativeDropScene:
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
            f"http://127.0.0.1:{port}/dropzones/index.html",
            wait_until="commit", timeout=12000,
        )
        _require(response is not None and response.status == 200,
                 "H183 original demo not served")
        drag = page.locator("#" + spec["drag"])
        drop = page.locator("#" + spec["drop"])
        drag.wait_for(state="visible", timeout=15000)
        drop.wait_for(state="visible", timeout=15000)
        _require(drag.count() == 1 and drop.count() == 1,
                 "H183 original draggable/dropzone missing")
        before = drag.inner_text().strip()
        untouched_id = "drag2" if spec["drag"] == "drag1" else "drag1"
        untouched = page.locator("#" + untouched_id)
        untouched_before = untouched.inner_text().strip()
        _require(before.startswith("Drag me") and
                 untouched_before.startswith("Drag me"),
                 "H183 original unmodified initial drag elements required")
        source = drag.bounding_box()
        target = drop.bounding_box()
        _require(source is not None and target is not None,
                 "H183 original pointer target geometry missing")
        sx = source["x"] + source["width"]/2
        sy = source["y"] + source["height"]/2
        tx = target["x"] + target["width"]/2
        ty = target["y"] + target["height"]/2
        page.mouse.move(sx, sy)
        page.mouse.down()
        page.mouse.move(tx, ty, steps=30)
        page.mouse.up()
        page.wait_for_function(
            "(id) => document.getElementById(id)?.textContent?.trim() === 'Dropped'",
            arg=spec["drag"], timeout=8000,
        )
        after = drag.inner_text().strip()
        unrelated_unchanged = (
            untouched.inner_text().strip() == untouched_before
        )
        _require(after == "Dropped" and unrelated_unchanged,
                 "H183 original source native reward unsuccessful")
        page.mouse.move(0, 0)
        page.wait_for_timeout(400)
        png = page.screenshot(animations="disabled")
        image = Image.open(BytesIO(png)).convert("RGBA")
        _require(image.size == (WIDTH, HEIGHT),
                 "H183 native source viewport size changed")
        _require(not errors, "H183 original source JS page errors")
        return NativeDropScene(
            id=spec["id"], language=spec["language"],
            drag=spec["drag"], drop=spec["drop"], before=before,
            after=after, unrelated_drag_unchanged=unrelated_unchanged,
            rgba_sha256=sha256(image.tobytes()).hexdigest(),
            png_sha256=sha256(png).hexdigest(),
        )
    finally:
        context.close()


def run_native_multiscene(port: int, channel: str) -> dict:
    """Four original browser actions, two scenes, no image writes/uploads."""
    validate_scenes()
    _require(type(port) is int and 1 <= port <= 65535,
             "H183 invalid loopback port")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        try:
            return verify_native_replay(tuple(
                _one_scene(browser, port, spec)
                for spec in SCENES for _ in range(REPEATS)
            ))
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H183 interact original drag x2 fresh context; research only"
    )
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    proof = run_native_multiscene(args.port, args.browser_channel)
    print(
        f"{proof['status']} native_actions={proof['native_source_drag_successes']} "
        f"distinct_rgba={proof['distinct_original_rgba_scenes']} "
        f"languages={proof['languages']} "
        "NO_TRAIN_DEV_OOD_IMAGES_NO_MODEL_INFERENCE"
    )


if __name__ == "__main__":
    main()
