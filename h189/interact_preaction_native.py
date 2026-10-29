"""H189: native interact.js PRE-action DEV prospect (source-only).

Reuse H174 Playwright isolation + H183 genuine original accepted drag rules.
An ORIGINAL draggable #drag3 native seed-drop creates two visually
different pre-action UI states; only then screenshot the model-visible UI,
then execute the real target drag/drop with original source reward.
Never label POST-action image as an input. No data admission or GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Callable

from h174.pickr_native_smoke import (
    _require, allow_loopback_request, create_isolated_smoke_context,
)
from h183.interact_dual_native_smoke import (
    SCENES as PREVIOUSLY_VERIFIED_ORIGINAL_DROPS, REPEATS, WIDTH, HEIGHT,
)
from h185.manifest_sha_only import screen_unadmitted_new_case_hashes

# Original pinned interact.js index.js accepts #drag3 into every drop zone.
# Seed drop3 for target #drag1->#drop1, seed drop1 for #drag2->#drop2.
# Target native actions remain H183's already source-verified original pair.
SEED_DROPS = ("drop3", "drop1")
SPECS = tuple(
    {**row, "seed_drag": "drag3", "seed_drop": seed}
    for row, seed in zip(PREVIOUSLY_VERIFIED_ORIGINAL_DROPS, SEED_DROPS)
)
PUBLIC_FIELDS = frozenset({
    "id", "language", "drag", "drop", "instruction", "seed_drag", "seed_drop",
})


@dataclass(frozen=True)
class PreactionDropProof:
    id: str
    language: str
    drag: str
    drop: str
    seed_drop: str
    source_seed_native_reward: str
    source_target_before: str
    source_target_after: str
    unrelated_drag_untouched: bool
    pre_png_sha256: str
    pre_rgba_sha256: str
    native_target_start: tuple[float, float]
    native_target_end: tuple[float, float]


def validate_specification(specs: tuple[dict, ...] = SPECS) -> None:
    _require(type(specs) is tuple and len(specs) == 2,
             "H189 exactly two original DEV research candidates")
    for spec, expected in zip(specs, SPECS):
        _require(type(spec) is dict and set(spec) == PUBLIC_FIELDS
                 and spec == expected,
                 "H189 original bilingual source native seed/target changed")
        _require(spec["seed_drag"] == "drag3"
                 and spec["seed_drop"] in {"drop1", "drop3"}
                 and spec["seed_drop"] != spec["drop"],
                 "H189 native seed must not obscure target zone")


def _sha_ok(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def verify_preaction_replay(
    results: tuple[PreactionDropProof, ...],
    owner_old_sha_census: dict | None = None,
) -> dict:
    validate_specification()
    _require(type(results) is tuple and len(results) == len(SPECS) * REPEATS,
             "H189 source-native preaction scene count wrong")
    seen: dict[str, list[PreactionDropProof]] = {}
    for row in results:
        _require(type(row) is PreactionDropProof,
                 "H189 strict original source result only")
        spec = next((s for s in SPECS if s["id"] == row.id), None)
        _require(spec is not None
                 and row.language == spec["language"]
                 and row.drag == spec["drag"]
                 and row.drop == spec["drop"]
                 and row.seed_drop == spec["seed_drop"],
                 "H189 source/seed/target/lang identity drift")
        _require(row.source_seed_native_reward == "Dropped"
                 and row.source_target_before.startswith("Drag me")
                 and row.source_target_after == "Dropped"
                 and row.unrelated_drag_untouched is True,
                 "H189 source native seed or target success missing")
        _require(_sha_ok(row.pre_png_sha256)
                 and _sha_ok(row.pre_rgba_sha256),
                 "H189 invalid original preaction visual SHA")
        for point in (row.native_target_start, row.native_target_end):
            _require(
                type(point) is tuple and len(point) == 2
                and all(type(v) is float and 0 <= v < bound
                        for v, bound in zip(point, (WIDTH, HEIGHT))),
                "H189 source-native drag start/drop geometry invalid",
            )
        seen.setdefault(row.id, []).append(row)
    _require(set(seen) == {s["id"] for s in SPECS},
             "H189 one original preaction scene missing")
    for repetitions in seen.values():
        _require(len(repetitions) == REPEATS
                 and repetitions[0] == repetitions[1],
                 "H189 independent original preaction PNG/RGBA/geometry drift")
    firsts = [seen[s["id"]][0] for s in SPECS]
    _require(len({x.pre_png_sha256 for x in firsts}) == len(SPECS)
             and len({x.pre_rgba_sha256 for x in firsts}) == len(SPECS),
             "H189 native source seed states do not yield different preaction pixels")
    if owner_old_sha_census is not None:
        screen_unadmitted_new_case_hashes([
            {
                "image_sha256": row.pre_png_sha256,
                "visual_rgba_sha256": row.pre_rgba_sha256,
            } for row in firsts
        ], owner_old_sha_census)
    return {
        "status": "H189_REAL_ORIGINAL_INTERACT_PREACTION_DEV_SOURCE_ONLY",
        "original_preaction_scenes": len(SPECS),
        "native_seed_and_target_successes": len(results) * 2,
        "independent_contexts_per_scene": REPEATS,
        "distinct_preaction_decoded_rgba": len(SPECS),
        "owner_old_sha_candidates_screened": owner_old_sha_census is not None,
        "dev_dataset_images_admitted": 0,
        "train_dataset_images_admitted": 0,
        "ood_images_created": 0,
        "model_calls": 0,
        "full_historical_visual_provenance": False,
    }


def _bounding_center(locator) -> tuple[float, float]:
    box = locator.bounding_box()
    _require(box is not None, "H189 native original source target not visible")
    cx = float(round(box["x"] + box["width"] / 2, 4))
    cy = float(round(box["y"] + box["height"] / 2, 4))
    _require(0 <= cx < WIDTH and 0 <= cy < HEIGHT,
             "H189 original native target center outside screenshot")
    return (cx, cy)


def _real_native_drop(page, drag_id: str, drop_id: str) -> None:
    drag = page.locator("#" + drag_id)
    dest = page.locator("#" + drop_id)
    before = drag.inner_text().strip()
    _require(before.startswith("Drag me"),
             "H189 native original draggable was already used")
    sx, sy = _bounding_center(drag)
    dx, dy = _bounding_center(dest)
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(dx, dy, steps=30)
    page.mouse.up()
    page.wait_for_function(
        "(id) => document.getElementById(id)?.textContent?.trim() === 'Dropped'",
        arg=drag_id, timeout=8000,
    )
    _require(drag.inner_text().strip() == "Dropped",
             "H189 original interact.js native ondrop not delivered")


def _one_scene(
    browser, port: int, spec: dict,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> PreactionDropProof:
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
                 "H189 pristine original interact.js demo not served")
        for ident in ("drag1", "drag2", "drag3", "drag4", "drop1", "drop2", "drop3"):
            page.locator("#" + ident).wait_for(state="visible", timeout=15000)
        target = page.locator("#" + spec["drag"])
        unrelated = page.locator("#drag4")
        unrelated_before = unrelated.inner_text().strip()
        target_before = target.inner_text().strip()
        _require(target_before.startswith("Drag me")
                 and unrelated_before.startswith("Drag me"),
                 "H189 pristine original target initial state drift")
        _real_native_drop(page, spec["seed_drag"], spec["seed_drop"])
        seed_state = page.locator("#drag3").inner_text().strip()
        _require(seed_state == "Dropped"
                 and target.inner_text().strip() == target_before,
                 "H189 original native seed altered the target unexpectedly")
        # Input screenshot BEFORE target drag: only original app native UI,
        # with a different original seed-drop state per scenario.
        page.mouse.move(0, 0)
        page.wait_for_timeout(400)
        start = _bounding_center(target)
        end = _bounding_center(page.locator("#" + spec["drop"]))
        png = page.screenshot(animations="disabled")
        with Image.open(BytesIO(png)) as raw:
            _require(raw.format == "PNG" and raw.size == (WIDTH, HEIGHT),
                     "H189 original preaction browser image format/size changed")
            rgba_sha = sha256(raw.convert("RGBA").tobytes()).hexdigest()
        # Actual unchanged original source pointer action AFTER input image.
        _real_native_drop(page, spec["drag"], spec["drop"])
        after = target.inner_text().strip()
        unrelated_ok = unrelated.inner_text().strip() == unrelated_before
        _require(after == "Dropped" and unrelated_ok and not errors,
                 "H189 original reference action failed or original JS error")
        if capture_preaction is not None:
            capture_preaction(
                dict(spec), bytes(png),
                {"start": list(start), "drop": list(end)},
            )
        return PreactionDropProof(
            id=spec["id"], language=spec["language"],
            drag=spec["drag"], drop=spec["drop"],
            seed_drop=spec["seed_drop"],
            source_seed_native_reward=seed_state,
            source_target_before=target_before,
            source_target_after=after,
            unrelated_drag_untouched=unrelated_ok,
            pre_png_sha256=sha256(png).hexdigest(),
            pre_rgba_sha256=rgba_sha,
            native_target_start=start, native_target_end=end,
        )
    finally:
        context.close()


def run_native_preaction(
    port: int, channel: str,
    owner_old_sha_census: dict | None = None,
    capture_preaction: Callable[[dict, bytes, dict], None] | None = None,
) -> dict:
    validate_specification()
    _require(type(port) is int and 1 <= port <= 65535,
             "H189 only loopback original source static server")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        try:
            return verify_preaction_replay(tuple(
                _one_scene(browser, port, spec, capture_preaction)
                for spec in SPECS for _ in range(REPEATS)
            ), owner_old_sha_census)
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H189 original interact.js native PREACTION source-only QA"
    )
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--browser-channel", default="chromium")
    args = parser.parse_args()
    proof = run_native_preaction(args.port, args.browser_channel)
    print(
        f"{proof['status']} native_seed_and_target="
        f"{proof['native_seed_and_target_successes']} "
        f"preaction_scenes={proof['original_preaction_scenes']} "
        f"distinct_pre_rgba={proof['distinct_preaction_decoded_rgba']} "
        "NO_DEV_DATASET_NO_GPU_NO_OLD_OOD_NO_MEDIA_WRITES"
    )


if __name__ == "__main__":
    main()
