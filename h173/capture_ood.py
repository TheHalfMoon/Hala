"""H173 owner-only original-app OOD screenshot capture and independent replay.

Only source-native Tingle modal close and Tiny Slider carousel navigation.
No model, inference, training, API credentials, or gold uploaded.
This is *candidate* OOD; any visual instability quarantines the whole set.
"""
import argparse
import json
from pathlib import Path

from h173.source_preflight import (
    SOURCE_PINS,
    OOD_COUNT,
    preregister_provisional_cases,
    sha,
    verified_original_source,
)
from h173.stable_capture import same_original_pixels, stabilized_original_png

MANIFEST = "hala.h173.genuine-novel-two-original-app-ood32.v1"
OWNER = "hala.h173.owner-only-original-reference-clicks.v1"
ZOOMS = (0.85, 1.00, 1.15, 1.30)
# Last two carousel variants use different starting slides, not the same image.
SLIDERS = (
    "base", "start-index", "non-loop", "rewind",
    "center", "fixedWidth", "base", "fixedWidth",
)


def previous_image_hashes(owner, h168, h169, h171, h172):
    """Read only SHA metadata from past datasets, never consumed pixels."""
    from h171.source import prior_hashes

    previous = prior_hashes(owner, h168, h169)
    for directory, schema in (
        (h171, "hala.h171.fresh-two-original-click-app-train-only.v1"),
        (h172, "hala.h172.new-four-app-candidate-dev32-owner-only.v1"),
    ):
        record = json.loads(
            (Path(directory) / "manifest.json").read_text(encoding="utf8")
        )
        if record.get("schema") != schema or record.get("count") != OOD_COUNT:
            raise ValueError("H173 prior TRAIN/DEV source registry missing")
        scenes = record.get("private_cases", [])
        if len(scenes) != OOD_COUNT:
            raise ValueError("H173 previous source SHA list truncated")
        for scene in scenes:
            digest = scene.get("image_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("H173 previous source SHA invalid")
            previous.add(digest)
    consumed = (
        Path(owner)
        / "private-h172"
        / "H172_FRESH_NATIVE_DEV_ONE_SHOT_CONSUMED.lock"
    )
    if not consumed.is_file():
        raise ValueError("H173 must not proceed until old H172 DEV consumed")
    return previous


def original_native_scene(browser, url, spec, *, execute_reference):
    """Produce genuinely original-browser pixels, then optional owner action.

    The reference click is NEVER passed into model input. In replay mode,
    not even the owner reference action is executed.
    """
    from h138.probe import center
    from h159.stable_source_qa import stable_png

    context = browser.new_context(
        viewport={"width": 960, "height": 720},
        device_scale_factor=1,
        locale="en-US",
        service_workers="block",
    )
    try:
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(url)
                else route.abort()
            ),
        )
        app, variant = spec["app"], spec["variant"]
        source_html = SOURCE_PINS[app]["original_html"]
        response = page.goto(
            url + app + "/" + source_html,
            wait_until="load",
            timeout=30000,
        )
        if response is None or response.status != 200:
            raise ValueError("H173 pristine upstream HTML source unavailable")
        if app == "tingle":
            zoom = ZOOMS[variant // 4]
            modal = (variant % 4) + 1
            page.evaluate(
                "(value) => { document.documentElement.style.zoom=String(value); }",
                zoom,
            )
            trigger = page.locator(".js-tingle-modal-" + str(modal))
            trigger.scroll_into_view_if_needed()
            trigger.click()
            page.locator(".tingle-modal--visible").wait_for(timeout=10000)
            target = page.locator(
                ".tingle-modal--visible .tingle-modal__close"
            )
            target.wait_for(timeout=8000)
            if page.locator(".tingle-modal--visible").count() != 1:
                raise ValueError("H173 modal not open before model-visible image")
            instruction = (
                "Close the open Tingle dialog using its close button."
                if spec["language"] == "en"
                else "أغلق نافذة Tingle المفتوحة باستخدام زر الإغلاق."
            )
            prestate = {"kind": "modal_visible", "value": True}
        else:
            name = SLIDERS[variant // 2]
            direction = "prev" if variant % 2 == 0 else "next"
            seed = (
                (4 if direction == "prev" else 5)
                if variant // 2 >= 6
                else (2 if direction == "prev" else 3)
            )
            page.wait_for_function(
                "() => typeof sliders !== 'undefined' && !!sliders.base",
                timeout=20000,
            )
            page.evaluate(
                "([name, value]) => sliders[name].goTo(value)",
                [name, seed],
            )
            page.wait_for_timeout(550)
            target = page.locator(
                "#" + name + "_wrapper .tns-controls button"
                '[data-controls="' + direction + '"]'
            )
            target.wait_for(timeout=8000)
            target.scroll_into_view_if_needed()
            original_index = page.evaluate(
                "(name) => sliders[name].getInfo().index",
                name,
            )
            direction_en = (
                "previous" if direction == "prev" else "next"
            )
            instruction = (
                "Move the " + name + " carousel to the "
                + direction_en + " slide."
                if spec["language"] == "en"
                else "انتقل إلى الشريحة "
                + ("السابقة" if direction == "prev" else "التالية")
                + " في شريط " + name + "."
            )
            prestate = {
                "kind": "carousel_index",
                "name": name,
                "direction": direction,
                "value": original_index,
            }
        # Hover state may persist from *previous* page mouse activity.
        # Reset and demand two full-size EXACT decoded RGBA equal frames.
        # Do NOT disable original CSS animations or alter source files.
        image = stabilized_original_png(
            page,
            stable_png,
            modal=app == "tingle",
        )
        point = center(target)
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                type(value) is not int or not 0 <= value < bound
                for value, bound in zip(point, (960, 720))
            )
        ):
            raise ValueError("H173 original reference click offscreen")
        if execute_reference:
            page.mouse.click(*point)
            page.wait_for_timeout(550 if app == "tiny-slider" else 140)
            if app == "tingle":
                success = page.locator(".tingle-modal--visible").count() == 0
            else:
                actual_index = page.evaluate(
                    "(name) => sliders[name].getInfo().index",
                    name,
                )
                success = actual_index != prestate["value"]
            if not success:
                raise ValueError("H173 original native reference action failed")
        return image, instruction, point, prestate
    finally:
        context.close()


def run(owner, upstream, h168, h169, h171, h172, destination, mode):
    """Create only NEW owner-private paths. Quarantines on any failure."""
    from playwright.sync_api import sync_playwright
    from h130.browser_env import _chromium_binary
    from h138.probe import serve

    dest = Path(destination)
    proof = verified_original_source(upstream)
    prior = previous_image_hashes(owner, h168, h169, h171, h172)
    if mode == "capture" and dest.exists():
        raise FileExistsError("H173 cannot overwrite an old quarantined OOD capture")
    frozen = None
    if mode == "verify":
        frozen = json.loads(
            (dest / "manifest.json").read_text(encoding="utf8")
        )
        if frozen.get("schema") != MANIFEST or frozen.get("count") != OOD_COUNT:
            raise ValueError("H173 original OOD capture is not frozen")
    specs = preregister_provisional_cases()
    if mode == "smoke":
        specs = [specs[ix] for ix in (0, 1, 4, 8, 16, 17, 18, 19)]
    scenes, actions, pixels = [], [], {}
    seen = set(prior)
    with serve(Path(upstream)) as url:
        with sync_playwright() as playwright:
            options = {"headless": True}
            executable = _chromium_binary()
            if executable:
                options["executable_path"] = executable
            browser = playwright.chromium.launch(**options)
            try:
                for spec in specs:
                    raw, task, point, state = original_native_scene(
                        browser, url, spec, execute_reference=mode != "verify"
                    )
                    digest = sha(raw)
                    if digest in seen:
                        raise ValueError(
                            "H173 duplicate OOD pixel image or consumed history"
                        )
                    seen.add(digest)
                    entry = {
                        **spec,
                        "split": "OOD",
                        "image": "images/" + str(spec["index"]) + ".png",
                        "image_sha256": digest,
                        "instruction": task,
                        "original_reference_success": True,
                    }
                    if mode == "verify":
                        original = next(
                            (
                                row
                                for row in frozen["private_cases"]
                                if row["index"] == spec["index"]
                            ),
                            None,
                        )
                        source_bytes = (dest / entry["image"]).read_bytes()
                        if (
                            original is None
                            or original.get("instruction") != task
                            or sha(source_bytes) != original.get("image_sha256")
                            or not same_original_pixels(source_bytes, raw)
                        ):
                            raise ValueError(
                                "H173 original OOD independent exact RGBA replay failed"
                            )
                    else:
                        scenes.append(entry)
                        actions.append({
                            "index": spec["index"],
                            "kind": "click",
                            "point": point,
                            "prestate": state,
                            "original_native_reference_success": True,
                        })
                        pixels[spec["index"]] = raw
                    print(
                        "H173_" + mode.upper(),
                        spec["app"],
                        spec["index"],
                        digest[:14],
                        flush=True,
                    )
            finally:
                browser.close()
    if mode == "smoke":
        return {
            "status": "H173_ORIGINAL_UNSEEN_SOURCE_STABILIZED_SMOKE_ONLY",
            "model_calls": 0,
            "cases": len(specs),
        }
    if mode == "verify":
        if len(specs) != OOD_COUNT:
            raise ValueError("H173 replay must cover all 32 original OOD cases")
        receipt_file = dest / "owner-source-replay-receipt.json"
        if receipt_file.exists():
            raise FileExistsError("H173 source already independently replayed")
        private_manifest = (dest / "manifest.json").read_bytes()
        receipt = {
            "schema": "hala.h173.original-ood-32-pixel-replay-receipt.v1",
            "status": "H173_PRIVATE_NEW_OOD_32_EXACT_PIXEL_REPLAY_PASS",
            "source_manifest_sha256": sha(private_manifest),
            "cases": OOD_COUNT,
            "independent_rgba_pixel_replays": OOD_COUNT,
            "prior_train_dev_ood_excluded": True,
            "owner_reference_actions_not_in_model_input": True,
            "model_calls": 0,
            "optimizer_steps": 0,
        }
        receipt_file.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
        return receipt
    if len(scenes) != OOD_COUNT or len(seen) - len(prior) != OOD_COUNT:
        raise ValueError("H173 32 unique actual OOD original app scenes required")
    manifest = {
        "schema": MANIFEST,
        "split": "OOD",
        "count": OOD_COUNT,
        "source_proof": proof,
        "no_old_train_dev_ood_images": True,
        "no_model_calls_before_frozen_capture": True,
        "owner_gold_uploaded": False,
        "private_cases": scenes,
    }
    owner_gold = {
        "schema": "hala.h173.owner-only-original-reference-clicks.v1",
        "owner_reference_actions": actions,
    }
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "images").mkdir()
    for index, raw in pixels.items():
        (dest / "images" / (str(index) + ".png")).write_bytes(raw)
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    (dest / "owner-reference-private.json").write_text(
        json.dumps(owner_gold, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf8",
    )
    return {
        "status": "H173_ORIGINAL_UNSEEN_32_OOD_SOURCE_CAPTURED_NOT_SCORED",
        "cases": 32,
        "model_calls": 0,
        "optimizer_steps": 0,
        "owner_gold_uploaded": False,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("owner", "upstream", "h168", "h169", "h171", "h172", "dest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("smoke", "capture", "verify"), required=True)
    args = parser.parse_args()
    result = run(
        args.owner,
        args.upstream,
        args.h168,
        args.h169,
        args.h171,
        args.h172,
        args.dest,
        args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
