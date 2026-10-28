"""Reproduce 32 pristine ORIGINAL external GUI app scenes in isolated CI.

Strictly a source reproducibility check, NOT an OOD model evaluation.
No owner-private datasets, model weights, owner gold, Kaggle calls, or
prior consumed screenshots are available. Do not report OOD scores.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from h130.browser_env import _chromium_binary
from h138.probe import serve
from h171.image_integrity import same_exact_png_pixels
from h173.capture_ood import original_native_scene
from h173.source_preflight import (
    OOD_COUNT, preregister_provisional_cases, verified_original_source,
)


def visual_hash(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.size != (960, 720):
            raise ValueError("H173 real original screenshot not 960x720 PNG")
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def run(upstream):
    from playwright.sync_api import sync_playwright

    upstream = Path(upstream)
    verified_original_source(upstream)
    specs = preregister_provisional_cases()
    first, replay = {}, {}
    with serve(upstream) as url:
        with sync_playwright() as playwright:
            opts = {"headless": True}
            installed = _chromium_binary()
            if installed:
                opts["executable_path"] = installed
            browser = playwright.chromium.launch(**opts)
            try:
                for index, item in enumerate(specs):
                    # Real original-app reference click must succeed before
                    # any screenshot becomes a candidate for later OOD.
                    raw, task, _, state = original_native_scene(
                        browser, url, item, execute_reference=True
                    )
                    first[item["index"]] = (raw, task, state)
                    print("H173_PUBLIC_SOURCE_CAPTURE_ONLY", index + 1, OOD_COUNT,
                          item["app"], flush=True)
                for index, item in enumerate(specs):
                    raw, task, _, state = original_native_scene(
                        browser, url, item, execute_reference=False
                    )
                    saved, original_task, original_state = first[item["index"]]
                    if (
                        task != original_task
                        or state["kind"] != original_state["kind"]
                        or state["value"] != original_state["value"]
                        or not same_exact_png_pixels(saved, raw)
                    ):
                        raise ValueError(
                            "H173 PRISTINE CI source not reproducible in original app: "
                            + str(item["index"])
                        )
                    replay[item["index"]] = visual_hash(raw)
                    print("H173_PUBLIC_SOURCE_INDEPENDENT_PIXEL_REPLAY",
                          index + 1, OOD_COUNT, item["app"], flush=True)
            finally:
                browser.close()
    if len(replay) != OOD_COUNT or len(set(replay.values())) != OOD_COUNT:
        raise ValueError("H173 expected 32 VISUALLY distinct original app cases")
    return {
        "status": "H173_PUBLIC_PRISTINE_BROWSER_SOURCE_32_REPRO_ONLY",
        "original_native_reference_successes": OOD_COUNT,
        "exact_original_rgba_replays": OOD_COUNT,
        "visually_distinct_scenes": OOD_COUNT,
        "source_app_families": 2,
        "arabic_scenes": 16,
        "english_scenes": 16,
        "owner_private_dataset_seen": False,
        "previous_consumed_scenes_seen": False,
        "owner_gold_uploaded": False,
        "model_gpu_calls": 0,
        "native_model_rewards": 0,
        "ood_generalization_qualified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream), indent=2, sort_keys=True))
