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

from h173.browser_runtime import original_browser_options
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
            # A second context in the same process hid real Windows drift.
            # Separate Chromium OS processes as in owner capture / verify.
            for phase in ("capture", "replay"):
                browser = playwright.chromium.launch(
                    **original_browser_options()
                )
                try:
                    for index, item in enumerate(specs):
                        raw, task, reference, state = original_native_scene(
                            browser, url, item,
                            execute_reference=(phase == "capture"),
                        )
                        if phase == "capture":
                            if reference is None:
                                raise ValueError(
                                    "H173 public native reference point absent"
                                )
                            first[item["index"]] = (raw, task, state)
                            print(
                                "H173_PUBLIC_SOURCE_CAPTURE_ONLY",
                                index + 1, OOD_COUNT, item["app"], flush=True,
                            )
                        else:
                            saved, original_task, original_state = first[item["index"]]
                            if (
                                reference is not None
                                or task != original_task
                                or state["kind"] != original_state["kind"]
                                or state["value"] != original_state["value"]
                                or not same_exact_png_pixels(saved, raw)
                            ):
                                raise ValueError(
                                    "H173 independent Chromium process source not "
                                    "reproducible: " + str(item["index"])
                                )
                            replay[item["index"]] = visual_hash(raw)
                            print(
                                "H173_PUBLIC_SOURCE_INDEPENDENT_PIXEL_REPLAY",
                                index + 1, OOD_COUNT, item["app"], flush=True,
                            )
                finally:
                    browser.close()
    if len(replay) != OOD_COUNT or len(set(replay.values())) != OOD_COUNT:
        raise ValueError("H173 expected 32 VISUALLY distinct original app cases")
    return {
        "status": "H173_PUBLIC_PRISTINE_BROWSER_SOURCE_32_REPRO_ONLY",
        "original_native_reference_successes": OOD_COUNT,
        "exact_original_rgba_replays": OOD_COUNT,
        "independent_chromium_processes": 2,
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
