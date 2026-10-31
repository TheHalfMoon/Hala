"""H286: align H285 TRAIN candidates with actual Qwen relative output grammar.

Research TRAIN CANDIDATES ONLY, not official admission, DEV, blind OOD or
proof of adapted-model skill. Media, labels and gold stay in volatile memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from h199.native_action_sft import parse_actions, source_task_prompt
from h273.relative_native_bridge import (
    HEIGHT, WIDTH, normalized_to_pixel, normalize_strict_action,
)
from h282.private_clinicgrid_pretrained_probe import RELATIVE_SUFFIX
from h285.train_only_browser_candidates import (
    ROLE, SCENES, SOURCE_BLOB, exact_task, produce_train_only_batch,
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H286 private TRAIN candidate alignment refused: " + reason)


def pixel_to_normalized_exact(pixel: int, extent: int) -> int:
    """Minimum Qwen normalized integer whose H273 decode is this pixel."""
    require(type(pixel) is int and type(extent) is int
            and extent in (WIDTH, HEIGHT) and 0 <= pixel < extent,
            "strict original screenshot integer pixel required")
    relative = (pixel * 1000 + extent - 1) // extent
    require(0 <= relative <= 999
            and normalized_to_pixel(relative, extent) == pixel,
            "relative conversion cannot recover original pixel")
    return relative


def aligned_relative_sample(
    scene: tuple, original_prompt: str, original_completion: str,
) -> tuple[str, str]:
    """Only registered H285 supervised clicks; NEVER transform predictions."""
    require(type(scene) is tuple and scene in SCENES,
            "unregistered source TRAIN scene")
    task = exact_task(scene)
    require(type(original_prompt) is str
            and original_prompt == source_task_prompt(task),
            "original source prompt identity mismatch")
    action = parse_actions(original_completion, training_target=True)["actions"][0]
    require(action["kind"] == "click"
            and type(action["point"]) is list
            and len(action["point"]) == 2
            and all(type(p) is int for p in action["point"]),
            "strict original pixel click supervision required")
    x, y = action["point"]
    nx = pixel_to_normalized_exact(x, WIDTH)
    ny = pixel_to_normalized_exact(y, HEIGHT)
    relative = json.dumps({"actions": [{"kind": "click", "point": [nx, ny]}]},
                          separators=(",", ":"), ensure_ascii=False)
    require(normalize_strict_action(relative) == json.dumps(
        {"actions": [{"kind": "click", "point": [x, y]}]},
        separators=(",", ":"), ensure_ascii=False),
        "strict normalized-to-native roundtrip changed TRAIN target")
    prompt = task + RELATIVE_SUFFIX
    require(0 < len(prompt) <= 2000
            and "RELATIVE" in prompt and "Never output absolute pixel" in prompt,
            "training prompt not aligned with real H282 relative inference")
    return prompt, relative


def produce_relative_train_candidates(
    browser: object, source_path: Path,
    trusted_sink: Callable[[bytes, str, str], None], *, role: str,
) -> dict:
    """Check all 24 real browser cases before any private sink receives data.

    Trusted sink receives (PNG, Qwen-relative prompt, relative JSON).
    Sink is not atomic; owner must keep its outputs in ephemeral private
    TRAIN-only memory/storage, never evaluation or public output.
    """
    require(type(role) is str and role == ROLE,
            "no DEV/OOD/BLIND or official TRAIN admission")
    require(callable(trusted_sink), "explicit trusted private TRAIN sink required")
    rows: list[tuple[bytes, str, str]] = []

    def collect(png: bytes, original_prompt: str, completion: str) -> None:
        require(len(rows) < len(SCENES), "too many original candidate rows")
        require(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n")
                and len(png) < 4 * 1024 * 1024,
                "original TRAIN image bytes invalid")
        prompt, relative = aligned_relative_sample(
            SCENES[len(rows)], original_prompt, completion)
        rows.append((png, prompt, relative))

    original = produce_train_only_batch(
        browser, source_path, collect, role=ROLE)
    require(type(original) is dict
            and original.get("role") == ROLE
            and original.get("original_source_blob") == SOURCE_BLOB
            and original.get("actual_browser_source_actions_successful") == 24
            and original.get("distinct_png_screenshots") == 24
            and original.get("distinct_decoded_rgba_screenshots") == 24
            and original.get("english_candidates") == 12
            and original.get("arabic_candidates") == 12
            and original.get("official_train_manifest_admitted") is False
            and original.get("independent_dev_ood_evaluated") is False
            and len(rows) == len(SCENES) == 24,
            "incomplete/changed original browser TRAIN source proof")
    for png, prompt, completion in rows:
        trusted_sink(png, prompt, completion)
    return {
        "status": "H286_RELATIVE_ALIGNED_FIRSTPARTY_TRAIN_CANDIDATES_ONLY",
        "original_source_blob": SOURCE_BLOB,
        "role": ROLE,
        "actual_browser_supervisions": 24,
        "relative_targets_roundtrip_verified": 24,
        "english_candidates": 12,
        "arabic_candidates": 12,
        "model_visible_frame": "QWEN_RELATIVE_0_999",
        "native_executor_frame": "H199_PIXELS_960_720",
        "real_model_generations": 0,
        "optimizer_steps": 0,
        "official_train_admitted": False,
        "source_disjoint_blind_ood": False,
        "adapted_gui_gain_proven": False,
        "media_or_labels_saved_to_public_output": False,
    }
