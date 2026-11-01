"""H328: compare independently authored original GUI TRAIN-candidate and
PUBLIC EXPOSED evaluation-family screenshot fingerprints in memory.

A safety-focused source-only disjointness check. It cannot prove authorship,
third-party rights, legal title, historical exclusion, independent DEV/BLIND
OOD or permission to train; H199 official admission remains EMPTY.

No samples, PNGs, points, source prompts, labels, model text, or hashes are
written, uploaded, returned or logged. No Qwen model/GPU is used.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
import json

from h199.native_action_sft import WIDTH, HEIGHT
from h263.native_browser_executor import execute_one
from h319.beaconboard_dynamic_research import (
    IDS as EVAL_IDS,
    SCENES as EXPOSED_SCENES,
    checked_source as checked_exposed_source,
    inspect_all_stations,
    original_scene,
)
from h325.two_step_train_candidates import (
    ROLE as TRAIN_CANDIDATE_ROLE,
    SOURCE_BLOB as RAILKIT_BLOB,
    SCENES as TRAIN_SCENES,
    checked_source as checked_train_source,
    produce_two_step_train_candidates,
)
from h322.source_capture import source_capture_png

ROLE = "H328_ORIGINAL_TRAIN_VS_EXPOSED_SOURCE_VISUAL_SPLIT_AUDIT_ONLY"
EXPECTED_TRAIN_FRAMES = 32
EXPECTED_EXPOSED_FRAMES = 24
BEACONBOARD_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"


def need(value: bool) -> None:
    if not value:
        raise ValueError("H328 ORIGINAL_VISUAL_SPLIT_REFUSED")


def decoded_rgb_digest(png: object) -> bytes:
    """Canonical RGB comparison: not PNG compression/metadata bytes."""
    from PIL import Image
    need(type(png) is bytes and 100 <= len(png) <= 4 * 1024 * 1024
         and png.startswith(b"\x89PNG\r\n\x1a\n"))
    try:
        with Image.open(BytesIO(png)) as image:
            need(image.format == "PNG" and image.size == (WIDTH, HEIGHT))
            image.load()
            return sha256(image.convert("RGB").tobytes()).digest()
    except (OSError, ValueError, TypeError):
        raise ValueError("H328 ORIGINAL_VISUAL_SPLIT_REFUSED") from None


def compare_family_fingerprints(
    train: object, exposed: object,
) -> dict:
    """Raw digests are kept internally. Return only non-identifying counts."""
    need(type(train) is list and type(exposed) is list
         and len(train) == EXPECTED_TRAIN_FRAMES
         and len(exposed) == EXPECTED_EXPOSED_FRAMES
         and all(type(d) is bytes and len(d) == 32 for d in train + exposed))
    train_set, exposed_set = set(train), set(exposed)
    need(len(train_set) == EXPECTED_TRAIN_FRAMES
         and len(exposed_set) == EXPECTED_EXPOSED_FRAMES)
    need(not train_set.intersection(exposed_set))
    return {
        "status": ROLE,
        "train_family": "RailKit",
        "exposed_research_family": "BeaconBoard",
        "original_train_source_scenes": len(TRAIN_SCENES),
        "public_exposed_source_scenes": len(EXPOSED_SCENES),
        "train_candidate_rgb_unique": len(train_set),
        "public_exposed_rgb_unique": len(exposed_set),
        "cross_family_rgb_identical": 0,
        "whole_source_and_rights_history_independently_proven": False,
        "official_train_admitted": False,
        "dev_admitted": False,
        "blind_ood_qualified": False,
        "model_generations": 0,
        "optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "can_release": False,
    }


def _exposed_original_frame_pair(
    browser: Any, source: Path, url: str, scene: tuple,
) -> tuple[bytes, bytes]:
    """Trusted source oracle; no model callback or model gold transfer."""
    target = EVAL_IDS[scene[4]]
    with original_scene(browser, source, url, scene) as page:
        initial = {station: (False, "") for station in EVAL_IDS}
        need(inspect_all_stations(page) == initial)
        first = source_capture_png(page)
        first_hash = decoded_rgb_digest(first)
        need(first_hash == decoded_rgb_digest(source_capture_png(page)))
        button = page.locator('button[data-station="' + target + '"]')
        need(button.count() == 1 and button.is_visible())
        box = button.bounding_box()
        need(box is not None and box["width"] >= 30 and box["height"] >= 30)
        x = int(box["x"] + box["width"] / 2)
        y = int(box["y"] + box["height"] / 2)
        need(0 <= x < WIDTH and 0 <= y < HEIGHT)
        native_click = json.dumps(
            {"actions": [{"kind": "click", "point": [x, y]}]},
            separators=(",", ":"),
        )
        need(execute_one(page, native_click) ==
             {"kind": "click", "executed": True})
        expected = {station: (station == target, "") for station in EVAL_IDS}
        need(inspect_all_stations(page) == expected)
        focused = page.evaluate("""() => {
            const el = document.activeElement;
            return el instanceof HTMLInputElement
              ? el.dataset.station : null;
        }""")
        need(focused == target and page.url == url)
        second = source_capture_png(page)
        second_hash = decoded_rgb_digest(second)
        need(first_hash != second_hash
             and second_hash == decoded_rgb_digest(source_capture_png(page)))
        return first_hash, second_hash


def audit_original_train_exposed_split(
    browser: Any, train_source: Path, exposed_source: Path, *, role: str,
) -> dict:
    """Stage complete sources first. No output data escapes before full audit."""
    need(type(role) is str and role == ROLE)
    need(isinstance(train_source, Path) and
         isinstance(exposed_source, Path))
    train_path, _ = checked_train_source(train_source)
    exposed_path, url = checked_exposed_source(exposed_source)
    need(train_path.resolve() != exposed_path.resolve()
         and RAILKIT_BLOB != BEACONBOARD_BLOB
         and len(TRAIN_SCENES) == 16 and len(EXPOSED_SCENES) == 12)

    train_fingerprints: list[bytes] = []
    callback_count = 0

    def _trusted_volatile_batch(
        rows: tuple[tuple[bytes, str, str], ...],
    ) -> bool:
        nonlocal callback_count
        need(callback_count == 0
             and type(rows) is tuple
             and len(rows) == EXPECTED_TRAIN_FRAMES)
        callback_count += 1
        for png, task, label in rows:
            need(type(task) is str and type(label) is str
                 and 10 < len(task) <= 700
                 and 20 < len(label) <= 150)
            train_fingerprints.append(decoded_rgb_digest(png))
        return True

    train_result = produce_two_step_train_candidates(
        browser, train_path, role=TRAIN_CANDIDATE_ROLE,
        trusted_batch_sink=_trusted_volatile_batch,
    )
    need(callback_count == 1 and
         train_result["candidate_handoff_mode"] == "one_shot"
         and train_result["train_candidate_messages"] ==
             EXPECTED_TRAIN_FRAMES
         and train_result["official_train_admitted"] is False
         and len(train_fingerprints) == EXPECTED_TRAIN_FRAMES)
    exposed_fingerprints: list[bytes] = []
    for scene in EXPOSED_SCENES:
        exposed_fingerprints.extend(
            _exposed_original_frame_pair(browser, exposed_path, url, scene)
        )
    return compare_family_fingerprints(
        train_fingerprints, exposed_fingerprints
    )
