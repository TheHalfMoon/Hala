"""H344: source-verified cross-family visual similarity for THREE public GUI families.

RailKit(32 candidate action-stage frames), SignalDesk(256 candidate
action-stage frames), BeaconBoard(24 PUBLIC EXPOSED research frames).
No source family is TRAIN/DEV/OOD approved, no learned gain is claimed.
Actual screenshots, decoded RGB digests, lossy dHashes, actions, prompts,
reference gold, coordinates, scene IDs never escape this trusted code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from h263.native_browser_executor import execute_one
from h319.beaconboard_dynamic_research import (
    IDS as BEACON_IDS, SCENES as BEACON_SCENES,
    checked_source as checked_beacon, inspect_all_stations, original_scene,
)
from h325.two_step_train_candidates import (
    ROLE as RAIL_ROLE, checked_source as checked_rail,
    produce_two_step_train_candidates,
)
from h338.original_large_train_candidates import (
    ROLE as SIGNAL_ROLE, checked_source as checked_signal,
    produce_large_train_candidates,
)
from h342.original_visual_neighborhoods import (
    HASH_BITS, BANDS, _band, _fingerprint,
)
from h322.source_capture import source_capture_png
from h199.native_action_sft import WIDTH, HEIGHT

ROLE = "H344_PUBLIC_ORIGINAL_THREE_FAMILY_VISUAL_AUDIT_ONLY"
COUNTS = {"RailKit": 32, "SignalDesk": 256, "BeaconBoard": 24}
PAIRS = (("RailKit", "SignalDesk"),
         ("RailKit", "BeaconBoard"),
         ("SignalDesk", "BeaconBoard"))


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H344_THREE_FAMILY_VISUAL_AUDIT_REFUSED")


def _source_owned_exposed_pair(
    browser: Any, source: Path, url: str, scene: tuple,
) -> tuple[bytes, bytes]:
    """Source oracle is trusted; no policy sees source DOM or target."""
    target = BEACON_IDS[scene[4]]
    with original_scene(browser, source, url, scene) as page:
        empty = {station: (False, "") for station in BEACON_IDS}
        need(inspect_all_stations(page) == empty)
        first = source_capture_png(page)
        fprint = _fingerprint(first)
        need(_fingerprint(source_capture_png(page))[1] == fprint[1])
        button = page.locator(
            'button[data-station="' + target + '"]')
        need(button.count() == 1 and button.is_visible())
        box = button.bounding_box()
        need(box is not None and box["width"] >= 30 and box["height"] >= 30)
        point = [int(box["x"] + box["width"] / 2),
                 int(box["y"] + box["height"] / 2)]
        need(all(type(v) is int for v in point)
             and 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT)
        action = json.dumps(
            {"actions": [{"kind": "click", "point": point}]},
            separators=(",", ":"))
        need(execute_one(page, action) == {"kind": "click", "executed": True})
        expected = {station: (station == target, "") for station in BEACON_IDS}
        need(inspect_all_stations(page) == expected)
        focus = page.evaluate("""() => {
            const e = document.activeElement;
            return e instanceof HTMLInputElement ? e.dataset.station : null;
        }""")
        need(focus == target and page.url == url)
        second = source_capture_png(page)
        second_fp = _fingerprint(second)
        need(second_fp[1] != fprint[1]
             and second_fp[1] == _fingerprint(source_capture_png(page))[1])
        return first, second


def cross_family_counts(families: object) -> dict:
    """Only category totals, with no individual frame/scene hashes leaked."""
    need(type(families) is dict and set(families) == set(COUNTS))
    digests = set()
    for name, required in COUNTS.items():
        values = families[name]
        need(type(values) is tuple and len(values) == required)
        for f in values:
            need(type(f) is tuple and len(f) == 2
                 and type(f[0]) is int
                 and 0 <= f[0] < (1 << HASH_BITS)
                 and type(f[1]) is bytes and len(f[1]) == 32
                 and f[1] not in digests)
            digests.add(f[1])
    need(len(digests) == sum(COUNTS.values()))
    reports = {}
    cross_total = 0
    cross_exact = 0
    for left, right in PAIRS:
        bands = {band: 0 for band in BANDS}
        a, b = families[left], families[right]
        for f1 in a:
            for f2 in b:
                distance = (f1[0] ^ f2[0]).bit_count()
                bands[_band(distance)] += 1
        n = len(a) * len(b)
        need(sum(bands.values()) == n)
        reports[left + "_vs_" + right] = {
            "pairs": n, "perceptual_distance_histogram": bands,
            "perceptual_near_0_to_32_pairs":
                sum(bands[band] for band in BANDS[:3]),
            "rgb_identical_pairs": 0,
        }
        cross_total += n
        cross_exact += bands["zero"]
    need(cross_total == (32 * 256 + 32 * 24 + 256 * 24))
    return {
        "status": ROLE,
        "public_source_frame_counts": dict(COUNTS),
        "candidate_source_families": 2,
        "public_exposed_not_blind_research_families": 1,
        "total_source_frames": 312,
        "cross_family_frame_pairs_tested": cross_total,
        "cross_family_rgb_identical": 0,
        "cross_family_lossy_perceptual_zero_pairs": cross_exact,
        "per_family_pair": reports,
        "perceptual_distance_is_lossy_not_semantic_or_rights_evidence": True,
        "training_admitted": False, "independent_dev_or_blind_ood": False,
        "actual_pretrained_or_adapted_generations": 0,
        "new_optimizer_steps": 0, "measured_model_gain": False,
        "can_release": False,
    }


def _trusted_batch_frames(batch: tuple, count: int) -> tuple:
    need(type(batch) is tuple and len(batch) == count)
    values = []
    for row in batch:
        need(type(row) is tuple and len(row) == 3
             and type(row[0]) is bytes
             and type(row[1]) is str and type(row[2]) is str)
        values.append(_fingerprint(row[0]))
    return tuple(values)


def audit_original_three_family_images(
    browser: Any, rail_path: Path, signal_path: Path, beacon_path: Path,
    *, role: str,
) -> dict:
    """Only after ALL original browser action stages pass emit counts."""
    need(type(role) is str and role == ROLE
         and all(isinstance(p, Path) for p in (
             rail_path, signal_path, beacon_path)))
    checked_rail(rail_path)
    checked_signal(signal_path)
    _, beacon_url = checked_beacon(beacon_path)
    need(len({rail_path.resolve(), signal_path.resolve(),
              beacon_path.resolve()}) == 3
         and len(BEACON_SCENES) == 12)
    sources = {}
    handoffs = {"RailKit": 0, "SignalDesk": 0}

    def sink(name: str, batch: tuple) -> bool:
        need(handoffs[name] == 0)
        handoffs[name] += 1
        sources[name] = _trusted_batch_frames(batch, COUNTS[name])
        return True

    rail = produce_two_step_train_candidates(
        browser, rail_path, role=RAIL_ROLE,
        trusted_batch_sink=lambda batch: sink("RailKit", batch))
    need(rail["train_candidate_messages"] == COUNTS["RailKit"]
         and rail["official_train_admitted"] is False)
    signal = produce_large_train_candidates(
        browser, signal_path,
        lambda batch: sink("SignalDesk", batch), role=SIGNAL_ROLE)
    need(signal["validated_native_action_stages"] == COUNTS["SignalDesk"]
         and signal["official_train_admitted"] is False)
    need(handoffs == {"RailKit": 1, "SignalDesk": 1})
    beacon_source = beacon_path.absolute()
    beacon_frames = []
    for scene in BEACON_SCENES:
        png1, png2 = _source_owned_exposed_pair(
            browser, beacon_source, beacon_url, scene)
        beacon_frames.extend((_fingerprint(png1), _fingerprint(png2)))
    sources["BeaconBoard"] = tuple(beacon_frames)
    return cross_family_counts(sources)
