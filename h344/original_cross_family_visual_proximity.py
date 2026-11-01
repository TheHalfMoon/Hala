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
    # Same visual pipeline and screenshot custody: compare all source
    # permutations *within* each original GUI family, with no new renders.
    # This reveals synthetic monoculture risk without assuming that a
    # perceptual hash defines independent samples.
    within_reports = {}
    within_total = 0
    for family, originals in families.items():
        histogram = {band: 0 for band in BANDS}
        for i, left in enumerate(originals):
            for right in originals[i + 1:]:
                histogram[_band((left[0] ^ right[0]).bit_count())] += 1
        n = len(originals) * (len(originals) - 1) // 2
        need(sum(histogram.values()) == n)
        within_reports[family] = {
            "pairs": n,
            "perceptual_distance_histogram": histogram,
            "perceptual_near_0_to_32_pairs": sum(
                histogram[band] for band in BANDS[:3]),
            "rgb_identical_pairs": 0,
        }
        within_total += n
    need(within_total == 33412)
    return {
        "status": ROLE,
        "public_source_frame_counts": dict(COUNTS),
        "candidate_source_families": 2,
        "public_exposed_not_blind_research_families": 1,
        "total_source_frames": 312,
        "cross_family_frame_pairs_tested": cross_total,
        "within_family_frame_pairs_tested": within_total,
        "within_family_pair": within_reports,
        "cross_family_rgb_identical": 0,
        "cross_family_lossy_perceptual_zero_pairs": cross_exact,
        "per_family_pair": reports,
        "perceptual_distance_is_lossy_not_semantic_or_rights_evidence": True,
        "training_admitted": False, "independent_dev_or_blind_ood": False,
        "actual_pretrained_or_adapted_generations": 0,
        "new_optimizer_steps": 0, "measured_model_gain": False,
        "can_release": False,
    }



def source_only_public_band_receipt(report: object) -> dict:
    """Allowlist only tiny aggregate PUBLIC source-owned perceptual bins.

    No frame or sample fingerprint, source gold, raw image, prediction, or
    private model action can be accepted or serialized by this receipt.
    An observed histogram is *not* evidence of rights, semantic OOD,
    trained model gain, or independent sample size.
    """
    need(type(report) is dict and set(report) == {
        "status", "public_source_frame_counts", "candidate_source_families",
        "public_exposed_not_blind_research_families",
        "total_source_frames", "cross_family_frame_pairs_tested",
        "within_family_frame_pairs_tested", "within_family_pair",
        "cross_family_rgb_identical",
        "cross_family_lossy_perceptual_zero_pairs",
        "per_family_pair",
        "perceptual_distance_is_lossy_not_semantic_or_rights_evidence",
        "training_admitted", "independent_dev_or_blind_ood",
        "actual_pretrained_or_adapted_generations", "new_optimizer_steps",
        "measured_model_gain", "can_release",
    })
    need(report["status"] == ROLE
         and report["public_source_frame_counts"] == COUNTS
         and report["candidate_source_families"] == 2
         and report["public_exposed_not_blind_research_families"] == 1
         and report["total_source_frames"] == 312
         and report["cross_family_frame_pairs_tested"] == 15104
         and type(report["within_family_frame_pairs_tested"]) is int
         and report["within_family_frame_pairs_tested"] == 33412
         and report["cross_family_rgb_identical"] == 0
         and report[
             "perceptual_distance_is_lossy_not_semantic_or_rights_evidence"
         ] is True
         and all(report[k] is False for k in (
             "training_admitted", "independent_dev_or_blind_ood",
             "measured_model_gain", "can_release",
         ))
         and type(report["actual_pretrained_or_adapted_generations"]) is int
         and report["actual_pretrained_or_adapted_generations"] == 0
         and type(report["new_optimizer_steps"]) is int
         and report["new_optimizer_steps"] == 0)
    original = report["per_family_pair"]
    expected_keys = {left + "_vs_" + right for left, right in PAIRS}
    need(type(original) is dict and set(original) == expected_keys)
    safe_bins: dict[str, dict[str, int]] = {}
    total_exact = 0
    for left, right in PAIRS:
        name = left + "_vs_" + right
        entry = original[name]
        n = COUNTS[left] * COUNTS[right]
        need(type(entry) is dict and set(entry) == {
            "pairs", "perceptual_distance_histogram",
            "perceptual_near_0_to_32_pairs", "rgb_identical_pairs",
        } and type(entry["pairs"]) is int and entry["pairs"] == n
             and type(entry["rgb_identical_pairs"]) is int
             and entry["rgb_identical_pairs"] == 0)
        hist = entry["perceptual_distance_histogram"]
        need(type(hist) is dict and set(hist) == set(BANDS)
             and all(type(hist[k]) is int and 0 <= hist[k] <= n
                     for k in BANDS)
             and sum(hist.values()) == n
             and type(entry["perceptual_near_0_to_32_pairs"]) is int
             and entry["perceptual_near_0_to_32_pairs"]
                 == sum(hist[k] for k in BANDS[:3]))
        safe_bins[name] = {k: hist[k] for k in BANDS}
        total_exact += hist["zero"]
    need(type(report["cross_family_lossy_perceptual_zero_pairs"]) is int
         and report["cross_family_lossy_perceptual_zero_pairs"] == total_exact)
    within_original = report["within_family_pair"]
    need(type(within_original) is dict
         and set(within_original) == set(COUNTS))
    safe_within = {}
    for family, count in COUNTS.items():
        item = within_original[family]
        n = count * (count - 1) // 2
        need(type(item) is dict and set(item) == {
            "pairs", "perceptual_distance_histogram",
            "perceptual_near_0_to_32_pairs", "rgb_identical_pairs",
        } and type(item["pairs"]) is int and item["pairs"] == n
             and type(item["rgb_identical_pairs"]) is int
             and item["rgb_identical_pairs"] == 0)
        hist = item["perceptual_distance_histogram"]
        need(type(hist) is dict and set(hist) == set(BANDS)
             and all(type(hist[k]) is int and 0 <= hist[k] <= n
                     for k in BANDS)
             and sum(hist.values()) == n
             and type(item["perceptual_near_0_to_32_pairs"]) is int
             and item["perceptual_near_0_to_32_pairs"]
                 == sum(hist[k] for k in BANDS[:3]))
        safe_within[family] = {k: hist[k] for k in BANDS}
    return {
        "status": "H344_PUBLIC_EXPOSED_SOURCE_ONLY_AGGREGATE_BANDS",
        "public_source_frames": 312,
        "cross_family_compared_pairs": 15104,
        "within_family_compared_pairs": 33412,
        "decoded_rgb_identical_frames_refused": True,
        "per_family_pair_bins": safe_bins,
        "within_family_pair_bins": safe_within,
        "not_rights_or_semantic_ood_or_model_gain": True,
        "train_admitted": False, "blind_ood": False,
        "trained_gain": False, "release": False,
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
