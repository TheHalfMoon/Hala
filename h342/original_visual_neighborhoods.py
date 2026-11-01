"""H342: source-only visual-neighborhood audit for original SignalDesk.

Lossy perceptual similarity of public candidate frames; not semantic
equivalence, independent families, rights, TRAIN admission, Qwen or blind OOD.
No raw images, hashes, tasks, labels or points returned.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from h199.native_action_sft import WIDTH, HEIGHT
from h338.original_large_train_candidates import (
    ROLE as CANDIDATE_ROLE, SOURCE_BLOB, produce_large_train_candidates,
)

ROLE = "H342_SOURCE_ONLY_SIGNALDESK_VISUAL_NEIGHBORHOODS_NOT_ADMITTED"
FRAME_COUNT = 256
HASH_BITS = 1024
BANDS = ("zero", "1_to_8", "9_to_32", "33_to_128", "129_to_1024")


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H342_SOURCE_VISUAL_NEIGHBORHOOD_REFUSED")


def _fingerprint(png: bytes) -> tuple[int, bytes]:
    """Decode in RAM. A zero lossy hash distance is NOT RGB equality."""
    from PIL import Image, UnidentifiedImageError
    need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n"))
    try:
        with Image.open(BytesIO(png)) as im:
            need(im.format == "PNG" and im.size == (WIDTH, HEIGHT))
            rgb = im.convert("RGB")
            rgb.load()
            digest = sha256(rgb.tobytes()).digest()
            small = rgb.convert("L").resize((33, 32), Image.Resampling.BOX)
            pixels = list(small.getdata())
            need(len(pixels) == 33 * 32)
    except (OSError, ValueError, UnidentifiedImageError):
        raise ValueError("H342_SOURCE_VISUAL_NEIGHBORHOOD_REFUSED") from None
    bits = 0
    for y in range(32):
        offset = y * 33
        for x in range(32):
            bits = (bits << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    need(0 <= bits < 1 << HASH_BITS)
    return bits, digest


def _band(distance: int) -> str:
    need(type(distance) is int and 0 <= distance <= HASH_BITS)
    if distance == 0:
        return "zero"
    if distance <= 8:
        return "1_to_8"
    if distance <= 32:
        return "9_to_32"
    if distance <= 128:
        return "33_to_128"
    return "129_to_1024"


def _aggregate(fingerprints: tuple[tuple[int, bytes], ...]) -> dict:
    """Counts only. Frozen source order: 64 EN + 64 AR scenes, 2 stages."""
    need(type(fingerprints) is tuple and len(fingerprints) == FRAME_COUNT)
    rgb_seen = set()
    for item in fingerprints:
        need(type(item) is tuple and len(item) == 2)
        bits, digest = item
        need(type(bits) is int and 0 <= bits < 1 << HASH_BITS
             and type(digest) is bytes and len(digest) == 32
             and digest not in rgb_seen)
        rgb_seen.add(digest)
    histogram = {key: 0 for key in BANDS}
    within_language_near = 0
    cross_language_near = 0
    for i in range(FRAME_COUNT):
        for j in range(i + 1, FRAME_COUNT):
            distance = (fingerprints[i][0] ^ fingerprints[j][0]).bit_count()
            histogram[_band(distance)] += 1
            if distance <= 32:
                if (i < 128) == (j < 128):
                    within_language_near += 1
                else:
                    cross_language_near += 1
    total = FRAME_COUNT * (FRAME_COUNT - 1) // 2
    need(sum(histogram.values()) == total
         and within_language_near + cross_language_near
         == sum(histogram[k] for k in BANDS[:3]))
    return {
        "status": ROLE,
        "source_html_git_blob": SOURCE_BLOB,
        "source_frames": FRAME_COUNT,
        "decoded_rgb_unique_frames": FRAME_COUNT,
        "source_language_frames": {"en": 128, "ar_rtl": 128},
        "within_one_family_pairs": total,
        "perceptual_hamming_distance_pair_counts": histogram,
        "near_0_to_32_within_language_pairs": within_language_near,
        "near_0_to_32_cross_language_pairs": cross_language_near,
        "perceptual_zero_distance_but_distinct_rgb_pairs": histogram["zero"],
        "hash_is_lossy_not_semantic_similarity_or_leakage_proof": True,
        "independent_source_families_attested": False,
        "independent_rights_and_ml_screenshot_grant": False,
        "official_h199_train_admitted": False,
        "dev_or_blind_ood_qualified": False,
        "pretrained_or_adapted_model_generations": 0,
        "optimizer_steps": 0,
        "measured_model_gain": False,
        "can_release": False,
    }


def summarize_trusted_source_batch(batch: tuple) -> dict:
    """Never parse task or native label. Only source-owned decoded PNGs."""
    need(type(batch) is tuple and len(batch) == FRAME_COUNT)
    fingerprints = []
    for item in batch:
        need(type(item) is tuple and len(item) == 3
             and type(item[0]) is bytes
             and type(item[1]) is str and type(item[2]) is str)
        fingerprints.append(_fingerprint(item[0]))
    return _aggregate(tuple(fingerprints))


def audit_original_signaldesk(browser: Any, source_path: Path) -> dict:
    """Source-native validation precedes one ephemeral trusted batch sink."""
    need(isinstance(source_path, Path))
    report = None

    def sink(batch: tuple) -> bool:
        nonlocal report
        need(report is None)
        report = summarize_trusted_source_batch(batch)
        return True

    source = produce_large_train_candidates(
        browser, source_path, sink, role=CANDIDATE_ROLE)
    need(report is not None and source["source_scenes"] == 128
         and source["validated_native_action_stages"] == FRAME_COUNT
         and source["distinct_decoded_rgb_images"] == FRAME_COUNT
         and source["official_train_admitted"] is False
         and source["measured_model_improvement"] is False)
    return report
