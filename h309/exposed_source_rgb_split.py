"""H309: source-only model-visible RGB split audit for exposed original GUIs.

This audits only the already-public ShiftDesk TRAIN *candidates* and
FieldPort exposed DEV *research*. It never admits data, opens private OOD,
runs Qwen, records media or labels, or certifies legal provenance.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image

from h285.train_only_browser_candidates import (
    ROLE as TRAIN_ROLE, SOURCE_BLOB as TRAIN_BLOB,
    produce_train_only_batch,
)
from h296.fieldport_sequence_research import (
    BLOB as DEV_BLOB, IDS, SCENES,
    checked_source as checked_dev_source,
    inspect_all_fields, original_scene,
)
from h264.screenshot_locked_research import _observed_png

MAX_PNG_BYTES = 4 * 1024 * 1024
FRAME_SIZE = (960, 720)
TRAIN_COUNT = 24
DEV_COUNT = 12


class SplitAuditRefused(ValueError):
    """No private data, hashes, scene names or labels in errors."""


def _need(ok: bool) -> None:
    if not ok:
        raise SplitAuditRefused("H309 SOURCE_SPLIT_REFUSED")


def model_visible_rgb_fingerprint(png: bytes) -> bytes:
    """Private ephemeral digest of exactly the RGB image Qwen would see."""
    try:
        _need(type(png) is bytes and 100 <= len(png) < MAX_PNG_BYTES
              and png.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(BytesIO(png)) as image:
            _need(image.format == "PNG" and image.size == FRAME_SIZE)
            rgb = image.convert("RGB")
            rgb.load()
            return sha256(rgb.tobytes()).digest()
    except Exception:
        raise SplitAuditRefused("H309 INVALID_MODEL_VISIBLE_FRAME") from None


class ExposedSplitRegistry:
    """Memory-only TRAIN/DEV research RGB disjointness; never an OOD seal."""

    def __init__(self) -> None:
        self._train: set[bytes] = set()
        self._dev: set[bytes] = set()
        self._sealed = False

    def _add(self, png: bytes, *, dev: bool) -> None:
        _need(not self._sealed)
        try:
            _need(len(self._train) == TRAIN_COUNT if dev else not self._dev)
            target = self._dev if dev else self._train
            limit = DEV_COUNT if dev else TRAIN_COUNT
            _need(len(target) < limit)
            digest = model_visible_rgb_fingerprint(png)
            _need(digest not in target)
            _need(not dev or digest not in self._train)
            target.add(digest)
        except Exception:
            self._sealed = True
            raise SplitAuditRefused("H309 SOURCE_SPLIT_REFUSED") from None

    def add_train_candidate(self, png: bytes) -> None:
        self._add(png, dev=False)

    def add_exposed_dev_research(self, png: bytes) -> None:
        self._add(png, dev=True)

    def finish(self) -> dict:
        _need(not self._sealed)
        self._sealed = True
        _need(len(self._train) == TRAIN_COUNT and len(self._dev) == DEV_COUNT
              and self._train.isdisjoint(self._dev))
        return {
            "status": "H309_EXPOSED_ORIGINAL_RGB_SPLIT_SOURCE_AUDIT_ONLY",
            "public_train_candidates": TRAIN_COUNT,
            "public_exposed_dev_research_screens": DEV_COUNT,
            "distinct_model_visible_rgb_screens": TRAIN_COUNT + DEV_COUNT,
            "two_original_source_families_checked": True,
            "official_train_admitted": False,
            "official_dev_admitted": False,
            "independent_blind_ood": False,
            "model_generations": 0,
            "optimizer_steps": 0,
            "adapted_gain_proven": False,
            "raw_images_labels_or_hashes_published": False,
            "can_release": False,
        }


def audit_original_exposed_rgb_split(
    browser: object, train_source: Path, dev_source: Path,
) -> dict:
    """Run genuine offline source browsers; retain only ephemeral RGB digests.

    H285 internally checks 24 source-owned native actions before calling
    the collector. H296 contributes 12 fresh initial-state screenshots.
    The collector deliberately ignores all source task and gold arguments.
    No policy/model is invoked, and no DEV reference action is executed.
    """
    from h285.train_only_browser_candidates import checked_source as checked_train
    registry = ExposedSplitRegistry()
    stage = "SOURCE_PREFLIGHT"
    try:
        _need(isinstance(train_source, Path) and isinstance(dev_source, Path))
        checked_train(train_source)
        dev_path, dev_url = checked_dev_source(dev_source)

        def only_frame(png: bytes, _private_task: str, _private_label: str) -> None:
            registry.add_train_candidate(png)

        stage = "TRAIN_SOURCE_BROWSER"
        proof = produce_train_only_batch(
            browser, train_source, only_frame, role=TRAIN_ROLE)
        stage = "TRAIN_SOURCE_PROOF"
        _need(type(proof) is dict
              and proof.get("role") == TRAIN_ROLE
              and proof.get("original_source_blob") == TRAIN_BLOB
              and proof.get("actual_browser_source_actions_successful") == TRAIN_COUNT
              and proof.get("english_candidates") == 12
              and proof.get("arabic_candidates") == 12
              and proof.get("official_train_manifest_admitted") is False
              and proof.get("independent_dev_ood_evaluated") is False
              and proof.get("pretrained_model_generations") == 0
              and proof.get("optimizer_steps") == 0
              and proof.get("public_private_media_or_labels_saved") is False)
        _need(DEV_BLOB == "14bc9041fabf59d76adb0fc78b6a16d6d53c6e80")
        stage = "DEV_SOURCE_BROWSER"
        for scene in SCENES:
            with original_scene(browser, dev_path, dev_url, scene) as page:
                _need(inspect_all_fields(page) == {item: "" for item in IDS})
                registry.add_exposed_dev_research(_observed_png(page))
        stage = "SPLIT_FINALIZE"
        return registry.finish()
    except Exception:
        # Safe fixed stage only: never expose raw source or per-case detail.
        raise SplitAuditRefused("H309 SOURCE_SPLIT_AUDIT_FAILED_" + stage) from None
