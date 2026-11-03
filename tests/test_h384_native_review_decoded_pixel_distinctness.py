"""H384: PNG byte identity is NOT decoded visual-frame identity.

All fixtures are locally generated flat-color 960x720 PNGs under a
temporary directory. No browser, real gold, data admission, model or GPU.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS, source_task_prompt
from h359.native_candidate_bridge import (
    ROLE357, SCHEMA, SOURCES, audit_review_bundle,
)

FAMILY = "controlgarden-single-app"


def invented_review_bundle(root: Path, *, alias_pixels: bool) -> None:
    # Pillow is installed ONLY for synthetic image validation and the
    # existing H359 original source audit; never fetch external images.
    from PIL import Image, PngImagePlugin

    root.mkdir()
    images = root / "images"
    images.mkdir()
    source = SOURCES[FAMILY]
    rows = []
    byte_hashes = set()

    for index in range(64):
        # The same 960x720 raster with 64 different PNG tEXt fields yields
        # 64 byte-distinct PNG files but *one* identical decoded RGBA frame.
        color = ((11, 23, 47) if alias_pixels
                 else (index, (3 * index) % 256, (5 * index) % 256))
        image = Image.new("RGB", (960, 720), color)
        info = PngImagePlugin.PngInfo()
        info.add_text("h384-invented-metadata", "nonce-%03d" % index)
        stream = BytesIO()
        image.save(stream, format="PNG", pnginfo=info)
        raw = stream.getvalue()
        digest = sha256(raw).hexdigest()
        assert digest not in byte_hashes
        byte_hashes.add(digest)
        path = images / ("invented-%03d.png" % index)
        path.write_bytes(raw)
        language = "en" if index % 2 == 0 else "ar"
        task = ("Select an invented card" if language == "en"
                else "اختر البطاقة الخيالية")
        rows.append({
            "id": "h384-invented-%03d" % index,
            "family": FAMILY,
            "split": source["split"],
            "language": language,
            "prompt": source_task_prompt(task),
            "completion": '{"actions":[{"kind":"click","point":[200,110]}]}',
            "image_path": "images/" + path.name,
            "image_sha256": digest,
            "original_source_sha256": source["source_sha"],
        })

    (root / "candidate-rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in rows), encoding="utf-8",
    )
    manifest = {
        "schema": source["schema"],
        "role": ROLE357,
        "original_source_sha256": source["source_sha"],
        "families": [FAMILY],
        "case_count": 64,
        "language_counts": {"en": 32, "ar": 32},
        "original_native_browser_successes": 64,
        "exact_pixel_double_captures": 64,
        "official_train_admitted": False,
        "independent_rights_reviewed": False,
        "fresh_blind_ood": False,
        "gpu_calls": 0,
        "can_train": False,
        "can_release": False,
    }
    (root / "candidate-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8",
    )


class H384DecodedDistinctnessTests(unittest.TestCase):
    def test_metadata_only_png_differences_cannot_claim_64_visual_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate"
            invented_review_bundle(root, alias_pixels=True)
            with self.assertRaisesRegex(
                ValueError, "H359_UNADMITTED_NATIVE_SFT_REVIEW_REFUSED"
            ):
                audit_review_bundle(root, FAMILY)

    def test_genuinely_distinct_decoded_pixels_pass_unadmitted_review(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate"
            invented_review_bundle(root, alias_pixels=False)
            report = audit_review_bundle(root, FAMILY)
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["records"], 64)
            self.assertEqual(report["distinct_images"], 64)
            self.assertEqual(report["distinct_decoded_rgba_images"], 64)
            self.assertEqual((report["en"], report["ar"]), (32, 32))
            self.assertFalse(report["official_train_admitted"])
            self.assertFalse(report["independent_rights_verified"])
            self.assertFalse(report["can_train"])
            self.assertFalse(report["blind_ood"])
            self.assertFalse(report["can_release"])
            self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)


if __name__ == "__main__":
    unittest.main()
