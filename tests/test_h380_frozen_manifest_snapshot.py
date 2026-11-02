"""H380: no post-SHA manifest reread in historical freeze verification.

Invented metadata with public synthetic source code; NO TRAIN, model weights,
real screenshots, seals, signatures or private data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from development.freeze import freeze_bundle, verify_bundle


class H380ManifestSameBytesTests(unittest.TestCase):
    def test_manifest_swapped_after_sha_read_cannot_change_verified_identity(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "invented-frozen-source"
            frozen = freeze_bundle(root, {"name": "invented-frozen-settings"})
            path = root / "manifest.json"
            signed_raw = path.read_bytes()
            externally_pinned_sha = hashlib.sha256(signed_raw).hexdigest()
            forged = dict(frozen, invented_unauthorized_claim=True)
            original_read = Path.read_bytes
            swapped = [0]

            def read_and_swap(file):
                result = original_read(file)
                if Path(file) == path:
                    swapped[0] += 1
                    # Simulate a local actor mutating metadata immediately
                    # AFTER the verifier opens/hash-reads the original bytes.
                    path.write_bytes(json.dumps(
                        forged, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8"))
                return result

            with patch.object(Path, "read_bytes", read_and_swap):
                actual = verify_bundle(
                    root, require_media=False,
                    expected_manifest_sha256=externally_pinned_sha,
                )
            self.assertEqual(swapped[0], 1)
            self.assertNotIn("invented_unauthorized_claim", actual)
            self.assertEqual(actual, frozen)
            self.assertNotEqual(path.read_bytes(), signed_raw)

    def test_external_manifest_sha_still_strictly_required(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "invented-frozen-source"
            freeze_bundle(root, {"name": "invented-frozen-settings"})
            with self.assertRaisesRegex(ValueError, "external manifest hash mismatch"):
                verify_bundle(
                    root, require_media=False,
                    expected_manifest_sha256="0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "archived manifest"):
                verify_bundle(root, require_media=True)


if __name__ == "__main__":
    unittest.main()
