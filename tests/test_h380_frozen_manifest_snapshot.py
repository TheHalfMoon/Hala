"""H380: immutable freeze verifier cannot promote substituted manifest in trainer.

Real source freeze reference remains byte-for-byte unchanged. All test metadata
is invented, with no TRAIN, model, patient, weights, screenshot or blind OOD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from development.freeze import freeze_bundle
from training.development_data import _verified_frozen_manifest


class H380ManifestSameBytesTests(unittest.TestCase):
    def test_forged_manifest_returned_after_sha_snapshot_is_denied(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "invented-frozen-source"
            frozen = freeze_bundle(root, {"name": "invented-frozen-settings"})
            path = root / "manifest.json"
            signed_raw = path.read_bytes()
            externally_pinned_sha = hashlib.sha256(signed_raw).hexdigest()
            forged = dict(frozen, invented_unauthorized_claim=True)
            calls = []

            def simulated_legacy_verifier(verified_root, *,
                                          require_media, expected_manifest_sha256):
                calls.append((verified_root, require_media, expected_manifest_sha256))
                # Imitate legacy verifier reading different bytes after an
                # earlier SHA check while returning the forged metadata.
                path.write_bytes(json.dumps(
                    forged, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"))
                return forged

            with patch("training.development_data.verify_bundle",
                       side_effect=simulated_legacy_verifier):
                with self.assertRaisesRegex(
                    ValueError, "Frozen manifest changed after external SHA"
                ):
                    _verified_frozen_manifest(root, externally_pinned_sha)
            self.assertEqual(calls, [(root, True, externally_pinned_sha)])
            self.assertNotEqual(path.read_bytes(), signed_raw)

    def test_hash_bound_unmodified_manifest_passes_test_scoped_verifier(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "invented-frozen-source"
            frozen = freeze_bundle(root, {"name": "invented-frozen-settings"})
            pinned_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            with patch("training.development_data.verify_bundle",
                       return_value=frozen) as verifier:
                self.assertEqual(_verified_frozen_manifest(root, pinned_sha), frozen)
                verifier.assert_called_once_with(
                    root, require_media=True, expected_manifest_sha256=pinned_sha)

    def test_wrong_or_missing_external_digest_is_denied_before_verifier(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "invented-frozen-source"
            freeze_bundle(root, {"name": "invented-frozen-settings"})
            with patch("training.development_data.verify_bundle") as verifier:
                for pin in ("0" * 64, None, "ABC" * 21):
                    with self.subTest(pin=pin), self.assertRaises(ValueError):
                        _verified_frozen_manifest(root, pin)
                verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
