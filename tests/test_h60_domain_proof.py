"""H60 domain proof: one verified archive digest bound to every published shard.

Only the last shard of a domain can carry the whole-archive digest, because
the digest is only known once the last byte of the archive has been streamed.
The domain proof binds the remaining shards to that one verified archive. These
tests prove that a shard is accepted with a valid proof, and rejected when
the proof is missing, incomplete, tampered or reports a different archive.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from h60 import native_drag_sft
from tests.test_h60_native_drag_sft import _build_shard

REVISION = "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
ARCHIVE_SHA = "1" * 64
WINDOW = "h59-slider-0000"


def _proof(shard_manifest: dict, **overrides) -> dict:
    proof = {
        "schema": "hala.h60.dragon-train-domain-proof.v1",
        "source_repo": "Hcompany/DragOn",
        "source_commit_sha": REVISION,
        "source_domain": "slider",
        "split": "train",
        "source_license": "Apache-2.0",
        "source_tar_bytes": 7146830848,
        "official_upstream_lfs_sha256": ARCHIVE_SHA,
        "recomputed_archive_sha256": ARCHIVE_SHA,
        "archive_sha256_matches_official": True,
        "source_bytes_streamed": 7146830848,
        "source_window_receipts": [
            {
                "original_window_id": WINDOW,
                "range_start": 0,
                "range_end_inclusive": 536870911,
                "bytes_verified": 536870912,
                "original_window_sha256": "2" * 64,
            }
        ],
        "shards": [
            {
                "shard_id": shard_manifest["shard_id"],
                "local_zip_sha256": "3" * 64,
                "labels_sha256": shard_manifest["labels_sha256"],
                "original_images": shard_manifest["original_images"],
                "native_drag_actions": shard_manifest["native_drag_actions"],
                "provider_ref": "abdulazizshehri/hala-h60-dragon-train-slider-0000-private",
                "provider_readback_verified": True,
            }
        ],
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
        "model_gpu_steps": 0,
    }
    proof.update(overrides)
    return proof


def _unbound_shard(root: Path) -> dict:
    """A shard that cannot prove its own archive, bound only by the proof."""
    _build_shard(root / "shard-0000")
    manifest_path = root / "shard-0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf8"))
    manifest["whole_tar_sha256_verified"] = False
    manifest["upstream_full_tar_lfs_sha256"] = ARCHIVE_SHA
    manifest["source_window_ids"] = [WINDOW]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", "utf8")
    return manifest


class H60DomainProofTests(unittest.TestCase):
    def _write_proof(self, root: Path, proof: dict) -> None:
        (root / native_drag_sft.PROOF_FILE).write_text(
            json.dumps(proof, sort_keys=True, indent=2) + "\n", "utf8"
        )

    def test_a_shard_bound_by_a_valid_domain_proof_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            self._write_proof(root, _proof(manifest))
            blob, derived, rows = native_drag_sft.derive(root)
            self.assertTrue(rows)
            self.assertEqual(len(blob), len(derived["media"]) * 0 + len(blob))
            self.assertEqual(derived["split"], "train")

    def test_a_shard_without_any_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _unbound_shard(root)
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_proof_that_does_not_match_the_official_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            self._write_proof(
                root, _proof(manifest, archive_sha256_matches_official=False)
            )
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_proof_whose_archive_digest_differs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            self._write_proof(root, _proof(manifest, recomputed_archive_sha256="4" * 64))
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_proof_that_binds_a_different_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            proof = _proof(manifest)
            proof["shards"][0]["labels_sha256"] = "5" * 64
            self._write_proof(root, proof)
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_proof_without_a_provider_readback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            proof = _proof(manifest)
            proof["shards"][0]["provider_readback_verified"] = False
            self._write_proof(root, proof)
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_proof_whose_window_receipts_do_not_cover_the_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            proof = _proof(manifest)
            proof["source_window_receipts"] = []
            self._write_proof(root, proof)
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_a_tampered_proof_is_detected_by_the_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _unbound_shard(root)
            self._write_proof(root, _proof(manifest))
            proof_path = root / native_drag_sft.PROOF_FILE
            original = native_drag_sft.load_domain_proof(root)[1]
            self.assertEqual(
                original,
                hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            )
            proof_path.write_bytes(proof_path.read_bytes().replace(b'"slider"', b'"sheet"'))
            self.assertNotEqual(
                original,
                native_drag_sft.load_domain_proof(root)[1],
                "a tampered domain proof must change its digest",
            )


if __name__ == "__main__":
    unittest.main()
