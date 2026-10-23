"""H74 zero-network synthetic cache probe tests: no Hugging Face downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h74.weight_cache_preflight import REVISION, inspect_snapshot, local_only_probe


class CachePreflight(unittest.TestCase):
    def make_snapshot(self, root: Path, *, shard_names=("s1.safetensors",)):
        snapshot = root / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf8")
        weight_map = {"language_model.weight": shard_names[0]}
        for name in shard_names:
            (snapshot / name).write_bytes(b"fake synthetic shard test bytes")
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf8")
        return snapshot

    def test_offline_cold_cache_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                inspect_snapshot(Path(temp) / "absent")

    def test_mocked_local_only_snapshot_never_downloads(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = self.make_snapshot(Path(temp))
            fetch = mock.Mock(return_value=str(snapshot))
            report = local_only_probe(Path(temp), snapshot_fetch=fetch)
            self.assertTrue(fetch.call_args.kwargs["local_files_only"])
            self.assertEqual(fetch.call_args.kwargs["revision"], REVISION)
            self.assertEqual(report["status"], "CACHED_WEIGHT_FILES_CANDIDATE_ONLY")
            self.assertFalse(report["runtime_qualified"])
            self.assertFalse(report["checkpoint_produced"])

    def test_missing_weight_blocks_false_warm_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = self.make_snapshot(Path(temp))
            (snapshot / "s1.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                inspect_snapshot(snapshot)

    def test_wrong_model_revision_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = self.make_snapshot(Path(temp))
            other = snapshot.with_name("b" * 40)
            snapshot.rename(other)
            with self.assertRaisesRegex(ValueError, "revision"):
                inspect_snapshot(other)

    def test_traversal_weight_name_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = self.make_snapshot(Path(temp))
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": "../../stolen.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "untrusted"):
                inspect_snapshot(snapshot)

    def test_external_symlink_weight_refused_when_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self.make_snapshot(root)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"external")
            weight = snapshot / "s1.safetensors"
            weight.unlink()
            try:
                weight.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "escaping"):
                inspect_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
