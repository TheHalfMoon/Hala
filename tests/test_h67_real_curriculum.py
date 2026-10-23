"""H67 end-to-end stage using the REAL H61 curriculum re-derivation gate.

Only a tiny synthetic TEST fixture is used; no downloaded screenshots,
credentials, provider writes or GPU session is involved.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from h61.curriculum import export, load_export
from h67.stage import make_stage, restore_stage, SCHEMA
from tests.test_h61_curriculum import write_shard


class RealCurriculumRoundTrip(unittest.TestCase):
    def test_cpu_export_and_full_rederivation_after_gpu_hydration(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            cpu = parent / "cpu"
            corpus = cpu / "export"
            corpus.mkdir(parents=True)
            shard = write_shard(corpus, "slider", 2, 1)
            image_dir = shard / "images" / "slider"
            image_dir.mkdir(parents=True)
            labels = []
            for line in (shard / "labels.jsonl").read_text().splitlines():
                row = json.loads(line)
                jpeg = b"synthetic-fixture-image-" + row["image_id"].encode()
                (image_dir / (row["image_id"] + ".jpg")).write_bytes(jpeg)
                row["image_sha256"] = hashlib.sha256(jpeg).hexdigest()
                labels.append(row)
            (shard / "labels.jsonl").write_text(
                "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n"
                        for r in labels))
            source_manifest = json.loads((shard / "manifest.json").read_text())
            source_manifest["labels_sha256"] = hashlib.sha256(
                (shard / "labels.jsonl").read_bytes()).hexdigest()
            (shard / "manifest.json").write_text(json.dumps(source_manifest))
            export(corpus, target_rows=2, max_rows_per_image=1)
            (cpu / "public-dev-baseline-receipt.json").write_text(
                json.dumps({"schema": "synthetic-baseline-test-fixture"}))

            bundle = parent / "cpu.zip"
            sha = "d" * 40
            built = make_stage(cpu, bundle, sha)
            self.assertEqual(built["schema"], SCHEMA)
            self.assertEqual(built["media"], 2)
            remote = parent / "remote_run"
            restored = restore_stage(bundle, built["sha256"], remote, sha)
            self.assertEqual(restored["status"], "RESTORED_TRAIN_STAGE")
            manifest = json.loads((remote / "export/h61_drag_curriculum_manifest.json").read_text())
            rows = load_export(remote / "export/h61_drag_curriculum.jsonl", manifest)
            self.assertEqual(len(rows), 2)
            self.assertEqual(restored["dataset_sha256"], manifest["dataset_sha256"])


if __name__ == "__main__":
    unittest.main()
