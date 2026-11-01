"""H324 invented-only code-bundle tests: never submit, never run GPU."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from h324.build_private_payload import build, SCRIPT, KERNEL_ID


class PrivateCodeOnlyBundle(unittest.TestCase):
    def test_actual_pinned_source_builds_two_file_private_bundle(self):
        source = Path(__file__).resolve().parents[1] / "h324" / (
            "private_pretrained_beaconboard_pilot.py")
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "private-only"
            result = build(source, dest)
            self.assertFalse(result["gpu_run_executed"])
            self.assertEqual(result["media_weights_gold_uploaded"], 0)
            self.assertEqual({p.name for p in dest.iterdir()},
                             {SCRIPT, "kernel-metadata.json"})
            metadata = json.loads((dest / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], KERNEL_ID)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            assembled = (dest / SCRIPT).read_text()
            ast.parse(assembled)
            self.assertIn("HALA_H324_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4",
                          assembled)
            self.assertIn("651364ab952c7e90dabb4c7c240e312ebd8d417f",
                          assembled)
            with self.assertRaises(ValueError):
                build(source, dest)

    def test_reject_source_not_h324_or_missing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                build(Path(root) / "not-there.py", Path(root) / "out")


if __name__ == "__main__":
    unittest.main()
