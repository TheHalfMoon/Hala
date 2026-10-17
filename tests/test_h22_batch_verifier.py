"""Independent CPU-only adversarial tests of Kaggle Saved Version H22 evidence."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import verify_h22_batch as audit
from tools import h22_batch_operator as batch


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write(root):
    assigned = {"schema": "hala.h22.batch_assigned_hardware.v1",
        "provider": "kaggle", "kernel_run_type": "Batch", "gpu_name": "Tesla T4",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "source_commit": batch.SOURCE, "source_freeze_sha256": batch.FREEZE,
        "h19_manifest_sha256": batch.H19_MANIFEST,
        "adapter_manifest_sha256": batch.ADAPTER_MANIFEST,
        "batch_profile_order": list(batch.MODES), "training_runs": 0,
        "no_adoption": True}
    raw_assigned = canonical(assigned)
    (root / "assigned-hardware.json").write_bytes(raw_assigned)
    status = {"schema": "hala.h22.batch_test_triplet.v1",
              "status": "completed_unqualified", "source_commit": batch.SOURCE,
              "earlier_int8_outcome": "unknown", "earlier_nf4_outcome": "not_started",
              "training_runs": 0, "model_adopted": False,
              "assigned_hardware": assigned, "profiles": []}
    for i, mode in enumerate(batch.MODES):
        name = f"profile-{i}-{mode}-test"
        folder = root / name
        folder.mkdir()
        complete = {"all": 2, "en": 2, "ar": 0}
        measured = {"status": "measured", "source_commit": batch.SOURCE,
            "model_revision": batch.MODEL_REV, "model_id": "Qwen/Qwen3-VL-2B-Instruct", "split": "test",
            "gpu_uuid": assigned["gpu_uuid"], "adapter_loaded": True,
            "precision_profile": {"mode": mode, "source_freeze_sha256": batch.FREEZE},
            "summary": {"all": {"episodes": 8, "episodes_complete": 2},
                        "en": {"episodes": 4, "episodes_complete": 2},
                        "ar": {"episodes": 4, "episodes_complete": 0}}}
        raw_result = canonical(measured)
        profile = {"mode": mode, "split": "test", "status": "finished", "returncode": 0,
                   "gpu_uuid": assigned["gpu_uuid"], "result_status": "measured",
                   "result_sha256": audit.sha(raw_result), "completed_by_language": complete}
        status["profiles"].append(profile)
        (folder / "profile-status.json").write_bytes(canonical(profile))
        (folder / "profile.log").write_text("synthetic fixture only", encoding="utf-8")
        (folder / "run").mkdir()
        (folder / "run/result.json").write_bytes(raw_result)
        with zipfile.ZipFile(root / (name + "-evidence.zip"), "w") as z:
            for path in folder.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(folder).as_posix())
            z.writestr("assigned-hardware.json", raw_assigned)
            z.writestr("h22_batch_operator.py", Path(batch.__file__).read_bytes())
        archived = (root / (name + "-evidence.zip")).read_bytes()
        profile["archive_sha256"] = audit.sha(archived)
        profile["archive_bytes"] = len(archived)
    (root / "batch-status.json").write_bytes(canonical(status))
    return status


class SavedVersionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.status = write(self.root)

    def test_complete_three_profile_retention(self):
        report = audit.verify(self.root)
        self.assertTrue(report["all_three_profile_archives_verified"])
        self.assertFalse(report["general_task_quality_qualified"])
        self.assertEqual([x["mode"] for x in report["profiles"]], list(batch.MODES))

    def test_earlier_unknown_must_stay_unknown(self):
        self.status["earlier_int8_outcome"] = "measured"
        (self.root / "batch-status.json").write_bytes(canonical(self.status))
        with self.assertRaises(ValueError):
            audit.verify(self.root)

    def test_missing_nf4_zip_fails(self):
        (self.root / "profile-2-nf4-test-evidence.zip").unlink()
        with self.assertRaises(ValueError):
            audit.verify(self.root)

    def test_tampered_zip_fails_sha(self):
        target = self.root / "profile-1-int8-test-evidence.zip"
        target.write_bytes(target.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            audit.verify(self.root)

    def test_false_hardware_identity_fails(self):
        self.status["assigned_hardware"]["gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
        (self.root / "batch-status.json").write_bytes(canonical(self.status))
        with self.assertRaises(ValueError):
            audit.verify(self.root)

    def test_wrong_completion_count_fails(self):
        self.status["profiles"][0]["completed_by_language"]["all"] = 8
        (self.root / "batch-status.json").write_bytes(canonical(self.status))
        with self.assertRaises(ValueError):
            audit.verify(self.root)

    def test_duplicate_json_key_fails(self):
        with self.assertRaises(ValueError):
            audit.strict_object(b'{"x":1,"x":2}')

    def test_nonfinite_json_fails(self):
        with self.assertRaises(ValueError):
            audit.strict_object(b'{"elapsed":NaN}')

    def test_missing_saved_status_does_not_claim_retention(self):
        (self.root / "batch-status.json").unlink()
        with self.assertRaises(ValueError):
            audit.verify(self.root)


if __name__ == "__main__":
    unittest.main()
