"""H353 entirely invented CPU tests, NO Qwen inference or GPU job."""
from __future__ import annotations
import copy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h352.private_qwen_paired_snap import (
    safe_report, paired_arm_order, aggregate_pairs,
)
from h319.beaconboard_dynamic_research import SCENES
from tests.test_h352_private_paired_snap import mock_pairs, HW
from h353.build_private_payload import (
    build, SOURCE_REV, SCRIPT, REPORT, STATUS, FLAG, KERNEL_ID,
)
from h353.verify_private_receipt import parse_unique, verify_success, verify_stage_only


def invented_report():
    data = safe_report(mock_pairs(), HW)
    data["status"] = STATUS
    return data


class H353FrozenBalancedResearchTests(unittest.TestCase):
    def test_frozen_counterbalanced_order_on_all_original_scenes(self):
        for lang in ("en", "ar"):
            cases = [paired_arm_order(scene) for scene in SCENES
                     if scene[1] == lang]
            self.assertEqual(len(cases), 6)
            self.assertEqual(cases.count(("candidate", "baseline")), 3)
            self.assertEqual(cases.count(("baseline", "candidate")), 3)

    def test_unique_private_code_only_frozen_experiment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "private"
            status = build(Path(".").resolve(), dest)
            self.assertEqual(status["source_rev"], SOURCE_REV)
            self.assertFalse(status["gpu_job_submitted"])
            self.assertEqual(status["file_count"], 2)
            self.assertEqual(status["media_or_weights_uploaded"], 0)
            self.assertEqual(
                {f.name for f in dest.iterdir()},
                {SCRIPT, "kernel-metadata.json"})
            script = (dest / SCRIPT).read_bytes()
            self.assertEqual(status["code_sha256"], sha256(script).hexdigest())
            script_text = script.decode("utf-8")
            for required in (FLAG, REPORT, STATUS, "def paired_arm_order(scene):",
                             'candidate_first = bool((index % 2) ^ (scene[1] == "ar"))'):
                self.assertIn(required, script_text)
            self.assertNotIn("hala-h352-private-paired-snap-aggregate.json",
                             script_text)
            meta = json.loads((dest / "kernel-metadata.json").read_text())
            self.assertEqual(meta["id"], KERNEL_ID)
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], [])
            self.assertEqual(meta["model_sources"], [])
            with self.assertRaises(ValueError):
                build(Path(".").resolve(), dest)

    def test_strict_private_aggregate_enforces_paired_arithmetic(self):
        data = invented_report()
        accepted = verify_success(parse_unique(json.dumps(data).encode()))
        self.assertEqual(accepted["by_language"]["ar"]["paired_wins"], 2)
        self.assertFalse(accepted["adapted_model_gain_proven"])
        for mutation in ("old", "gain", "blind", "mismatch", "extra",
                         "forged", "bool"):
            row = copy.deepcopy(data)
            if mutation == "old":
                row["status"] = "H352_PRIVATE_REAL_QWEN_EXPOSED_PIXEL_SNAP_PAIRED"
            elif mutation == "gain":
                row["adapted_model_gain_proven"] = True
            elif mutation == "blind":
                row["fresh_blind_ood"] = True
            elif mutation == "mismatch":
                row["by_language"]["ar"]["pixel_snap_success"] = 4
            elif mutation == "extra":
                row["private_coordinate"] = [500, 250]
            elif mutation == "forged":
                row["by_language"]["ar"]["paired_losses"] = 4
            else:
                row["by_language"]["ar"]["scenes"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_success(row)

    def test_duplicate_key_and_stage_only_cannot_claim_model_gain(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b"x" * 4096):
            with self.assertRaises(ValueError):
                parse_unique(raw)
        stage = {
            "status": "H353_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE",
            "stage": "MODEL_LOAD", "can_release": False,
            "fresh_blind_ood": False, "adapted_model_gain_proven": False,
            "official_train_admitted": False,
            "raw_private_data_saved": False,
        }
        self.assertFalse(verify_stage_only(stage)["can_release"])
        with self.assertRaises(ValueError):
            verify_success(stage)


if __name__ == "__main__":
    unittest.main()
