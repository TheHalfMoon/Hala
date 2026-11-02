"""H352 invented-only paired study contracts; NO actual model/GPU here."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from h352.private_qwen_paired_snap import (
    aggregate_pairs, safe_report, stage_only_report,
)
from h352.verify_private_receipt import (
    parse_unique, verify_success, verify_stage_only,
)
from h352.build_private_payload import build, SCRIPT


def mock_pairs():
    result = []
    for lang in ("en", "ar"):
        for index in range(6):
            good = lang == "en" or index < 4
            result.append({"language": lang, "baseline": good,
                           "candidate": True, "changed": True})
    return result

HW = {"free_owner_private_t4": True, "t4_count": 2}


class H352InventedPairedProtocolTests(unittest.TestCase):
    def test_arithmetic_is_actual_paired_not_independent_scene_counts(self):
        groups = aggregate_pairs(mock_pairs())
        self.assertEqual(groups["ar"]["baseline_success"], 4)
        self.assertEqual(groups["ar"]["pixel_snap_success"], 6)
        self.assertEqual(groups["ar"]["paired_wins"], 2)
        self.assertEqual(groups["ar"]["paired_losses"], 0)
        self.assertEqual(groups["ar"]["ties_both_success"], 4)
        self.assertEqual(groups["en"]["baseline_success"], 6)
        self.assertEqual(groups["en"]["pixel_snap_success"], 6)
        self.assertEqual(groups["en"]["paired_wins"], 0)

    def test_unpaired_or_extra_data_cannot_be_promoted(self):
        for mutation in ("short", "extra", "language", "bool", "dict"):
            data = mock_pairs()
            if mutation == "short":
                data.pop()
            elif mutation == "extra":
                data[0]["raw_point"] = [40, 20]
            elif mutation == "language":
                data[0]["language"] = "xx"
            elif mutation == "bool":
                data[0]["baseline"] = 1
            else:
                data = {"rows": data}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate_pairs(data)

    def test_receipt_requires_all_counts_and_inference_no_training(self):
        receipt = safe_report(mock_pairs(), HW)
        got = verify_success(receipt)
        self.assertEqual(got["by_language"]["ar"]["paired_wins"], 2)
        self.assertFalse(got["adapted_model_gain_proven"])
        self.assertFalse(got["can_release"])
        for mutation in ("source", "model", "trained", "blinded", "release",
                         "hidden", "bad_wins", "bad_loss", "wrong_type",
                         "model_generations"):
            bad = copy.deepcopy(receipt)
            if mutation == "source":
                bad["source_commit"] = "0" * 40
            elif mutation == "model":
                bad["model_revision"] = "0" * 40
            elif mutation == "trained":
                bad["adapted_model_gain_proven"] = True
            elif mutation == "blinded":
                bad["fresh_blind_ood"] = True
            elif mutation == "release":
                bad["can_release"] = True
            elif mutation == "hidden":
                bad["per_case_point"] = [100, 200]
            elif mutation == "bad_wins":
                bad["by_language"]["ar"]["paired_wins"] = 5
            elif mutation == "bad_loss":
                bad["by_language"]["ar"]["paired_losses"] = 1
            elif mutation == "wrong_type":
                bad["by_language"]["ar"]["scenes"] = True
            else:
                bad["claimed_wrapped_model_generations"] = 11
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_success(bad)

    def test_json_strict_no_duplicates_nonfinite_or_raw_data(self):
        accepted = safe_report(mock_pairs(), HW)
        self.assertEqual(verify_success(parse_unique(
            json.dumps(accepted).encode("utf-8")))["claimed_model_generations"], 12)
        for raw in (b'{"k":1,"k":2}', b'{"x":NaN}',
                    b'{"x":"\\uZZZZ"}', b'x' * 5000):
            with self.subTest(raw_prefix=raw[:4]), self.assertRaises(ValueError):
                parse_unique(raw)

    def test_stage_only_failure_never_becomes_model_score(self):
        result = stage_only_report()
        self.assertFalse(verify_stage_only(result)["can_release"])
        with self.assertRaises(ValueError):
            verify_success(result)

    def test_fresh_private_code_only_bundle_pins_pixel_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path("h352")
            out = Path(temp) / "private"
            status = build(root / "private_qwen_paired_snap.py",
                           root / "pixel_button_snap.py", out)
            self.assertFalse(status["gpu_job_submitted"])
            self.assertEqual(status["uploaded_files"], 2)
            self.assertEqual(status["media_weights_gold_uploaded"], 0)
            self.assertEqual({f.name for f in out.iterdir()},
                             {SCRIPT, "kernel-metadata.json"})
            metadata = json.loads((out / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertFalse(metadata["dataset_sources"])
            uploaded = (out / SCRIPT).read_text(encoding="utf-8")
            self.assertIn("def snap_native_first_click(", uploaded)
            self.assertNotIn("H352_SNAP_SOURCE_PENDING", uploaded)
            self.assertIn("06596fdd70214be8f6729eb94615f569d9a7ec85", uploaded)
            with self.assertRaises(ValueError):
                build(root / "private_qwen_paired_snap.py",
                      root / "pixel_button_snap.py", out)


if __name__ == "__main__":
    unittest.main()
