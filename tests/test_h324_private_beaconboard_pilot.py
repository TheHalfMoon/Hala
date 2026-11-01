"""H324 invented-only CPU contract tests. NO real Qwen, GPU or media."""
import unittest

from h324.private_pretrained_beaconboard_pilot import (
    HEAD, HTML_BLOB, MODEL, REVISION, INDICES, ROW_KEYS,
    STATUS, aggregate, safe_failure,
)


def row(n=2, d=2, ok=True, why="none"):
    return {
        "status": "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY",
        "source_original_task_success": ok,
        "real_model_generations_claimed_by_wrapper": n,
        "native_actions_dispatched": d,
        "source_categorical_failure": why,
        "pretrained_model_identity_independently_attested": False,
        "source_public_exposed": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "trained_model_gain_proven": False,
        "can_release": False,
    }


GPU = {"free_t4": True, "t4_count": 1}


class ProspectivePrivateAggregate(unittest.TestCase):
    def test_pinned_distinct_pretrained_source_only(self):
        self.assertEqual(INDICES, (0, 6))
        self.assertEqual(len(HEAD), 40)
        self.assertEqual(len(HTML_BLOB), 40)
        self.assertEqual(MODEL, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(REVISION), 40)

    def test_two_genuine_success_outcomes_only_aggregate(self):
        receipt = aggregate([row(), row()], GPU)
        self.assertEqual(receipt["status"], STATUS)
        self.assertEqual(receipt["original_four_station_task_successes"], 2)
        self.assertEqual(receipt["real_pretrained_model_generations"], 4)
        self.assertEqual(receipt["native_browser_actions_dispatched"], 4)
        self.assertEqual(receipt["source_research_scenes"], 2)
        self.assertFalse(receipt["adapted_gain_proven"])
        self.assertFalse(receipt["official_train_dev_ood_admitted"])
        self.assertNotIn("rows", receipt)
        self.assertNotIn("coordinates", receipt)
        self.assertNotIn("task", receipt)

    def test_first_turn_abstention_and_second_turn_abstention(self):
        receipt = aggregate([
            row(1, 0, False, "abstain"),
            row(2, 1, False, "abstain"),
        ], GPU)
        self.assertEqual(receipt["real_pretrained_model_generations"], 3)
        self.assertEqual(receipt["native_browser_actions_dispatched"], 1)
        self.assertEqual(receipt["source_abstentions"], 2)
        self.assertEqual(receipt["original_four_station_task_successes"], 0)

    def test_wrong_focus_and_blocked_write_are_not_dispatched(self):
        receipt = aggregate([
            row(1, 1, False, "wrong_focus_or_state"),
            row(2, 1, False, "wrong_code"),
        ], GPU)
        self.assertEqual(receipt["real_pretrained_model_generations"], 3)
        self.assertEqual(receipt["native_browser_actions_dispatched"], 2)
        self.assertEqual(receipt["original_four_station_task_successes"], 0)

    def test_source_state_mismatch_not_success(self):
        receipt = aggregate([row(2, 2, False, "source_state_mismatch"),
                             row()], GPU)
        self.assertEqual(receipt["original_four_station_task_successes"], 1)

    def test_forged_counts_and_boolean_rejected(self):
        cases = [
            row(1, 2, False, "abstain"),
            row(2, 0, False, "wrong_code"),
            row(2, 1, True, "none"),
            row(2, 2, True, "wrong_code"),
            row(True, 0, False, "abstain"),
            row(2, False, False, "wrong_code"),
            row(2, 2, 1, "none"),
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aggregate([row(), bad], GPU)

    def test_reject_extra_private_fields_and_false_attestation(self):
        for key, value in [
            ("model_output", "secret"),
            ("predicted_point", [1, 2]),
            ("pretrained_model_identity_independently_attested", True),
            ("official_train_dev_ood_admitted", True),
        ]:
            bad = row()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate([row(), bad], GPU)
        self.assertEqual(set(row()), ROW_KEYS)

    def test_reject_forged_gpu_or_incomplete_rows(self):
        for hardware in [
            {"free_t4": 1, "t4_count": 1},
            {"free_t4": True, "t4_count": True},
            {"free_t4": True, "t4_count": 0},
            {"free_t4": True, "t4_count": 1, "paid": False},
        ]:
            with self.subTest(hardware=hardware), self.assertRaises(ValueError):
                aggregate([row(), row()], hardware)
        with self.assertRaises(ValueError):
            aggregate([row()], GPU)
        with self.assertRaises(ValueError):
            aggregate([row(), row(), row()], GPU)

    def test_failure_receipt_categorical_no_exception_string(self):
        report = safe_failure("MODEL_LOAD", "OutOfMemoryError")
        self.assertEqual(report["exception_class"], "OutOfMemoryError")
        self.assertFalse(report["verified_score"])
        self.assertNotIn("raw_output", report)
        self.assertEqual(safe_failure("SOURCE", "SensitiveError")[
            "exception_class"], "OTHER")
        with self.assertRaises(ValueError):
            safe_failure("UNREGISTERED", "ValueError")


if __name__ == "__main__":
    unittest.main()
