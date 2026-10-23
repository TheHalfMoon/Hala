"""Tests for H63: visual sensitivity needs matched reruns, not box shuffling."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from h63.visual_intervention import (
    INCONCLUSIVE, NO_CHANGE, SUFFICIENT, TOO_SMALL, UNSTABLE,
    evaluate_pairs, load_jsonl,
)


def pair(index):
    # Same instruction/model/prompt on real inference arms (mock receipts).
    common = {"model_sha256": "c" * 64, "prompt_sha256": "d" * 64,
              "valid_output": True}
    ref_box = [10 + index * 10, 20, 18 + index * 10, 30]
    swap_box = [410 + index * 10, 20, 418 + index * 10, 30]
    return {
        "case_id": f"case-{index}",
        "reference": {**common, "image_sha256": f"{index+1:064x}",
                      "target_box": ref_box, "point": [ref_box[0] + 3, 25]},
        "repeat": {**common, "image_sha256": f"{index+1:064x}",
                   "target_box": ref_box, "point": [ref_box[0] + 3, 25]},
        "intervention": {**common, "image_sha256": f"{index+1001:064x}",
                         "target_box": swap_box, "point": [swap_box[0] + 3, 25]},
    }


class PairedInterventionTests(unittest.TestCase):
    def test_separate_matched_inferences_show_measured_sensitivity(self):
        report = evaluate_pairs([pair(i) for i in range(8)])
        self.assertEqual(report["verdict"], SUFFICIENT)
        self.assertEqual(report["paired_successes"], 8)
        self.assertEqual(report["repeat_stable_pairs"], 8)

    def test_constant_coordinate_no_longer_passes_box_shuffle_as_visual_evidence(self):
        pairs = [pair(i) for i in range(8)]
        for item in pairs:
            for arm in ("reference", "repeat", "intervention"):
                item[arm]["point"] = [300, 100]
        report = evaluate_pairs(pairs)
        self.assertEqual(report["verdict"], NO_CHANGE)
        self.assertEqual(report["paired_successes"], 0)

    def test_prompt_only_prediction_that_ignores_swapped_pixels_is_not_credited(self):
        pairs = [pair(i) for i in range(8)]
        for item in pairs:
            item["intervention"]["point"] = list(item["reference"]["point"])
        report = evaluate_pairs(pairs)
        self.assertEqual(report["verdict"], NO_CHANGE)
        self.assertEqual(report["reference_hits"], 8)
        self.assertEqual(report["intervention_hits"], 0)

    def test_unstable_same_pixel_reruns_prevent_positive_result(self):
        pairs = [pair(i) for i in range(8)]
        pairs[0]["repeat"]["point"] = [999, 999]
        self.assertEqual(evaluate_pairs(pairs)["verdict"], UNSTABLE)

    def test_too_few_pairs_never_prove_sensitivity(self):
        self.assertEqual(evaluate_pairs([pair(i) for i in range(7)])["verdict"], TOO_SMALL)

    def test_undecodable_outputs_count_as_failures_not_missing_rows(self):
        pairs = [pair(i) for i in range(8)]
        pairs[0]["intervention"]["point"] = None
        pairs[0]["intervention"]["valid_output"] = False
        report = evaluate_pairs(pairs)
        self.assertEqual(report["pairs"], 8)
        self.assertEqual(report["intervention_hits"], 7)
        self.assertEqual(report["paired_successes"], 7)
        self.assertEqual(report["verdict"], SUFFICIENT)

    def test_model_and_prompt_changes_are_refused(self):
        for field in ("model_sha256", "prompt_sha256"):
            pairs = [pair(0)]
            pairs[0]["intervention"][field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_pairs(pairs)

    def test_incorrect_image_pairing_is_refused(self):
        pairs = [pair(0)]
        pairs[0]["repeat"]["image_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            evaluate_pairs(pairs)
        pairs = [pair(0)]
        pairs[0]["intervention"]["image_sha256"] = pairs[0]["reference"]["image_sha256"]
        with self.assertRaises(ValueError):
            evaluate_pairs(pairs)

    def test_overlapping_target_boxes_are_refused(self):
        pairs = [pair(0)]
        pairs[0]["intervention"]["target_box"] = pairs[0]["reference"]["target_box"]
        with self.assertRaises(ValueError):
            evaluate_pairs(pairs)

    def test_duplicate_cases_and_malformed_digests_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluate_pairs([pair(0), pair(0)])
        pairs = [pair(0)]
        pairs[0]["reference"]["image_sha256"] = "unsigned"
        with self.assertRaises(ValueError):
            evaluate_pairs(pairs)

    def test_explicit_partial_evidence_is_inconclusive(self):
        pairs = [pair(i) for i in range(8)]
        for row in pairs[:3]:
            row["intervention"]["point"] = list(row["reference"]["point"])
        self.assertEqual(evaluate_pairs(pairs)["verdict"], INCONCLUSIVE)

    def test_jsonl_reader_preserves_case_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "paired.jsonl"
            path.write_text("\n".join(json.dumps(pair(i)) for i in range(8)) + "\n",
                            encoding="utf8")
            self.assertEqual(len(load_jsonl(path)), 8)

    def test_receipt_claim_is_explicitly_limited(self):
        report = evaluate_pairs([pair(i) for i in range(8)])
        self.assertIn("externally verified", report["claim_limit"])
        self.assertNotIn("VISION_PROVED", report["verdict"])


if __name__ == "__main__":
    unittest.main()
