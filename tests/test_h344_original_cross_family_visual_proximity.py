"""H344 source-only cross-family aggregate contract tests.

All fingerprints here are invented, not screenshots or Qwen output.
"""
from hashlib import sha256
import unittest

from h344.original_cross_family_visual_proximity import (
    BANDS, COUNTS, PAIRS, ROLE, cross_family_counts,
)


def fictional_fingerprints():
    output = {}
    index = 0
    for name, total in COUNTS.items():
        result = []
        for _ in range(total):
            result.append((
                (index * 137 + 17) % (1 << 1024),
                sha256(str(index).encode("ascii")).digest(),
            ))
            index += 1
        output[name] = tuple(result)
    return output


class H344InventedOnlyContracts(unittest.TestCase):
    def test_expected_three_family_counts(self):
        self.assertEqual(COUNTS, {
            "RailKit": 32, "SignalDesk": 256, "BeaconBoard": 24,
        })
        self.assertEqual(sum(COUNTS.values()), 312)
        self.assertEqual(sum(COUNTS[a] * COUNTS[b] for a, b in PAIRS),
                         15104)

    def test_bounded_aggregates_do_not_promote_train_or_ood(self):
        summary = cross_family_counts(fictional_fingerprints())
        self.assertEqual(summary["status"], ROLE)
        self.assertEqual(summary["total_source_frames"], 312)
        self.assertEqual(summary["cross_family_frame_pairs_tested"], 15104)
        self.assertEqual(summary["cross_family_rgb_identical"], 0)
        self.assertEqual(
            sum(sum(item["perceptual_distance_histogram"].values())
                for item in summary["per_family_pair"].values()),
            15104)
        self.assertTrue(summary[
            "perceptual_distance_is_lossy_not_semantic_or_rights_evidence"])
        for key in ("training_admitted", "independent_dev_or_blind_ood",
                    "measured_model_gain", "can_release"):
            self.assertIs(summary[key], False)

    def test_duplicate_rgb_cross_family_rejected(self):
        sample = fictional_fingerprints()
        changed = list(sample["SignalDesk"])
        changed[0] = (changed[0][0], sample["RailKit"][0][1])
        sample["SignalDesk"] = tuple(changed)
        with self.assertRaises(ValueError):
            cross_family_counts(sample)

    def test_malformed_and_additional_families_rejected(self):
        for bad in (True, -1, 1 << 1024, 0.5):
            sample = fictional_fingerprints()
            vals = list(sample["RailKit"])
            vals[0] = (bad, vals[0][1])
            sample["RailKit"] = tuple(vals)
            with self.assertRaises(ValueError):
                cross_family_counts(sample)
        sample = fictional_fingerprints()
        sample["unknown"] = tuple()
        with self.assertRaises(ValueError):
            cross_family_counts(sample)
        sample = fictional_fingerprints()
        sample["BeaconBoard"] = sample["BeaconBoard"][:-1]
        with self.assertRaises(ValueError):
            cross_family_counts(sample)

    def test_perceptual_zero_does_not_imply_rgb_identical(self):
        sample = fictional_fingerprints()
        row = list(sample["BeaconBoard"])
        row[0] = (sample["RailKit"][0][0], row[0][1])
        sample["BeaconBoard"] = tuple(row)
        summary = cross_family_counts(sample)
        self.assertGreaterEqual(
            summary["cross_family_lossy_perceptual_zero_pairs"], 1)
        self.assertEqual(summary["cross_family_rgb_identical"], 0)


if __name__ == "__main__":
    unittest.main()
