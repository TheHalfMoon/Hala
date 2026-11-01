"""H344 source-only cross-family aggregate contract tests.

All fingerprints here are invented, not screenshots or Qwen output.
"""
from hashlib import sha256
import json
import unittest

from h344.original_cross_family_visual_proximity import (
    BANDS, COUNTS, PAIRS, ROLE, cross_family_counts,
    source_only_public_band_receipt,
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

    def test_frozen_source_only_receipt_cannot_expose_private_fields(self):
        result = cross_family_counts(fictional_fingerprints())
        safe = source_only_public_band_receipt(result)
        self.assertEqual(safe["public_source_frames"], 312)
        self.assertEqual(safe["cross_family_compared_pairs"], 15104)
        self.assertTrue(safe["not_rights_or_semantic_ood_or_model_gain"])
        self.assertFalse(safe["train_admitted"])
        self.assertEqual(
            sum(sum(v.values()) for v in
                safe["per_family_pair_bins"].values()), 15104)
        self.assertNotIn("model_output", json.dumps(safe))
        self.assertNotIn("source_gold", json.dumps(safe))
        for injection in (
            {"raw_model_point": [1, 2]},
            {"pixel_data": "PRIVATE"},
            {"generated_model_gain": True},
        ):
            with self.subTest(injection=injection), self.assertRaises(
                ValueError
            ):
                source_only_public_band_receipt(dict(result, **injection))
        corrupted = dict(result)
        corrupted["training_admitted"] = True
        with self.assertRaises(ValueError):
            source_only_public_band_receipt(corrupted)
        corrupted = dict(result)
        corrupted["cross_family_lossy_perceptual_zero_pairs"] = 15104
        with self.assertRaises(ValueError):
            source_only_public_band_receipt(corrupted)

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


@unittest.skipUnless(
    __import__("os").environ.get("HALA_H344_ORIGINAL_BROWSER") == "1",
    "Full source-only browser qualification is explicit and opt-in",
)
class H344OriginalThreeFamilyBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        import os
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H344_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("H344 browser not allowlisted")
            cls.browser = cls.pw.chromium.launch(
                headless=True, **(
                    {"channel": "msedge"} if channel == "msedge"
                    else {"args": ["--no-sandbox"]}
                ))
        except Exception:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_all_312_true_source_frames_and_15104_pairs(self):
        from pathlib import Path
        from h344.original_cross_family_visual_proximity import (
            ROLE, audit_original_three_family_images,
        )
        root = Path(__file__).resolve().parents[1]
        result = audit_original_three_family_images(
            self.browser,
            root / "h325" / "railkit.html",
            root / "h338" / "signaldesk.html",
            root / "h319" / "beaconboard.html",
            role=ROLE,
        )
        self.assertEqual(result["total_source_frames"], 312)
        self.assertEqual(result["cross_family_frame_pairs_tested"], 15104)
        self.assertEqual(result["cross_family_rgb_identical"], 0)
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["independent_dev_or_blind_ood"])
        self.assertEqual(result["actual_pretrained_or_adapted_generations"], 0)
        safe = source_only_public_band_receipt(result)
        self.assertEqual(safe["cross_family_compared_pairs"], 15104)
        self.assertFalse(safe["blind_ood"])
        # Public aggregate-only evidence, never model actions/images/points.
        print("H344_ACTUAL_PUBLIC_SOURCE_BANDS="
              + json.dumps(safe, sort_keys=True, separators=(",", ":")),
              flush=True)


if __name__ == "__main__":
    unittest.main()
