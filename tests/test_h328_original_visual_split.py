"""H328: synthetic invariant tests plus opt-in real original GUI source split."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from h328.original_family_visual_split import (
    ROLE, EXPECTED_TRAIN_FRAMES, EXPECTED_EXPOSED_FRAMES,
    audit_original_train_exposed_split, compare_family_fingerprints,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "h325" / "railkit.html"
EXPOSED = ROOT / "h319" / "beaconboard.html"


def fictional_digest_series():
    return ([bytes([1, i]) + bytes(30) for i in range(32)],
            [bytes([2, i]) + bytes(30) for i in range(24)])


class H328SyntheticSplitContract(unittest.TestCase):
    def test_exact_32_24_disjoint_returns_no_hash_or_samples(self):
        train, exposed = fictional_digest_series()
        result = compare_family_fingerprints(train, exposed)
        self.assertEqual(result["train_candidate_rgb_unique"], 32)
        self.assertEqual(result["public_exposed_rgb_unique"], 24)
        self.assertEqual(result["cross_family_rgb_identical"], 0)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["dev_admitted"])
        self.assertFalse(result["blind_ood_qualified"])
        self.assertFalse(result["adapted_model_gain_proven"])
        self.assertEqual(result["model_generations"], 0)
        for value in result.values():
            self.assertNotIsInstance(value, bytes)
            self.assertNotIsInstance(value, list)

    def test_duplicate_within_train_refused(self):
        train, exposed = fictional_digest_series()
        train[4] = train[0]
        with self.assertRaises(ValueError):
            compare_family_fingerprints(train, exposed)

    def test_duplicate_within_exposed_refused(self):
        train, exposed = fictional_digest_series()
        exposed[9] = exposed[3]
        with self.assertRaises(ValueError):
            compare_family_fingerprints(train, exposed)

    def test_cross_family_collision_rejected(self):
        train, exposed = fictional_digest_series()
        exposed[7] = train[7]
        with self.assertRaises(ValueError):
            compare_family_fingerprints(train, exposed)

    def test_wrong_counts_or_hash_types_rejected(self):
        train, exposed = fictional_digest_series()
        for left, right in (
            (train[:31], exposed),
            (train, exposed[:23]),
            (tuple(train), exposed),
            (train, exposed + [bytes(32)]),
            ([None] + train[1:], exposed),
            ([b"not-a-sha256"] + train[1:], exposed),
        ):
            with self.subTest(length=(len(left), len(right))):
                with self.assertRaises(ValueError):
                    compare_family_fingerprints(left, right)

    def test_no_dev_ood_or_training_role_escalation(self):
        for role in ("TRAIN", "DEV", "BLIND_OOD", "H199_ADMITTED_TRAIN"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                audit_original_train_exposed_split(
                    None, TRAIN, EXPOSED, role=role)

    def test_source_check_failure_never_calls_browser_or_train_sink(self):
        with patch(
            "h328.original_family_visual_split.checked_train_source",
            side_effect=ValueError("bad source"),
        ), patch(
            "h328.original_family_visual_split.produce_two_step_train_candidates"
        ) as producer:
            with self.assertRaises(ValueError):
                audit_original_train_exposed_split(
                    None, TRAIN, EXPOSED, role=ROLE)
            producer.assert_not_called()

    def test_no_hidden_private_or_model_promotion(self):
        train, exposed = fictional_digest_series()
        result = compare_family_fingerprints(train, exposed)
        self.assertFalse(result["whole_source_and_rights_history_independently_proven"])
        self.assertFalse(result["can_release"])


@unittest.skipUnless(os.environ.get("HALA_H328_BROWSER_QUALIFY") == "1",
                     "full-repo original-source browser opt-in required")
class H328RealSplitBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H328_BROWSER_CHANNEL", "chromium")
            if channel not in {"chromium", "msedge"}:
                raise ValueError("unsupported original browser")
            cls.browser = cls.pw.chromium.launch(
                headless=True,
                **({"channel": "msedge"} if channel == "msedge"
                   else {"args": ["--no-sandbox"]}))
        except Exception:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_source_owned_32_versus_24_rgb_disjoint(self):
        result = audit_original_train_exposed_split(
            self.browser, TRAIN, EXPOSED, role=ROLE)
        self.assertEqual(result["train_candidate_rgb_unique"], 32)
        self.assertEqual(result["public_exposed_rgb_unique"], 24)
        self.assertEqual(result["cross_family_rgb_identical"], 0)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["can_release"])


if __name__ == "__main__":
    unittest.main()
