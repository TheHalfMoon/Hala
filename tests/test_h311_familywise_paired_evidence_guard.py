"""Invented-only H311 familywise exact-inference safety regression tests."""
import unittest
from fractions import Fraction

from h311.familywise_paired_evidence_guard import (
    FamilywiseRefused, hypothetical_familywise_gate,
)


class FamilywiseGuardTests(unittest.TestCase):
    def test_two_strong_invented_families_holm(self):
        out = hypothetical_familywise_gate(
            (("family_1", 0, 8), ("family_2", 0, 7)),
            expected_family_count=2,
        )
        self.assertEqual(
            [r["two_sided_exact_p"] for r in out["by_hypothetical_family"]],
            ["1/128", "1/64"],
        )
        self.assertEqual(
            [r["holm_adjusted_p"] for r in out["by_hypothetical_family"]],
            ["1/64", "1/64"],
        )
        self.assertTrue(out["all_families_positive_and_holm_significant_hypothetically"])
        self.assertFalse(out["real_base_vs_adapted_gain_proven"])
        self.assertFalse(out["can_release"])

    def test_one_weak_family_blocks_all_families_claim(self):
        out = hypothetical_familywise_gate(
            (("family_1", 0, 5), ("family_2", 0, 8)),
            expected_family_count=2,
        )
        self.assertEqual(out["by_hypothetical_family"][0]["holm_adjusted_p"], "1/16")
        self.assertEqual(out["by_hypothetical_family"][1]["holm_adjusted_p"], "1/64")
        self.assertFalse(out["all_families_positive_and_holm_significant_hypothetically"])

    def test_significant_negative_direction_is_not_gain(self):
        out = hypothetical_familywise_gate(
            (("family_1", 8, 0), ("family_2", 0, 8)),
            expected_family_count=2,
        )
        self.assertFalse(out["by_hypothetical_family"][0]["positive_direction"])
        self.assertFalse(out["all_families_positive_and_holm_significant_hypothetically"])

    def test_no_discordance_not_significant(self):
        out = hypothetical_familywise_gate(
            (("family_1", 0, 0), ("family_2", 0, 8)),
            expected_family_count=2,
        )
        self.assertEqual(out["by_hypothetical_family"][0]["two_sided_exact_p"], "1")
        self.assertFalse(out["all_families_positive_and_holm_significant_hypothetically"])

    def test_three_family_ties_deterministic_and_monotone(self):
        out = hypothetical_familywise_gate(
            (("family_1", 0, 8), ("family_2", 0, 8), ("family_3", 0, 8)),
            expected_family_count=3,
        )
        self.assertEqual(
            [r["holm_adjusted_p"] for r in out["by_hypothetical_family"]],
            ["3/128", "3/128", "3/128"],
        )

    def test_reject_missing_or_reordered_families(self):
        for families, count in (
            ((("family_1", 0, 8),), 2),
            ((("family_2", 0, 8), ("family_1", 0, 8)), 2),
            ((("family_1", 0, 8), ("family_1", 0, 8)), 2),
            ((("family_1", 0, 8), ("family_2", 0, 8)), 3),
        ):
            with self.subTest(families=families):
                with self.assertRaises(FamilywiseRefused):
                    hypothetical_familywise_gate(families, expected_family_count=count)

    def test_reject_forged_counts_and_extra_fields(self):
        bad = (
            (("family_1", True, 8), ("family_2", 0, 8)),
            (("family_1", -1, 8), ("family_2", 0, 8)),
            (("family_1", 0, 129), ("family_2", 0, 8)),
            (("family_1", 0, 8, "raw"), ("family_2", 0, 8)),
        )
        for families in bad:
            with self.subTest(families=families):
                with self.assertRaises(FamilywiseRefused):
                    hypothetical_familywise_gate(families, expected_family_count=2)

    def test_reject_bad_alpha_and_count(self):
        good = (("family_1", 0, 8), ("family_2", 0, 8))
        for alpha in (True, 0.05, Fraction(0), Fraction(1)):
            with self.subTest(alpha=alpha):
                with self.assertRaises(FamilywiseRefused):
                    hypothetical_familywise_gate(good, expected_family_count=2, alpha=alpha)
        with self.assertRaises(FamilywiseRefused):
            hypothetical_familywise_gate(good, expected_family_count=True)

    def test_never_promote_hypothetical_to_real_evidence(self):
        out = hypothetical_familywise_gate(
            (("family_1", 0, 8), ("family_2", 0, 8)),
            expected_family_count=2,
        )
        for key in (
            "preregistration_independently_authenticated",
            "independent_within_family_pairs_verified",
            "independent_custodian_or_blind_ood_verified",
            "source_rights_or_train_dev_admission_verified",
            "real_base_vs_adapted_gain_proven",
            "can_release",
        ):
            self.assertIs(out[key], False)


if __name__ == "__main__":
    unittest.main()
