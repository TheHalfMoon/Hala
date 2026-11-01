"""Invented-only H341 exact power mathematics, never private model results."""
from fractions import Fraction as F
import unittest
from h341.familywise_design_sensitivity import (
    ALPHA, design_sensitivity, minimum_all_win_families_for_two_language_holm,
    positive_raw_sign_rejection_probability,
)


def as_fraction(value):
    return F(value["numerator"], value["denominator"])


class H341PowerTests(unittest.TestCase):
    def test_minimum_all_wins_for_two_language_holm(self):
        self.assertEqual(minimum_all_win_families_for_two_language_holm(), 7)

    def test_six_perfect_families_cannot_pass_both_holm(self):
        report = design_sensitivity(6, F(1), F(1), F(0))
        self.assertEqual(as_fraction(
            report["unknown_language_correlation_joint_holm_power_upper"]), 0)

    def test_seven_perfect_families_pass_in_idealized_model(self):
        report = design_sensitivity(7, F(1), F(1), F(0))
        self.assertEqual(as_fraction(
            report["unknown_language_correlation_joint_holm_power_lower"]), 1)
        self.assertEqual(as_fraction(
            report["unknown_language_correlation_joint_holm_power_upper"]), 1)

    def test_all_ties_zero_power(self):
        report = design_sensitivity(20, F(1), F(1), F(1))
        self.assertEqual(as_fraction(
            report["unknown_language_correlation_joint_holm_power_upper"]), 0)

    def test_null_positive_tail_not_inflated(self):
        value = positive_raw_sign_rejection_probability(
            16, F(1, 2), F(0), ALPHA / 2)
        self.assertLessEqual(value, ALPHA / 2)

    def test_bounds_ordered_without_language_independence(self):
        report = design_sensitivity(18, F(4, 5), F(3, 4), F(1, 10))
        lower = as_fraction(
            report["unknown_language_correlation_joint_holm_power_lower"])
        upper = as_fraction(
            report["unknown_language_correlation_joint_holm_power_upper"])
        self.assertTrue(0 <= lower <= upper <= 1)

    def test_scientific_claims_remain_false(self):
        report = design_sensitivity(32, F(1), F(1), F(0))
        for key in ("source_family_independence_attested",
                    "legal_training_rights_approved",
                    "blind_ood_custody_qualified",
                    "genuine_base_adapted_model_gain_proven",
                    "can_train_or_release"):
            self.assertIs(report[key], False)
        self.assertEqual(report["independently_approved_train_families"], 0)
        self.assertEqual(report["current_public_source_train_candidate_families"], 2)

    def test_refuse_float_boolean_and_out_of_range(self):
        invalid = [
            (True, F(1), F(1), F(0)),
            (0, F(1), F(1), F(0)),
            (65, F(1), F(1), F(0)),
            (7, 0.8, F(1), F(0)),
            (7, F(1), F(5, 4), F(0)),
            (7, F(1), F(1), F(-1, 4)),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                design_sensitivity(*args)

    def test_refuse_unregistered_threshold(self):
        with self.assertRaises(ValueError):
            positive_raw_sign_rejection_probability(7, F(1), F(0), F(1, 10))


if __name__ == "__main__":
    unittest.main()
