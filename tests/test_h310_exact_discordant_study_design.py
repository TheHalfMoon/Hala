"""Invented-only H310 study-design tests. No model/private data."""
import unittest
from fractions import Fraction

from h310.exact_discordant_study_design import (
    ALPHA, TARGET_POWER, DesignRefused, conditional_positive_power,
    exact_two_sided_p, prospective_discordance_grid,
)


class ExactPairedStudyDesignTests(unittest.TestCase):
    def test_exact_two_sided_conditional_mcnemar(self):
        self.assertEqual(exact_two_sided_p(0, 0), Fraction(1))
        self.assertEqual(exact_two_sided_p(6, 0), Fraction(1, 32))
        self.assertEqual(exact_two_sided_p(0, 6), Fraction(1, 32))
        self.assertEqual(exact_two_sided_p(5, 0), Fraction(1, 16))
        self.assertEqual(exact_two_sided_p(3, 3), Fraction(1))

    def test_hypothetical_power_exact_fraction(self):
        self.assertEqual(conditional_positive_power(0, Fraction(3, 4)), 0)
        self.assertEqual(
            conditional_positive_power(8, Fraction(3, 4)),
            Fraction(6561, 65536),
        )

    def test_discrete_power_not_monotonic(self):
        q = Fraction(3, 4)
        self.assertGreaterEqual(conditional_positive_power(30, q), TARGET_POWER)
        self.assertLess(conditional_positive_power(31, q), TARGET_POWER)
        self.assertLess(conditional_positive_power(32, q), TARGET_POWER)

    def test_stable_threshold_is_explicitly_bounded(self):
        result = prospective_discordance_grid()
        self.assertEqual(result["first_discordant_count_meeting_target"], 30)
        self.assertEqual(result["all_counts_from_here_meet_target_within_search_range"], 35)
        self.assertEqual(result["maximum_discordant_pairs_searched"], 128)
        self.assertEqual(result["power_at_eight_discordant_pairs"], "6561/65536")
        self.assertIsNone(result["number_of_gui_tasks_required"])
        self.assertFalse(result["discordance_rate_known"])
        self.assertFalse(result["source_family_independence_verified"])
        self.assertFalse(result["private_holdout_constructed"])
        self.assertFalse(result["observed_model_gain"])
        self.assertFalse(result["can_publish_or_release"])

    def test_short_grid_makes_no_threshold_up(self):
        result = prospective_discordance_grid(maximum=8)
        self.assertIsNone(result["first_discordant_count_meeting_target"])
        self.assertIsNone(result["all_counts_from_here_meet_target_within_search_range"])

    def test_rejects_fake_counts_and_invalid_probabilities(self):
        calls = [
            lambda: exact_two_sided_p(True, 0),
            lambda: exact_two_sided_p(-1, 2),
            lambda: exact_two_sided_p(129, 0),
            lambda: conditional_positive_power(True, Fraction(3, 4)),
            lambda: conditional_positive_power(8, 0.75),
            lambda: conditional_positive_power(8, Fraction(1, 2)),
            lambda: conditional_positive_power(8, Fraction(1)),
            lambda: conditional_positive_power(8, Fraction(3, 4), Fraction(0)),
            lambda: conditional_positive_power(8, Fraction(3, 4), Fraction(1)),
            lambda: prospective_discordance_grid(maximum=True),
            lambda: prospective_discordance_grid(maximum=129),
            lambda: prospective_discordance_grid(target_power=0.8),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                DesignRefused, "^H310_INVALID_PUBLIC_PLANNING_PARAMETER$"
            ):
                call()

    def test_no_private_or_release_fields_accepted(self):
        with self.assertRaises( TypeError ):
            prospective_discordance_grid(private_receipt="forbidden")
        with self.assertRaises( TypeError ):
            prospective_discordance_grid(official_train_admitted=True)

    def test_fractional_alpha_is_exact_not_float(self):
        self.assertEqual(ALPHA, Fraction(1, 20))
        self.assertEqual(TARGET_POWER, Fraction(4, 5))
        self.assertEqual(exact_two_sided_p(6, 0) <= ALPHA, True)
        self.assertEqual(exact_two_sided_p(5, 0) <= ALPHA, False)


if __name__ == "__main__":
    unittest.main()
