"""H340 invented-only CPU tests; no Qwen, OOD, private media or real results."""
from __future__ import annotations

from copy import deepcopy
import unittest

from h340.familywise_paired_inference import (
    ROLE, _two_sided_family_sign_p, analyze_paired_families,
)


def row(*, both=0, base_only=0, adapted_only=1, neither=0,
        base_abstain=0, adapted_abstain=0,
        base_unsafe=0, adapted_unsafe=0):
    return {
        "both": both, "base_only": base_only,
        "adapted_only": adapted_only, "neither": neither,
        "base_abstain": base_abstain, "adapted_abstain": adapted_abstain,
        "base_unsafe": base_unsafe, "adapted_unsafe": adapted_unsafe,
    }


def invented_families(n, *, en=None, ar=None):
    return [{"en": deepcopy(en if en is not None else row()),
             "ar": deepcopy(ar if ar is not None else row())}
            for _ in range(n)]


class H340FamilyPairedStatistics(unittest.TestCase):
    def test_exact_two_sided_sign_tails(self):
        self.assertEqual(_two_sided_family_sign_p(0, 0), 1)
        self.assertEqual(_two_sided_family_sign_p(6, 0), 1/32)
        self.assertEqual(_two_sided_family_sign_p(7, 0), 1/64)
        self.assertEqual(_two_sided_family_sign_p(3, 3), 1)
        self.assertEqual(_two_sided_family_sign_p(7, 1), 9/128)

    def test_six_families_per_language_cannot_pass_holm(self):
        r = analyze_paired_families(invented_families(6))
        self.assertEqual(r["status"], ROLE)
        self.assertFalse(r["all_language_positive_direction_and_holm"])
        for lang in ("en", "ar"):
            s = r["language_endpoints"][lang]
            self.assertEqual(s["exact_two_sided_family_sign_p"],
                             {"numerator": 1, "denominator": 32})
            self.assertEqual(s["holm_adjusted_p"],
                             {"numerator": 1, "denominator": 16})
            self.assertFalse(s["positive_direction_and_holm"])

    def test_seven_families_mathematically_significant_but_not_qualified(self):
        r = analyze_paired_families(invented_families(7))
        self.assertTrue(r["all_language_positive_direction_and_holm"])
        self.assertEqual(r["language_endpoints"]["ar"]["holm_adjusted_p"],
                         {"numerator": 1, "denominator": 32})
        self.assertEqual(r["language_endpoints"]["en"]["paired_scenes"], 7)
        self.assertEqual(r["language_endpoints"]["en"]["base_successes"], 0)
        self.assertEqual(r["language_endpoints"]["en"]["adapted_successes"], 7)
        for key in ("independent_family_custody_verified",
                    "rights_and_train_admission_verified",
                    "new_never_exposed_blind_ood_verified",
                    "genuine_pinned_base_and_adapted_models_verified",
                    "safety_and_abstention_qualified",
                    "learned_model_gain_proven", "can_release"):
            self.assertIs(r[key], False)

    def test_statistically_significant_worsening_is_not_improvement(self):
        r = analyze_paired_families(invented_families(
            7, ar=row(base_only=1, adapted_only=0)))
        self.assertTrue(r["language_endpoints"]["ar"]["holm_at_or_below_0_05"])
        self.assertFalse(r["language_endpoints"]["ar"]["positive_direction_and_holm"])
        self.assertFalse(r["all_language_positive_direction_and_holm"])

    def test_one_language_ties_do_not_promote_bilingual_gain(self):
        r = analyze_paired_families(invented_families(
            7, ar=row(both=1, adapted_only=0)))
        self.assertEqual(r["language_endpoints"]["ar"]["family_ties"], 7)
        self.assertEqual(r["language_endpoints"]["ar"]["exact_two_sided_family_sign_p"],
                         {"numerator": 1, "denominator": 1})
        self.assertFalse(r["all_language_positive_direction_and_holm"])

    def test_family_units_not_scene_level_pseudoreplication(self):
        r = analyze_paired_families(invented_families(
            2, en=row(adapted_only=200), ar=row(adapted_only=200)))
        self.assertEqual(r["language_endpoints"]["en"]["paired_scenes"], 400)
        self.assertEqual(r["language_endpoints"]["en"]["family_wins"], 2)
        self.assertEqual(r["language_endpoints"]["en"]["exact_two_sided_family_sign_p"],
                         {"numerator": 1, "denominator": 2})
        self.assertFalse(r["all_language_positive_direction_and_holm"])

    def test_abstention_unsafe_counters_reported_without_promotion(self):
        families = invented_families(2)
        families[0]["en"] = row(both=1, neither=1, base_abstain=1,
                                 adapted_abstain=1, base_unsafe=1,
                                 adapted_unsafe=1)
        r = analyze_paired_families(families)
        self.assertEqual(r["language_endpoints"]["en"]["adapted_unsafe"], 1)
        self.assertEqual(r["language_endpoints"]["en"]["adapted_abstentions"], 1)
        self.assertFalse(r["safety_and_abstention_qualified"])

    def test_reject_raw_or_private_fields_and_false_attestations(self):
        for field, value in (
            ("predicted_point", [10, 20]), ("source_family_id", "secret"),
            ("blind_ood_qualified", True), ("model_output", "private"),
            ("training_gold", "secret"), ("rights_granted", True),
        ):
            families = invented_families(2)
            families[0]["ar"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_paired_families(families)

    def test_reject_malformed_counts_and_forged_type(self):
        mutations = (
            ("both", True), ("both", -1), ("both", 257),
            ("adapted_only", 0.5), ("adapted_abstain", 1),
            ("base_abstain", 2), ("base_unsafe", 300),
        )
        for field, value in mutations:
            families = invented_families(2)
            families[0]["en"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_paired_families(families)
        for families in ([], [{}], [None, None], tuple(invented_families(2)),
                         invented_families(65)):
            with self.subTest(size=len(families)), self.assertRaises(ValueError):
                analyze_paired_families(families)

    def test_reject_zero_pair_family_and_malformed_language(self):
        families = invented_families(2)
        families[0]["en"] = row(adapted_only=0)
        with self.assertRaises(ValueError):
            analyze_paired_families(families)
        families = invented_families(2)
        families[0]["Arabic"] = families[0].pop("ar")
        with self.assertRaises(ValueError):
            analyze_paired_families(families)

    def test_pooled_effect_fraction_and_input_not_mutated(self):
        families = invented_families(2)
        families[0]["en"] = row(both=2, base_only=1, adapted_only=2,
                                 neither=1)
        families[1]["en"] = row(both=1, base_only=0, adapted_only=2)
        before = deepcopy(families)
        r = analyze_paired_families(families)
        self.assertEqual(families, before)
        self.assertEqual(r["language_endpoints"]["en"]["paired_scenes"], 9)
        self.assertEqual(r["language_endpoints"]["en"]["base_successes"], 4)
        self.assertEqual(r["language_endpoints"]["en"]["adapted_successes"], 7)
        self.assertEqual(r["language_endpoints"]["en"]["adapted_minus_base_success_rate"],
                         {"numerator": 1, "denominator": 3})


if __name__ == "__main__":
    unittest.main()
