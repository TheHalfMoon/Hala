"""H295 pure CPU regressions for statistical integrity, not new model scores."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from h295.paired_score_integrity import (
    AnalysisRefused, SCHEMA, analyze, analyze_bytes,
    analyze_aggregate_file, exact_two_sided_mc_nemar,
)


def row(family: str, n: int, base: int, adapted: int, paired=None):
    return {
        "family": family, "sample_count": n,
        "base_success": base, "adapted_success": adapted, "paired": paired,
    }


def known_h292_public_marginals():
    # Published small categorical research totals, not private raw rows.
    return {
        "schema": SCHEMA,
        "source_exposure": "public_previously_exposed",
        "groups": [
            row("queueboard", 2, 1, 1),
            row("routeboard", 6, 5, 5),
        ],
    }


def encoded(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


class H295ExposureAndPairedStatisticsTests(TestCase):
    def test_actual_published_h292_marginals_zero_gain_and_no_ood(self):
        safe = analyze_bytes(encoded(known_h292_public_marginals()))
        self.assertEqual(safe["total_case_count"], 8)
        self.assertEqual(safe["total_base_success"], 6)
        self.assertEqual(safe["total_adapted_success"], 6)
        self.assertEqual(safe["total_descriptive_delta"], 0)
        for result in safe["by_family"]:
            self.assertEqual(result["adapted_minus_base"], 0)
            self.assertTrue(result["exact_p_value_identifiable"])
            self.assertEqual(result["exact_two_sided_p"], 1.0)
            self.assertFalse(result["model_gain_or_generalization_proven"])
        self.assertFalse(safe["combined_p_value_calculated"])
        self.assertFalse(safe["new_independent_blind_ood_proven"])
        self.assertFalse(safe["can_publish_or_release"])

    def test_h291_queueboard_only_marginals(self):
        doc = known_h292_public_marginals()
        doc["groups"] = doc["groups"][:1]
        safe = analyze(doc)
        self.assertEqual(safe["total_case_count"], 2)
        self.assertEqual(safe["total_base_success"], 1)
        self.assertEqual(safe["total_adapted_success"], 1)
        self.assertEqual(safe["by_family"][0]["exact_two_sided_p"], 1.0)

    def test_equal_marginal_p_value_is_identifiable_without_pairing(self):
        for n, base, adapted in ((2, 1, 1), (6, 0, 0), (6, 6, 6), (6, 3, 3)):
            doc = known_h292_public_marginals()
            doc["groups"] = [row("queueboard", 2, 1, 1)] if n == 2 else [
                row("routeboard", 6, base, adapted)]
            safe = analyze(doc)
            self.assertTrue(safe["by_family"][0]["exact_p_value_identifiable"])
            self.assertEqual(safe["by_family"][0]["exact_two_sided_p"], 1.0)

    def test_unequal_marginals_do_not_make_up_pairing_or_p_value(self):
        doc = known_h292_public_marginals()
        doc["groups"][1] = row("routeboard", 6, 2, 4)
        safe = analyze(doc)
        outcome = safe["by_family"][1]
        self.assertEqual(outcome["adapted_minus_base"], 2)
        self.assertEqual(outcome["direction"], "higher")
        self.assertFalse(outcome["exact_p_value_identifiable"])
        self.assertIsNone(outcome["exact_two_sided_p"])
        self.assertFalse(outcome["model_gain_or_generalization_proven"])

    def test_true_exact_two_sided_conditional_mcnemar_extremes(self):
        self.assertEqual(exact_two_sided_mc_nemar(0, 0), 1.0)
        self.assertEqual(exact_two_sided_mc_nemar(1, 0), 1.0)
        self.assertEqual(exact_two_sided_mc_nemar(2, 0), 0.5)
        self.assertEqual(exact_two_sided_mc_nemar(6, 0), 0.03125)
        self.assertEqual(exact_two_sided_mc_nemar(3, 3), 1.0)
        self.assertEqual(exact_two_sided_mc_nemar(5, 1), 0.21875)

    def test_invented_complete_pairing_is_exactly_reconciled(self):
        doc = known_h292_public_marginals()
        doc["groups"][1] = row("routeboard", 6, 2, 4, {
            "both_success": 2, "base_only": 0,
            "adapted_only": 2, "both_fail": 2,
        })
        safe = analyze(doc)
        self.assertEqual(safe["by_family"][1]["exact_two_sided_p"], 0.5)
        self.assertTrue(safe["by_family"][1]["exact_p_value_identifiable"])
        self.assertEqual(safe["total_descriptive_delta"], 2)
        self.assertFalse(safe["statistically_validated_generalization"])

    def test_reject_wrong_marginal_pairing_and_nonint_types(self):
        base = known_h292_public_marginals()
        for wrong_pair in (
            {"both_success": 2, "base_only": 1,
             "adapted_only": 2, "both_fail": 1},
            {"both_success": 1, "base_only": 1,
             "adapted_only": 0, "both_fail": 4},
            {"both_success": True, "base_only": 0,
             "adapted_only": 0, "both_fail": 5},
            {"both_success": 5, "base_only": 0,
             "adapted_only": 0, "both_fail": 2},
        ):
            doc = deepcopy(base)
            doc["groups"][1]["paired"] = wrong_pair
            with self.subTest(pair=wrong_pair), self.assertRaises(AnalysisRefused):
                analyze(doc)
        doc = deepcopy(base)
        doc["groups"][0]["base_success"] = True
        with self.assertRaises(AnalysisRefused):
            analyze(doc)

    def test_reject_blind_relabeling_unknown_sources_and_protocol_drift(self):
        base = known_h292_public_marginals()
        mutations = (
            ("source_exposure", "independent_blind_ood"),
            ("schema", "h293_better"),
        )
        for field, value in mutations:
            d = deepcopy(base)
            d[field] = value
            with self.subTest(field=field), self.assertRaises(AnalysisRefused):
                analyze(d)
        for field, val in (
            ("family", "private_benchmark"),
            ("sample_count", 9),
            ("adapted_success", 7),
        ):
            d = deepcopy(base)
            d["groups"][1][field] = val
            with self.subTest(field=field), self.assertRaises(AnalysisRefused):
                analyze(d)

    def test_reject_raw_unknown_fields_and_duplicate_groups(self):
        d = known_h292_public_marginals()
        d["groups"][0]["raw_model_click"] = [400, 100]
        with self.assertRaises(AnalysisRefused):
            analyze(d)
        d = known_h292_public_marginals()
        d["model_weights"] = "private"
        with self.assertRaises(AnalysisRefused):
            analyze(d)
        d = known_h292_public_marginals()
        d["groups"][1]["family"] = "queueboard"
        with self.assertRaises(AnalysisRefused):
            analyze(d)

    def test_reject_duplicate_json_keys_including_escaped_and_nan(self):
        original = encoded(known_h292_public_marginals())
        mutated = (
            original.replace(b'"schema":',
                             b'"\\u0073chema":"fake","schema":', 1),
            original.replace(b'"sample_count":2',
                             b'"sample_count":NaN', 1),
            b"\xff" * 16,
            b" " * 4097,
            b"not-json",
        )
        for case in mutated:
            with self.subTest(length=len(case)), self.assertRaises(AnalysisRefused):
                analyze_bytes(case)

    def test_exact_aggregate_basename_and_refuse_symlink(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            correct = root / "research.h295-public-score.json"
            correct.write_bytes(encoded(known_h292_public_marginals()))
            self.assertEqual(analyze_aggregate_file(correct)["total_case_count"], 8)
            other = root / "raw-private.json"
            other.write_bytes(correct.read_bytes())
            with self.assertRaises(AnalysisRefused):
                analyze_aggregate_file(other)
            try:
                alias = root / "unsafe.h295-public-score.json"
                alias.symlink_to(correct)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(AnalysisRefused):
                analyze_aggregate_file(alias)
