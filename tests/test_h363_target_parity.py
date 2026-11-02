"""H363 adversarial task-blind audit on original authored H357 source candidates.

These are deterministic source-code tests, NOT a Qwen inference result,
a legal training admission, or an independent blind OOD evaluation.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h357.source_candidate import COLORS, SHAPES, candidate_specs


def parity_filter_expected_wins(rows):
    """Uniformly guess among visually parity-even tiles, else among all four.

    No instruction, oracle, DOM, target labels or case IDs are supplied to
    the attack at prediction time. Labels are used only for scoring.
    """
    expected = Fraction(0)
    for row in rows:
        eligible = [
            slot for slot, (shape, color) in enumerate(row["source_tiles"])
            if (slot + SHAPES.index(shape) + COLORS.index(color)) % 2 == 0
        ]
        if not eligible:
            expected += Fraction(1, 4)
        elif row["oracle_tile"] in eligible:
            expected += Fraction(1, len(eligible))
    return expected


class H363TargetParityAudit(unittest.TestCase):
    def test_pairwise_balance_and_both_target_parities_per_language(self):
        rows = candidate_specs()
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({
            tuple(map(tuple, r["source_tiles"])) for r in rows
        }), 64)
        for language in ("en", "ar"):
            arm = [r for r in rows if r["language"] == language]
            self.assertEqual(len(arm), 32)
            triples = [
                (r["oracle_tile"],
                 SHAPES.index(r["source_tiles"][r["oracle_tile"]][0]),
                 COLORS.index(r["source_tiles"][r["oracle_tile"]][1]))
                for r in arm
            ]
            for axes in ((0, 1), (0, 2), (1, 2)):
                tally = Counter(tuple(t[i] for i in axes) for t in triples)
                self.assertEqual(len(tally), 16)
                self.assertEqual(sorted(tally.values()), [2] * 16)
            self.assertEqual(
                Counter(sum(t) % 2 for t in triples), {0: 16, 1: 16}
            )

    def test_invented_only_parity_attacker_and_legacy_negative_control(self):
        rows = candidate_specs()
        expected = parity_filter_expected_wins(rows)
        # 15.75 expected correct out of 64, versus 16 for blind 1/4.
        self.assertEqual(expected, Fraction(63, 4))
        self.assertLessEqual(expected, Fraction(len(rows), 4))
        # The superseded source formula made ALL targets parity-even.
        # This is a counterfactual arithmetic regression fixture, not media.
        old_parity = Counter()
        for r in rows:
            pair = int(r["case_id"].rsplit("-", 1)[1]) // 2
            slot = r["oracle_tile"]
            shape_index = (pair // 4) % 4
            legacy_color_index = (slot + shape_index + 2 * (pair // 16)) % 4
            old_parity[(slot + shape_index + legacy_color_index) % 2] += 1
        self.assertEqual(old_parity, {0: 64})

    def test_never_promotes_candidate_to_train_or_ood(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        self.assertEqual(
            {r["split"] for r in candidate_specs()},
            {"TRAIN_DEV_CANDIDATE_NOT_ADMITTED"},
        )


if __name__ == "__main__":
    unittest.main()
