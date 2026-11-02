"""H364 source-only batch-label padding and fail-closed contracts.

The default tests use only Python stdlib. They are NOT evidence that real
Qwen model forward or actual TRAIN were performed.
"""
from __future__ import annotations

import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h364.native_batch import padded_token_plan, SCHEMA, ROLE


class H364NativeBatchPlanTests(unittest.TestCase):
    @staticmethod
    def samples():
        return [
            {"prefix_ids": [10,11,12], "full_ids": [10,11,12,30,31,32,33,34]},
            {"prefix_ids": [20,21,22,23,24], "full_ids": [
                20,21,22,23,24,40,41,42,43,44,45,46,47]},
        ]

    def test_two_variable_length_rows_have_exact_mask_and_pad(self):
        a, b = self.samples()
        plan = padded_token_plan([a,b], 99)
        self.assertEqual(plan["sequence_lengths"], [8,13])
        self.assertEqual(plan["max_length"], 13)
        self.assertEqual(plan["supervised_token_counts"], [5,8])
        self.assertEqual(plan["input_ids"][0], a["full_ids"] + [99]*5)
        self.assertEqual(plan["input_ids"][1], b["full_ids"])
        self.assertEqual(plan["labels"][0], [-100]*3 + [30,31,32,33,34] + [-100]*5)
        self.assertEqual(plan["labels"][1], [-100]*5 + b["full_ids"][5:])
        self.assertEqual(plan["attention_mask"][0], [1]*8 + [0]*5)
        self.assertEqual(plan["attention_mask"][1], [1]*13)
        self.assertEqual(sum(sum(v != -100 for v in x)
                             for x in plan["labels"]), 13)
        self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)
        self.assertTrue(SCHEMA.startswith("hala.h364."))
        self.assertIn("NOT_ADMITTED", ROLE)

    def test_malformed_prefix_token_or_oversize_batch_refuses_without_models(self):
        a, b = self.samples()
        failures = [
            ([a], 99), ([], 99), ([a,b,a,b,a], 99),
            ([a,b], -1), ([a,b], True),
            ([dict(a, prefix_ids=[10,11,13]), b], 99),
            ([dict(a, full_ids=[10,11,12,30]), b], 99),
            ([dict(a, full_ids=[10,11,12,30,31,32,33,True]), b], 99),
            ([dict(a, full_ids=[10,11,12,30,31,32,33,-1]), b], 99),
            ([dict(a, prefix_ids="not tokens"), b], 99),
            ([dict(a, full_ids="not tokens"), b], 99),
            ([dict(a, prefix_ids=[]), b], 99),
        ]
        for items, pad in failures:
            with self.subTest(items=str(items)[:80], pad=pad):
                with self.assertRaises(ValueError):
                    padded_token_plan(items, pad)

    def test_extra_view_of_second_row_cannot_unmask_original_user_tokens(self):
        a, b = self.samples()
        p = padded_token_plan([b,a], 0)
        self.assertEqual(p["sequence_lengths"], [13,8])
        self.assertEqual(p["attention_mask"][1][-5:], [0]*5)
        self.assertEqual(p["labels"][0][:5], [-100]*5)
        self.assertEqual(p["labels"][1][:3], [-100]*3)
        self.assertEqual(p["labels"][1][-5:], [-100]*5)


if __name__ == "__main__":
    unittest.main()
