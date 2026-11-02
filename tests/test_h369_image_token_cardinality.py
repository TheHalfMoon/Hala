"""H369 stdlib tests: invented token IDs only, no weights or media."""
from __future__ import annotations

import unittest

from h369.image_token_contract import (
    PINNED_IMAGE_TOKEN_ID as IMAGE,
    image_token_cardinality,
)


class H369ImageTokenParityTests(unittest.TestCase):
    @staticmethod
    def valid(patches=8):
        prefix = [101, 202] + [IMAGE] * (patches // 4) + [303]
        full = prefix + [901, 902, 903, 904, 905]
        return prefix, full, patches

    def test_valid_one_image_exact_cardinality(self):
        self.assertEqual(image_token_cardinality(*self.valid()), 2)
        self.assertEqual(image_token_cardinality(*self.valid(2640)), 660)

    def test_missing_image_token_from_consistent_prefix_full_rejected(self):
        prefix, full, patches = self.valid()
        # Under H368, this remains a matching prefix/full, with the
        # geometry unchanged; the old collator would have accepted.
        replacement = [x if x != IMAGE else 111 for x in prefix]
        changed_full = replacement + full[len(prefix):]
        with self.assertRaises(ValueError):
            image_token_cardinality(replacement, changed_full, patches)

    def test_extra_image_token_in_consistent_prefix_full_rejected(self):
        prefix, full, patches = self.valid()
        changed_prefix = [IMAGE] + prefix
        with self.assertRaises(ValueError):
            image_token_cardinality(
                changed_prefix, changed_prefix + full[len(prefix):], patches)

    def test_vision_token_in_assistant_completion_rejected(self):
        prefix, full, patches = self.valid()
        with self.assertRaises(ValueError):
            image_token_cardinality(prefix, full + [IMAGE], patches)

    def test_mismatched_prefix_rejected(self):
        prefix, full, patches = self.valid()
        with self.assertRaises(ValueError):
            image_token_cardinality([999] + prefix[1:], full, patches)

    def test_nonmergeable_patch_count_rejected(self):
        prefix, full, _ = self.valid()
        with self.assertRaises(ValueError):
            image_token_cardinality(prefix, full, 7)

    def test_boolean_and_noninteger_ids_rejected(self):
        prefix, full, patches = self.valid()
        for args in (
            (prefix, full, True),
            (prefix + [True], full, patches),
            (prefix, full + [1.5], patches),
            (tuple(prefix), full, patches),
        ):
            with self.subTest(args=str(args)[:70]):
                with self.assertRaises(ValueError):
                    image_token_cardinality(*args)


if __name__ == "__main__":
    unittest.main()
