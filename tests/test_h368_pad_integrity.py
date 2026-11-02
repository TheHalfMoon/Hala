"""H368: independent right-PAD token integrity regression (stdlib only).

Invented token IDs only. This is not model training, evaluation, or data admission.
"""
from __future__ import annotations

import unittest

from h367.four_row_processor import verify_token_row_integrity


class H368PaddingIntegrityTests(unittest.TestCase):
    @staticmethod
    def rows():
        prefix = [11, 12, 13]
        full = prefix + [21, 22, 23, 24, 25]
        pad_id = 0
        inputs = full + [pad_id] * 3
        labels = [-100] * len(prefix) + full[len(prefix):] + [-100] * 3
        attention = [1] * len(full) + [0] * 3
        return inputs, labels, attention, full, prefix, pad_id

    def test_genuine_right_padding_accepted(self):
        verify_token_row_integrity(*self.rows())

    def test_unpadded_equal_length_accepted(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        verify_token_row_integrity(full, labels[:len(full)],
                                   attention[:len(full)], full, prefix, pad_id)

    def test_masked_nonpad_input_is_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        inputs[-1] = 999  # prior verifier accepted despite label=-100, attention=0
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention,
                                       full, prefix, pad_id)

    def test_pad_token_mismatch_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention,
                                       full, prefix, pad_id + 1)

    def test_unmasked_pad_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        attention[-1] = 1
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention,
                                       full, prefix, pad_id)

    def test_supervised_pad_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        labels[-1] = 0
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention,
                                       full, prefix, pad_id)

    def test_truncated_attention_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention[:-1],
                                       full, prefix, pad_id)

    def test_boolean_pad_id_rejected(self):
        inputs, labels, attention, full, prefix, pad_id = self.rows()
        with self.assertRaises(ValueError):
            verify_token_row_integrity(inputs, labels, attention,
                                       full, prefix, False)


if __name__ == "__main__":
    unittest.main()
