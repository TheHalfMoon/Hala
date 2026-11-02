"""H377: Unicode Cf / bidi invisible-instruction gap after H376.

Invented text only; no screenshots, gold, model weights, training or OOD.
"""
from __future__ import annotations

import json
import unicodedata
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS,
    source_task_prompt,
    training_messages_for_synthetic_contract,
    validate_model_visible_task,
)

DENY = (
    0x061C,  # Arabic Letter Mark
    0x200E,  # Left to Right Mark
    0x200F,  # Right to Left Mark
    0x00AD,  # Soft Hyphen
    0x2060,  # Word Joiner
    0x180E,  # Mongolian Vowel Separator
    0x2028,  # Line Separator
    0x2029,  # Paragraph Separator
    0x200B,  # Zero Width Space
    0xFEFF,  # BOM
)


class H377BilingualTaskFormattingTests(unittest.TestCase):
    def test_actual_reproduced_invisible_direction_and_separator_controls_denied(self):
        for code in DENY:
            text = "احفظ" + chr(code) + "DELETE"
            with self.subTest(codepoint=hex(code)):
                with self.assertRaises(ValueError):
                    source_task_prompt(text)

    def test_all_unicode_cf_except_explicit_join_shaping_are_denied(self):
        # Exhaustive over the small set of Unicode format controls.
        for code in range(0x110000):
            if unicodedata.category(chr(code)) != "Cf":
                continue
            task = "click" + chr(code) + "Save"
            if code in (0x200C, 0x200D):
                self.assertEqual(validate_model_visible_task(task), task)
            else:
                with self.subTest(codepoint=hex(code)):
                    with self.assertRaises(ValueError):
                        validate_model_visible_task(task)

    def test_legitimate_arabic_persian_english_and_emoji_are_unchanged(self):
        for task in (
            "انقر على الحفظ",
            "اكتب السَّلَامُ عَلَيْكُمْ",
            "می\u200cخواهم گزینه\u200cها را ببینم",
            "حرف\u200dالربط",
            "Click Save ✅",
            "سطر أول\nسطر ثانٍ",
            "code\tvalue",
        ):
            with self.subTest(task_length=len(task)):
                self.assertEqual(validate_model_visible_task(task), task)
                self.assertTrue(source_task_prompt(task).endswith("Task: " + task))

    def test_invalid_unicode_never_reaches_h199_supervised_action_messages(self):
        action = json.dumps(
            {"actions": [{"kind": "click", "point": [100, 200]}]},
            separators=(",", ":"),
        )
        for code in (0x061C, 0x200E, 0x200F, 0x2060, 0x2028, 0x2029):
            with self.subTest(codepoint=hex(code)):
                with self.assertRaises(ValueError):
                    training_messages_for_synthetic_contract(
                        "انقر" + chr(code) + "Save", action)
        question, answer = training_messages_for_synthetic_contract(
            "انقر على زر حفظ", action)
        self.assertEqual(question[0]["content"][0], {"type": "image"})
        self.assertEqual(answer[-1]["role"], "assistant")
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())


if __name__ == "__main__":
    unittest.main()
