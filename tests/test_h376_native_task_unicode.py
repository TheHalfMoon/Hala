"""H376: adversarial source-only bilingual native task Unicode preflight.

No model weights, original screenshots, gold, train approval, sealed OOD or GPU.
"""
from __future__ import annotations

import json
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS,
    source_task_prompt,
    training_messages_for_synthetic_contract,
    validate_model_visible_task,
)


class H376TaskUnicodePreflightTests(unittest.TestCase):
    def test_bilingual_diacritics_emoji_and_normal_multiline_unchanged(self):
        examples = (
            "اضغط على زر حفظ",
            "افتح المِلَفَّ أولًا، ثم اضغط على حفظ",
            "حرّك المؤشر في الحقل 📝",
            "Open the Save button and confirm.",
            "خطوة أولى\nخطوة ثانية",
            "label\tcode",
            "می‌خواهم گزینه را انتخاب کنم",  # Persian ZWNJ is legitimate
        )
        for task in examples:
            with self.subTest(length=len(task)):
                self.assertEqual(validate_model_visible_task(task), task)
                self.assertTrue(source_task_prompt(task).endswith("Task: " + task))
                self.assertEqual(source_task_prompt(task).count("Task: "), 1)

    def test_bidi_isolation_override_embeds_denied(self):
        for code in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)):
            with self.subTest(code=hex(code)):
                with self.assertRaisesRegex(ValueError, "unsafe Unicode"):
                    source_task_prompt("احفظ" + chr(code) + "hidden")

    def test_lone_surrogate_is_denied_before_downstream_utf8_encoding(self):
        for code in (0xD800, 0xDBFF, 0xDC00, 0xDFFF):
            task = "private" + chr(code)
            with self.subTest(code=hex(code)):
                with self.assertRaises(ValueError):
                    source_task_prompt(task)

    def test_untrusted_terminal_controls_and_invisible_spoofs_denied(self):
        bad = (
            "\x00", "\r", "\x1b[31m", "\x7f",
            "\x08", "\x0b", "\u0085", "\u200b", "\ufeff",
        )
        for character in bad:
            with self.subTest(codepoint=tuple(map(ord, character))):
                with self.assertRaises(ValueError):
                    source_task_prompt("Click" + character + "Save")

    def test_strict_task_type_and_size_are_not_weakened(self):
        for task in (None, 1, True, "", "A" * 501):
            with self.subTest(task_type=type(task).__name__):
                with self.assertRaises(ValueError):
                    validate_model_visible_task(task)

    def test_invalid_task_rejected_in_native_sft_before_any_training(self):
        completion = json.dumps({
            "actions": [{"kind": "click", "point": [300, 200]}]
        }, separators=(",", ":"))
        with self.assertRaises(ValueError):
            training_messages_for_synthetic_contract(
                "مرحبا" + chr(0x202E), completion)
        question, answer = training_messages_for_synthetic_contract(
            "انقر على زر الحفظ", completion)
        self.assertEqual(question[0]["content"][0], {"type": "image"})
        self.assertIn("انقر على زر الحفظ", question[0]["content"][1]["text"])
        self.assertEqual(answer[-1]["role"], "assistant")
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())


if __name__ == "__main__":
    unittest.main()
