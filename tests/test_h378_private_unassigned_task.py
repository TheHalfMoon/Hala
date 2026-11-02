"""H378: invented-only H199 native task Unicode private-use/unassigned guard."""
import json
import unicodedata
import unittest
from h199.native_action_sft import (validate_model_visible_task,
    source_task_prompt, training_messages_for_synthetic_contract,
    ADMITTED_TRAIN_MANIFEST_SHAS)

BAD = (0xE000, 0xF8FF, 0xF0000, 0x10FFFD, 0x0378, 0xFDD0, 0xFFFF)

class H378NativeTaskCharacterTests(unittest.TestCase):
    def test_private_use_unassigned_and_noncharacters_refused(self):
        for code in BAD:
            self.assertIn(unicodedata.category(chr(code)), ("Co", "Cn"))
            with self.subTest(codepoint=hex(code)):
                with self.assertRaises(ValueError):
                    validate_model_visible_task("انقر" + chr(code) + "Save")

    def test_same_input_refused_before_native_sft_pack(self):
        action = json.dumps({"actions":[{"kind":"click","point":[100,200]}]})
        for code in (0xE000, 0x0378, 0xFDD0):
            with self.subTest(codepoint=hex(code)):
                with self.assertRaises(ValueError):
                    training_messages_for_synthetic_contract(
                        "Click " + chr(code) + " Save", action)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())

    def test_legitimate_arabic_persian_emoji_and_layout_unchanged(self):
        for task in ("انقر على الحفظ", "السَّلَامُ عَلَيْكُمْu200c",
                     "می‌خواهم", "حرف‍الربط", "Click Save ✅ 👩‍💻",
                     "السطر الأول\nالسطر الثاني", "key\tvalue"):
            with self.subTest(task_length=len(task)):
                self.assertEqual(validate_model_visible_task(task),task)
                self.assertTrue(source_task_prompt(task).endswith("Task: "+task))

if __name__ == "__main__":
    unittest.main()
