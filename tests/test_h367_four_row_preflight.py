"""H367 stdlib-only source contract: DOES NOT run real Qwen or TRAIN."""
from __future__ import annotations

from collections import Counter
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h357.source_candidate import candidate_specs
from h358.source_candidate import scenes
from h367.four_row_processor import SCHEMA, SELECTION, source_selection


class H367FourRowSourceSelectionTests(unittest.TestCase):
    def test_two_languages_and_two_actions_in_each_language(self):
        chosen = source_selection()
        self.assertEqual(chosen, SELECTION)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(Counter((lang, kind) for _, _, lang, kind in chosen), {
            ("en", "click"): 1, ("ar", "click"): 1,
            ("en", "type_text"): 1, ("ar", "type_text"): 1,
        })
        self.assertEqual(len({(family, idx) for family, idx, _, _ in chosen}), 4)
        self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)
        self.assertTrue(SCHEMA.startswith("hala.h367."))

    def test_selection_is_grounded_in_current_original_source_rules(self):
        controls, notes = candidate_specs(), scenes()
        for family, idx, lang, kind in SELECTION:
            with self.subTest(family=family, index=idx):
                if kind == "click":
                    self.assertEqual(family, "controlgarden-single-app")
                    self.assertEqual(controls[idx]["language"], lang)
                else:
                    self.assertEqual(family, "noterelay-original-multistep")
                    self.assertEqual(notes[idx // 4]["language"], lang)
                    self.assertEqual(("open", "focus", "type", "save")[idx % 4], "type")


if __name__ == "__main__":
    unittest.main()
