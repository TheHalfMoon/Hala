"""H362 exact bilingual source-balance properties, NOT TRAIN/DEV/OOD."""
from __future__ import annotations

from collections import Counter
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h355.signed_rights_intake import TRUSTED_REVIEWERS
from h356.sealed_ood_custody import TRUSTED_CUSTODIANS
from h357.source_candidate import candidate_specs
from h358.source_candidate import scenes


class H362NoLanguageShortcutTests(unittest.TestCase):
    def test_controlgarden_shape_color_and_position_are_balanced_for_each_language(self):
        cases = candidate_specs()
        self.assertEqual(len(cases), 64)
        self.assertEqual(len({tuple(tuple(t) for t in c["source_tiles"])
                              for c in cases}), 64)
        for lang in ("en", "ar"):
            rows = [c for c in cases if c["language"] == lang]
            self.assertEqual(len(rows), 32)
            cells = [(c["oracle_tile"],
                      *c["source_tiles"][c["oracle_tile"]])
                     for c in rows]
            slots = Counter(s for s, _, _ in cells)
            shapes = Counter(shape for _, shape, _ in cells)
            colors = Counter(color for _, _, color in cells)
            pairs_shape = Counter((s, shape) for s, shape, _ in cells)
            pairs_color = Counter((s, color) for s, _, color in cells)
            for tally in (slots, shapes, colors):
                self.assertEqual(sorted(tally.values()), [8]*4)
            self.assertEqual(len(pairs_shape), 16)
            self.assertEqual(len(pairs_color), 16)
            self.assertEqual(sorted(pairs_shape.values()), [2]*16)
            self.assertEqual(sorted(pairs_color.values()), [2]*16)

    def test_noterelay_nonlinguistic_scene_fields_mirror_across_language(self):
        rows = scenes()
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r["code"] for r in rows}), 16)
        self.assertEqual(len({r["case_id"] for r in rows}), 32)
        self.assertEqual(Counter(r["language"] for r in rows),
                         {"en":16, "ar":16})
        for i in range(0, 32, 2):
            en, ar = rows[i:i+2]
            self.assertEqual((en["language"], ar["language"]), ("en","ar"))
            for field in ("code", "workspace", "theme"):
                self.assertEqual(en[field], ar[field])
            self.assertNotEqual(en["task"], ar["task"])
            self.assertIn(en["code"], en["task"])
            self.assertIn(ar["code"], ar["task"])
        for lang in ("en", "ar"):
            arm = [r for r in rows if r["language"] == lang]
            self.assertEqual(Counter(r["theme"] for r in arm),
                             {"light": 8, "dark": 8})
            self.assertEqual(Counter(r["workspace"] % 2 for r in arm),
                             {0: 8, 1: 8})

    def test_balanced_sources_remain_unadmitted_candidate_only(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        self.assertEqual(len(TRUSTED_REVIEWERS), 0)
        self.assertEqual(len(TRUSTED_CUSTODIANS), 0)
        self.assertEqual({r["split"] for r in candidate_specs()},
                         {"TRAIN_DEV_CANDIDATE_NOT_ADMITTED"})
        self.assertEqual({r["split"] for r in scenes()},
                         {"ORIGINAL_MULTISTEP_TRAIN_DEV_REVIEW_CANDIDATE_NOT_ADMITTED"})


if __name__ == "__main__":
    unittest.main()
