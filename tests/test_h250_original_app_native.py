"""H250: synthetic-only original app replay receipt validation; no donor/media."""
from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h250.native_original_todo_app import (
    CASES, FILES, REPO, STAGES, STATUS, action, preflight, verify_row,
)


def fake_case(spec):
    kinds = (
        ("click", {"point": [340, 262]}),
        ("type_text", {"text": spec["text"]}),
        ("click", {"point": [600, 260]}),
        ("scroll", {"point": [480, 600], "dy": 480}),
    )
    edges = (
        ("before_focus", "after_focus"),
        ("after_focus", "after_type"),
        ("after_type", "after_add"),
        ("before_scroll", "after_scroll"),
    )
    return {
        "case_id": spec["id"], "language": spec["lang"],
        "native_add_verified": True, "native_reload_verified": True,
        "native_scroll_verified": True,
        "train": False, "dev": False, "ood": False,
        "frames": {stage: {"png_sha256": f"{i+1:064x}",
                           "rgba_sha256": f"{i+20:064x}"}
                   for i, stage in enumerate(STAGES)},
        "steps": [
            {"instruction": f"Perform {kind} for the original UI.",
             "native_action": action(kind, **attrs),
             "source_frame": edge[0], "result_frame": edge[1],
             "research_only": True}
            for (kind, attrs), edge in zip(kinds, edges)
        ],
    }


class H250OriginalAppResearchTests(unittest.TestCase):
    def test_exact_bilingual_one_action_horizons(self):
        self.assertEqual(len(CASES), 2)
        self.assertEqual(len(STAGES), 6)
        for case in CASES:
            with self.subTest(lang=case["lang"]):
                verify_row(fake_case(case), case)

    def test_h199_release_admission_stays_empty(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        self.assertIn("NOT_ADMITTED", STATUS)
        self.assertEqual(REPO, "serkanbyx/simple-to-do-list")
        self.assertEqual(set(FILES), {"LICENSE","index.html","css/style.css","js/app.js"})

    def test_swapped_screenshot_action_and_media_duplicates_denied(self):
        case = CASES[0]
        for modify in (
            lambda row: row["steps"][2].update(source_frame="before_focus"),
            lambda row: row["steps"][1].update(native_action=action("click", point=[1,2])),
            lambda row: row["steps"][1].update(
                native_action=action("type_text",text="wrong")),
            lambda row: row["steps"][0].update(research_only=False),
            lambda row: row["frames"]["after_type"].update(
                png_sha256=row["frames"]["after_focus"]["png_sha256"]),
            lambda row: row.update(native_scroll_verified=False),
            lambda row: row.update(train=True),
            lambda row: row.update(dev=True),
            lambda row: row.update(ood=True),
        ):
            row = fake_case(case)
            modify(row)
            with self.subTest(row=repr(row)[:80]), self.assertRaises(ValueError):
                verify_row(row, case)

    def test_reject_unsafe_combined_native_action(self):
        for bad in (
            {"actions":[{"kind":"click","point":[1,2]},
                        {"kind":"type_text","text":"hello"}]},
            {"actions":[{"kind":"abstain"}]},
        ):
            with self.assertRaises(ValueError):
                from h199.native_action_sft import parse_actions
                import json
                parse_actions(json.dumps(bad), training_target=True)

    def test_absent_original_checkout_denied_before_any_capture(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError,ValueError)):
                preflight(Path(tmp)/"missing-original-source",
                          Path(tmp)/"h250-owner-private-research-test")


if __name__ == "__main__":
    unittest.main()
