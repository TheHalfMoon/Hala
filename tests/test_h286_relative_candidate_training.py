"""H286 invented-only CPU tests; never opens private GUI screenshots/gold."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from h199.native_action_sft import source_task_prompt
from h273.relative_native_bridge import normalized_to_pixel, normalize_strict_action
from h285.train_only_browser_candidates import ROLE, SCENES, SOURCE_BLOB, exact_task
from h286.relative_candidate_training import (
    aligned_relative_sample, pixel_to_normalized_exact,
    produce_relative_train_candidates,
)


def invented_label(x: int, y: int) -> str:
    return json.dumps({"actions": [{"kind": "click", "point": [x, y]}]},
                      separators=(",", ":"))


def invented_source_summary() -> dict:
    return {
        "role": ROLE, "original_source_blob": SOURCE_BLOB,
        "actual_browser_source_actions_successful": 24,
        "distinct_png_screenshots": 24,
        "distinct_decoded_rgba_screenshots": 24,
        "english_candidates": 12, "arabic_candidates": 12,
        "official_train_manifest_admitted": False,
        "independent_dev_ood_evaluated": False,
    }


class H286RelativeTrainingContractTests(unittest.TestCase):
    def test_every_pixel_exactly_recovered_in_both_axes(self):
        for extent in (960, 720):
            for pixel in range(extent):
                n = pixel_to_normalized_exact(pixel, extent)
                self.assertEqual(normalized_to_pixel(n, extent), pixel)
                if n > 0:
                    self.assertNotEqual(normalized_to_pixel(n - 1, extent), pixel)

    def test_invented_arabic_and_english_labels_match_relative_prompt(self):
        for scene in SCENES:
            task = exact_task(scene)
            prompt, label = aligned_relative_sample(
                scene, source_task_prompt(task), invented_label(479, 359))
            self.assertIn(task, prompt)
            self.assertIn("RELATIVE", prompt)
            self.assertIn("Never output absolute pixel", prompt)
            self.assertEqual(normalize_strict_action(label), invented_label(479, 359))
        self.assertEqual(len(SCENES), 24)

    def test_coordinate_type_and_range_fail_closed(self):
        for point, extent in ((True, 960), (0.0, 960), (-1, 960),
                              (960, 960), (720, 720), (1, 800),
                              (1, True)):
            with self.subTest(point=point, extent=extent):
                with self.assertRaises(ValueError):
                    pixel_to_normalized_exact(point, extent)

    def test_wrong_task_or_scene_cannot_be_used_as_supervision(self):
        scene = SCENES[0]
        for wrong in (source_task_prompt(exact_task(SCENES[1])),
                      "Complete any card", "", None):
            with self.assertRaises(ValueError):
                aligned_relative_sample(scene, wrong, invented_label(100, 100))
        with self.assertRaises(ValueError):
            aligned_relative_sample(("unregistered",),
                                    source_task_prompt(exact_task(scene)),
                                    invented_label(100, 100))

    def test_no_abstain_drag_floats_or_extra_keys_as_training_targets(self):
        scene = SCENES[0]
        prompt = source_task_prompt(exact_task(scene))
        invalid = (
            '{"actions":[{"kind":"abstain"}]}',
            '{"actions":[{"kind":"drag","start":[10,10],"end":[20,20]}]}',
            '{"actions":[{"kind":"click","point":[12.5,20]}]}',
            '{"actions":[{"kind":"click","point":[20,20],"gold":1}]}',
            '{"actions":[{"kind":"click","point":[20,20],"point":[20,20]}]}',
        )
        for label in invalid:
            with self.assertRaises(ValueError):
                aligned_relative_sample(scene, prompt, label)

    def test_refuses_wrong_role_before_any_browser_or_sink(self):
        for role in ("TRAIN", "DEV", "OOD", "BLIND", "OFFICIAL_TRAIN", None):
            with self.assertRaises(ValueError):
                produce_relative_train_candidates(
                    None, Path("unused"), lambda *args: None, role=role)
        with self.assertRaises(ValueError):
            produce_relative_train_candidates(None, Path("unused"), None, role=ROLE)

    def test_24_complete_source_rows_convert_only_after_full_validation(self):
        emitted = []
        def fake_original(browser, source, sink, *, role):
            self.assertEqual(role, ROLE)
            for i, scene in enumerate(SCENES):
                sink(b"\x89PNG\r\n\x1a\n" + bytes([i]),
                     source_task_prompt(exact_task(scene)),
                     invented_label(200 + i, 300 + i))
            self.assertEqual(emitted, [])
            return invented_source_summary()
        with patch("h286.relative_candidate_training.produce_train_only_batch",
                   side_effect=fake_original):
            report = produce_relative_train_candidates(
                object(), Path("unused"), lambda *row: emitted.append(row),
                role=ROLE)
        self.assertEqual(len(emitted), 24)
        self.assertEqual(report["relative_targets_roundtrip_verified"], 24)
        self.assertEqual(report["real_model_generations"], 0)
        self.assertFalse(report["official_train_admitted"])
        self.assertFalse(report["source_disjoint_blind_ood"])
        self.assertFalse(report["adapted_gui_gain_proven"])
        self.assertNotIn("point", str(report).lower())
        self.assertNotIn("png", str(report).lower())

    def test_incomplete_or_wrong_source_report_never_emits_anything(self):
        for corruption in ("actual_browser_source_actions_successful",
                           "distinct_decoded_rgba_screenshots",
                           "official_train_manifest_admitted",
                           "original_source_blob"):
            emitted = []
            def fake_original(browser, source, sink, *, role):
                for i, scene in enumerate(SCENES):
                    sink(b"\x89PNG\r\n\x1a\n" + bytes([i]),
                         source_task_prompt(exact_task(scene)),
                         invented_label(100, 200))
                summary = invented_source_summary()
                summary[corruption] = True if corruption == "official_train_manifest_admitted" else "drift"
                return summary
            with patch("h286.relative_candidate_training.produce_train_only_batch",
                       side_effect=fake_original):
                with self.assertRaises(ValueError):
                    produce_relative_train_candidates(
                        object(), Path("unused"), lambda *row: emitted.append(row),
                        role=ROLE)
            self.assertEqual(emitted, [])

    def test_malformed_24th_row_does_not_partially_emit(self):
        emitted = []
        def fake_original(browser, source, sink, *, role):
            for i, scene in enumerate(SCENES):
                sink(b"\x89PNG\r\n\x1a\n" + bytes([i]),
                     source_task_prompt(exact_task(scene)),
                     invented_label(100, 200) if i != 23 else
                     '{"actions":[{"kind":"abstain"}]}')
            return invented_source_summary()
        with patch("h286.relative_candidate_training.produce_train_only_batch",
                   side_effect=fake_original):
            with self.assertRaises(ValueError):
                produce_relative_train_candidates(
                    object(), Path("unused"), lambda *row: emitted.append(row),
                    role=ROLE)
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
