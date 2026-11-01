"""H335: invented-only contracts and original GUI first-click negatives.

All buttons/points in these tests are deliberately source-owned oracles,
NEVER genuine Qwen generations. Never claim model success from them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from h199.native_action_sft import source_task_prompt
from h319.beaconboard_dynamic_research import IDS, SCENES, exact_task
from h335.original_first_click_error_attribution import (
    ROLE, CATEGORIES, classify_first_click, source_first_click,
    aggregate_first_click_categories, audit_first_click_source,
)

SOURCE = Path(__file__).resolve().parents[1] / "h319" / "beaconboard.html"


def original_row(lang: str, category: str) -> dict:
    return {
        "status": ROLE, "language": lang, "category": category,
        "source_public_exposed": True,
        "native_actions_dispatched": 0 if category == "abstain" else 1,
        "genuine_model_identity_attested": False,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "can_release": False,
    }


class H335InventedOnly(unittest.TestCase):
    def test_each_first_click_categorical_root_cause(self):
        empty = {key: (False, "") for key in IDS}
        correct = dict(empty, **{IDS[0]: (True, "")})
        wrong = dict(empty, **{IDS[1]: (True, "")})
        self.assertEqual(classify_first_click(correct, IDS[0], IDS[0]),
                         "correct_open_and_focus")
        self.assertEqual(classify_first_click(correct, None, IDS[0]),
                         "correct_open_wrong_focus")
        self.assertEqual(classify_first_click(wrong, IDS[1], IDS[0]),
                         "wrong_station_open")
        self.assertEqual(classify_first_click(empty, None, IDS[0]),
                         "no_station_open")

    def test_impossible_states_fail_closed(self):
        empty = {key: (False, "") for key in IDS}
        for bad_state, focus, target in (
            (dict(empty, **{IDS[0]: (True, "RAW-SIM")}), IDS[0], IDS[0]),
            (dict(empty, **{IDS[0]: (True, ""), IDS[1]: (True, "")}),
             IDS[0], IDS[0]),
            (empty, IDS[0], IDS[0]),
            (dict(empty, **{IDS[0]: (True, "")}), IDS[1], IDS[0]),
            (dict(empty, **{IDS[0]: (1, "")}), IDS[0], IDS[0]),
            ({"invented": (True, "")}, None, IDS[0]),
            (empty, None, "NOT_A_STATION"),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                classify_first_click(bad_state, focus, target)

    def test_only_5_fixed_bounded_diagnostic_categories(self):
        self.assertEqual(set(CATEGORIES), {
            "correct_open_and_focus", "correct_open_wrong_focus",
            "wrong_station_open", "no_station_open", "abstain"})
        self.assertNotIn("point", json.dumps(CATEGORIES))
        self.assertNotIn("model", json.dumps(CATEGORIES))

    def test_valid_aggregate_by_language_without_private_payloads(self):
        rows = ([original_row("en", "correct_open_and_focus")] * 4 +
                [original_row("en", "wrong_station_open")] * 2 +
                [original_row("ar", "no_station_open")] * 2 +
                [original_row("ar", "abstain")] * 4)
        summary = aggregate_first_click_categories(rows)
        self.assertEqual(summary["original_source_scenes"], 12)
        self.assertEqual(summary["by_language"]["en"]["categories"]
                         ["correct_open_and_focus"], 4)
        self.assertEqual(summary["by_language"]["ar"]["native_actions"], 2)
        self.assertFalse(summary["real_qwen_model_generations_attested"])
        self.assertFalse(summary["can_release"])
        self.assertNotIn("coordinates", json.dumps(summary))

    def test_reject_extra_private_fields_false_ood_or_missing_case(self):
        good = ([original_row("en", "correct_open_and_focus")] * 6 +
                [original_row("ar", "wrong_station_open")] * 6)
        for rows in (
            good[:11],
            good + [original_row("ar", "abstain")],
            [dict(good[0], raw_prediction="secret")] + good[1:],
            [dict(good[0], fresh_blind_ood=True)] + good[1:],
            [dict(good[0], native_actions_dispatched=True)] + good[1:],
            [dict(good[0], category="private_coord:42")] + good[1:],
            [original_row("en", "correct_open_and_focus")] * 12,
        ):
            with self.subTest(n=len(rows)), self.assertRaises(ValueError):
                aggregate_first_click_categories(rows)

    def test_invalid_roles_refuse_before_any_browser(self):
        for role in ("TRAIN", "DEV", "BLIND_OOD", "H333"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                source_first_click(
                    None, SOURCE, SCENES[0], lambda a, b: "anything",
                    role=role)
            with self.assertRaises(ValueError):
                audit_first_click_source(
                    None, SOURCE, lambda scene: lambda a,b: "",
                    role=role)


@unittest.skipUnless(os.environ.get("HALA_H335_BROWSER_QUALIFY") == "1",
                     "explicit original full browser source qualification required")
class H335OriginalBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H335_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("unsupported original source browser")
            cls.browser = cls.pw.chromium.launch(
                headless=True,
                **({"channel": "msedge"} if channel == "msedge"
                   else {"args": ["--no-sandbox"]}))
        except Exception:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def _invented_point(self, scene, *, wrong=False):
        page = self.browser.contexts[-1].pages[-1]
        target = (scene[4] + (1 if wrong else 0)) % 4
        box = page.locator(
            'button[data-station="' + IDS[target] + '"]').bounding_box()
        self.assertIsNotNone(box)
        return [int(box["x"] + box["width"] / 2),
                int(box["y"] + box["height"] / 2)]

    def _invented_callback(self, scene, behavior):
        def predict(png, task):
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(task, source_task_prompt(exact_task(scene)))
            if behavior == "abstain":
                return '{"actions":[{"kind":"abstain"}]}'
            point = (self._invented_point(scene, wrong=behavior == "wrong")
                     if behavior in ("correct", "wrong")
                     else [1, 1])
            return json.dumps({"actions": [{
                "kind": "click", "point": point}]}, separators=(",", ":"))
        return predict

    def test_all_12_original_correct_clicks_and_aggregate_only(self):
        result = audit_first_click_source(
            self.browser, SOURCE,
            lambda scene: self._invented_callback(scene, "correct"),
            role=ROLE)
        self.assertEqual(result["original_source_scenes"], 12)
        for lang in ("en", "ar"):
            self.assertEqual(result["by_language"][lang]["scenes"], 6)
            self.assertEqual(result["by_language"][lang]["categories"]
                             ["correct_open_and_focus"], 6)
        self.assertFalse(result["fresh_blind_ood"])

    def test_wrong_button_missed_click_and_abstain_in_both_languages(self):
        for scene in (SCENES[0], SCENES[8]):
            for behavior, expected in (
                ("wrong", "wrong_station_open"),
                ("miss", "no_station_open"),
                ("abstain", "abstain"),
            ):
                with self.subTest(scene=scene[0], behavior=behavior):
                    row = source_first_click(
                        self.browser, SOURCE, scene,
                        self._invented_callback(scene, behavior), role=ROLE)
                    self.assertEqual(row["category"], expected)
                    self.assertFalse(row["genuine_model_identity_attested"])
                    self.assertFalse(row["can_release"])


if __name__ == "__main__":
    unittest.main()
