"""H337 invented-only geometry tests; real source browser optional.

Test callbacks use **source-owned gold points only for qualification**.
No predicted Qwen point or model generation is used or persisted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from h319.beaconboard_dynamic_research import IDS, SCENES, exact_task
from h199.native_action_sft import source_task_prompt
from h337.source_miss_distance_bands import (
    ROLE, CATEGORIES, classify_source_missed_point,
    source_first_click_distance, audit_source_distance,
    aggregate_source_distance,
)

SOURCE = Path(__file__).resolve().parents[1] / "h319" / "beaconboard.html"


def boxes():
    return {key: {"x": 200 + 130*i, "y": 200,
                  "width": 90, "height": 60}
            for i, key in enumerate(IDS)}


def row(lang, cat):
    return {
        "status": ROLE, "language": lang, "category": cat,
        "native_actions_dispatched": 0 if cat == "abstain" else 1,
        "source_public_exposed": True,
        "real_model_identity_attested": False,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "learned_adapter_gain": False, "can_release": False,
    }


class H337InventedSourceBoundary(unittest.TestCase):
    def test_coarse_target_distance_bands_both_axes(self):
        data = boxes()
        expected = {
            (240, 230): "miss_inside_expected_button",
            (195, 230): "miss_near_target_0_to_12px",
            (187, 230): "miss_offset_target_12_to_48px",
            (140, 230): "miss_farther_than_48px",
            (245, 194): "miss_near_target_0_to_12px",
            (245, 173): "miss_offset_target_12_to_48px",
            (245, 80): "miss_farther_than_48px",
            (195, 195): "miss_near_target_0_to_12px",
        }
        for point, category in expected.items():
            with self.subTest(point=point):
                self.assertEqual(classify_source_missed_point(
                    list(point), IDS[0], data), category)

    def test_invalid_points_rects_target_and_extra_fields_rejected(self):
        base = boxes()
        for point in ([True, 1], [960, 0], [0, -1], [5.0, 6],
                      [0, 0, 1], "5,6", None):
            with self.subTest(point=point), self.assertRaises(ValueError):
                classify_source_missed_point(point, IDS[0], base)
        for bad in (
            {"x": 5, "y": 6, "width": 0, "height": 10},
            {"x": 5, "y": 6, "width": float("nan"), "height": 10},
            {"x": 5, "y": 6, "width": 956, "height": 10},
            {"x": 5, "y": 6, "width": 10, "height": True},
            {"x": 5, "y": 6, "width": 10, "height": 10,
             "secret": "gold"},
        ):
            with self.subTest(box=bad), self.assertRaises(ValueError):
                classify_source_missed_point(
                    [12, 12], IDS[0], dict(base, **{IDS[1]: bad}))
        with self.assertRaises(ValueError):
            classify_source_missed_point([1, 1], "NOT-STATION", base)

    def test_only_bounded_labels_and_no_model_evidence(self):
        self.assertEqual(len(CATEGORIES), 8)
        self.assertNotIn("point", json.dumps(CATEGORIES))
        self.assertNotIn("SIM-", json.dumps(CATEGORIES))
        records = (
            [row("en", "correct_open_and_focus")] * 5
            + [row("en", "miss_near_target_0_to_12px")]
            + [row("ar", "correct_open_and_focus")] * 4
            + [row("ar", "miss_offset_target_12_to_48px")]
            + [row("ar", "miss_farther_than_48px")]
        )
        result = aggregate_source_distance(records)
        self.assertEqual(result["by_language"]["en"]["scenes"], 6)
        self.assertEqual(result["by_language"]["ar"]["categories"]
                         ["miss_farther_than_48px"], 1)
        self.assertFalse(result["learned_gain_proven"])
        self.assertFalse(result["fresh_blind_ood"])
        self.assertFalse(result["can_release"])
        for label in ("coordinates", "image_bytes", "gold",
                      "source_target", "predicted_point"):
            self.assertNotIn(label, json.dumps(result))

    def test_no_partial_or_private_category_promotions(self):
        good = [row("en", "correct_open_and_focus")] * 6 + (
            [row("ar", "miss_farther_than_48px")] * 6)
        for bad in (
            good[:11], good + good[:1],
            [dict(good[0], raw_model_point=[9, 8])] + good[1:],
            [dict(good[0], fresh_blind_ood=True)] + good[1:],
            [dict(good[0], category="actual_pixel:20")] + good[1:],
            [dict(good[0], native_actions_dispatched=True)] + good[1:],
            [row("en", "correct_open_and_focus")] * 12,
        ):
            with self.subTest(n=len(bad)), self.assertRaises(ValueError):
                aggregate_source_distance(bad)

    def test_unauthorized_roles_refused_before_browser_source(self):
        for role in ("TRAIN", "BLIND_OOD", "PRIVATE_GOLD", None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                source_first_click_distance(
                    None, SOURCE, SCENES[0], lambda a, b: "",
                    role=role)
            with self.assertRaises(ValueError):
                audit_source_distance(
                    None, SOURCE, lambda scene: lambda a,b: "",
                    role=role)


@unittest.skipUnless(os.environ.get("HALA_H337_SOURCE_BROWSER") == "1",
                     "original source Playwright browser qualification opt-in")
class H337OriginalSourceBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H337_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("browser not allowlisted")
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

    def predictor(self, scene, mode):
        def fake(png, task):
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(task, source_task_prompt(exact_task(scene)))
            if mode == "abstain":
                return '{"actions":[{"kind":"abstain"}]}'
            page = self.browser.contexts[-1].pages[-1]
            station = IDS[(scene[4] + (1 if mode == "wrong" else 0)) % 4]
            box = page.locator(
                'button[data-station="' + station + '"]').bounding_box()
            self.assertIsNotNone(box)
            if mode == "far":
                point = [1, 1]
            else:
                x = round(box["x"] + box["width"] / 2)
                y = round(box["y"] + box["height"] / 2)
                point = ([x, max(1, int(box["y"]) - 5)]
                         if mode == "near" else [x, y])
            return json.dumps({"actions": [{"kind": "click",
                                            "point": point}]})
        return fake

    def test_original_12_source_owned_centers_only(self):
        result = audit_source_distance(
            self.browser, SOURCE,
            lambda scene: self.predictor(scene, "correct"), role=ROLE)
        for language in ("en", "ar"):
            self.assertEqual(result["by_language"][language]["categories"]
                             ["correct_open_and_focus"], 6)
        self.assertFalse(result["learned_gain_proven"])

    def test_real_english_and_arabic_miss_buckets_and_controls(self):
        for scene in (SCENES[0], SCENES[6]):
            for mode, expected in (
                ("near", "miss_near_target_0_to_12px"),
                ("far", "miss_farther_than_48px"),
                ("wrong", "wrong_station_open"),
                ("abstain", "abstain"),
            ):
                with self.subTest(lang=scene[1],mode=mode):
                    outcome = source_first_click_distance(
                        self.browser, SOURCE, scene,
                        self.predictor(scene, mode), role=ROLE)
                    self.assertEqual(outcome["category"], expected)
                    self.assertFalse(outcome["real_model_identity_attested"])


if __name__ == "__main__":
    unittest.main()
