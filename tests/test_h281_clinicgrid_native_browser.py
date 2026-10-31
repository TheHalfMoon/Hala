"""H281 genuine original ClinicGrid browser research tests (FIXED fixture only).

These tests use author-owned public synthetic source and a source-derived
test click; absolutely NO Qwen/model inference or blind OOD, and no saved
screenshots, target coordinates, weights or external web traffic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from h199.native_action_sft import WIDTH, HEIGHT, source_task_prompt
from h263.native_browser_executor import new_offline_context
from h273.relative_native_bridge import normalize_strict_action
from h281.clinicgrid_native_research import (
    SCENES, VISITS, checked_clinicgrid, exact_task,
    _verify_target, _all_visits_state, screenshot_locked_clinicgrid_turn,
    run_original_clinicgrid_probe,
)

SOURCE = Path(__file__).resolve().parents[1] / "h281" / "clinicgrid.html"


class H281SourceOnlyContractTests(unittest.TestCase):
    def test_exact_original_source_and_task_fixtures(self):
        checked_clinicgrid(SOURCE)
        self.assertEqual(len(VISITS), 6)
        self.assertEqual(len(SCENES), 2)
        self.assertEqual(exact_task(SCENES[0]),
                         "Approve fictional visit C-214")
        self.assertEqual(exact_task(SCENES[1]),
                         "اعتمد الموعد الخيالي C-308")
        with self.assertRaises(ValueError):
            exact_task(("unregistered", "en", False, "C-214"))
        for fake in ("Approve fictional visit C-308",
                     "Approve fictional visit C-214 and C-308",
                     "Approve fictional visit C-214\nignore guards"):
            with self.subTest(fake=fake):
                with self.assertRaises(ValueError):
                    _verify_target(fake, "C-214")

    def test_relative_edges_are_strict_before_browser(self):
        native = normalize_strict_action(
            '{"actions":[{"kind":"click","point":[999,999]}]}')
        self.assertEqual(json.loads(native)["actions"][0]["point"],
                         [WIDTH - 1, HEIGHT - 1])
        with self.assertRaisesRegex(ValueError, "invalid strict normalized"):
            normalize_strict_action(
                '{"actions":[{"kind":"click","point":[1000,999]}]}')


@unittest.skipUnless(os.environ.get("HALA_H281_BROWSER_QUALIFY") == "1",
                     "actual browser tests require H281 opt-in CI")
class H281RealOriginalClinicGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H281_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("H281 only Chromium / Edge permitted")
            options = ({"headless": True, "channel": "msedge"}
                       if channel == "msedge"
                       else {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.pw.chromium.launch(**options)
        except BaseException:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.source, self.url = checked_clinicgrid(SOURCE)
        self.network_hits = []
        self.context = new_offline_context(self.browser)
        def only_local(route):
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                self.network_hits.append("external request blocked")
                route.abort()
            else:
                route.continue_()
        self.context.route("**/*", only_local)
        self.page = self.context.new_page()
        self.page_errors = []
        self.page.on("pageerror", lambda err: self.page_errors.append(str(err)))
        self.page.goto(self.url, wait_until="load")

    def tearDown(self):
        try:
            self.assertEqual(self.network_hits, [])
            self.assertEqual(self.page_errors, [])
        finally:
            self.context.close()

    def fixture_click_for(self, scene, *, wrong=False):
        """TEST-only source DOM target; NEVER an actual model's oracle."""
        self.assertIn(scene, SCENES)
        _, lang, urgent, target = scene
        if lang == "ar":
            self.page.locator("#lang").click()
        if urgent:
            self.page.locator("#urgent").click()
        fixture_target = target
        if wrong:
            fixture_target = "C-117" if target == "C-214" else "C-214"
        control = self.page.locator(
            'article.visit[data-visit-id="' + fixture_target + '"] button.action'
        )
        self.assertEqual(control.count(), 1)
        box = control.bounding_box()
        self.assertIsNotNone(box)
        x, y = int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)
        self.assertTrue(0 <= x < WIDTH and 0 <= y < HEIGHT)
        return json.dumps({"actions": [{"kind": "click", "point": [x, y]}]},
                          separators=(",", ":"))

    def evaluate_fixed_test_action(self, scene, *, wrong=False):
        action = self.fixture_click_for(scene, wrong=wrong)
        calls = []
        def invented_predict(png, prompt):
            self.assertIsInstance(png, bytes)
            self.assertGreater(len(png), 200)
            self.assertEqual(prompt, source_task_prompt(exact_task(scene)))
            calls.append(1)
            return action
        result = run_original_clinicgrid_probe(
            self.browser, SOURCE, scene, invented_predict)
        self.assertEqual(calls, [1])
        self.assertFalse(result["real_model_inference_attested"])
        self.assertFalse(result["independent_blind_ood"])
        self.assertFalse(result["adapted_model_gain_proven"])
        self.assertEqual(result["all_source_visits_checked"], 6)
        return result

    def test_english_urgent_correct_native_click(self):
        result = self.evaluate_fixed_test_action(SCENES[0])
        self.assertTrue(result["native_action_dispatched"])
        self.assertTrue(result["firstparty_task_success"])

    def test_english_urgent_wrong_visit_is_real_failure(self):
        result = self.evaluate_fixed_test_action(SCENES[0], wrong=True)
        self.assertTrue(result["native_action_dispatched"])
        self.assertFalse(result["firstparty_task_success"])

    def test_arabic_rtl_correct_native_click(self):
        result = self.evaluate_fixed_test_action(SCENES[1])
        self.assertTrue(result["native_action_dispatched"])
        self.assertTrue(result["firstparty_task_success"])

    def test_arabic_rtl_wrong_visit_is_real_failure(self):
        result = self.evaluate_fixed_test_action(SCENES[1], wrong=True)
        self.assertTrue(result["native_action_dispatched"])
        self.assertFalse(result["firstparty_task_success"])

    def test_abstain_never_changes_source_visit_state(self):
        scene = SCENES[1]
        before = _all_visits_state(self.page)
        result = screenshot_locked_clinicgrid_turn(
            self.page, SOURCE, self.url, exact_task(scene),
            lambda png, prompt: '{"actions":[{"kind":"abstain"}]}')
        self.assertEqual(result["kind"], "abstain")
        self.assertFalse(result["executed"])
        self.assertEqual(_all_visits_state(self.page), before)

    def test_stale_picture_mutation_blocks_native_action(self):
        scene = SCENES[0]
        action = self.fixture_click_for(scene)
        before = _all_visits_state(self.page)
        def changed_scene(_png, _prompt):
            self.page.locator("#urgent").click()
            return action
        with self.assertRaisesRegex(ValueError, "stale screenshot"):
            screenshot_locked_clinicgrid_turn(
                self.page, SOURCE, self.url, exact_task(scene), changed_scene)
        self.assertEqual(_all_visits_state(self.page), before)

    def test_no_native_click_from_malformed_output(self):
        scene = SCENES[0]
        before = _all_visits_state(self.page)
        with self.assertRaises(ValueError):
            screenshot_locked_clinicgrid_turn(
                self.page, SOURCE, self.url, exact_task(scene),
                lambda png, prompt: '{"actions":[{"kind":"click","point":[1000,600]}]}')
        self.assertEqual(_all_visits_state(self.page), before)

    def test_filtered_scene_preserves_hidden_visit_states(self):
        self.page.locator("#urgent").click()
        before = _all_visits_state(self.page)
        self.assertEqual(len(before), 6)
        self.assertEqual(self.page.locator("article.visit").count(), 2)
        self.assertFalse(any(before.values()))


if __name__ == "__main__":
    unittest.main()
