"""H319: original public offline GUI, genuine browser actuator, fake callbacks only.

No pretrained Qwen is used here; all callback actions are test-owned and do
not prove any model performance, data admission or independent blind OOD.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import source_task_prompt
from h319.beaconboard_dynamic_research import (
    BLOB, IDS, ROLE, SCENES, checked_source, exact_task,
    run_original_dynamic_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h319" / "beaconboard.html"


class H319SourceContract(unittest.TestCase):
    def test_immutable_first_party_source_and_scene_design(self):
        self.assertEqual(BLOB, "6e8467e123c7f0fbac77ea553622ab8e659f9c16")
        path, url = checked_source(SOURCE)
        self.assertEqual(path, SOURCE)
        self.assertTrue(url.startswith("file:"))
        self.assertEqual(len(SCENES), 12)
        self.assertEqual(len(set(SCENES)), 12)
        self.assertEqual(sum(s[1] == "en" for s in SCENES), 6)
        self.assertEqual(sum(s[1] == "ar" for s in SCENES), 6)
        self.assertEqual({s[2] for s in SCENES}, {0, 1, 2})
        self.assertEqual({s[3] for s in SCENES}, {False, True})
        self.assertEqual({s[4] for s in SCENES}, set(range(4)))
        for case in SCENES:
            self.assertEqual(len(case[5]), 6)
            self.assertIn(case[5], exact_task(case))
            self.assertIn(IDS[case[4]], exact_task(case))

    def test_source_tamper_rejected(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "h319"
            directory.mkdir()
            forged = directory / "beaconboard.html"
            forged.write_bytes(SOURCE.read_bytes() + b"\n<!-- altered -->")
            with self.assertRaises(ValueError):
                checked_source(forged)

    def test_no_train_or_blind_role_promotion(self):
        self.assertIn("NOT_ADMITTED_NOT_OOD", ROLE)
        for role in ("TRAIN", "ADMITTED_DEV", "BLIND_OOD", "RELEASE"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                run_original_dynamic_sequence(
                    None, SOURCE, SCENES[0], lambda screen, task: "none",
                    role=role)
        with self.assertRaises(ValueError):
            exact_task(("invented", "en", 0, False, 0, "SIM-11"))


@unittest.skipUnless(os.environ.get("HALA_H319_BROWSER_QUALIFY") == "1",
                     "explicit H319 original offline browser qualifier required")
class H319BrowserContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.playwright = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H319_BROWSER_CHANNEL", "chromium")
            if channel not in {"chromium", "msedge"}:
                raise ValueError("unsupported original browser")
            kwargs = ({"channel": "msedge", "headless": True}
                      if channel == "msedge" else
                      {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.playwright.chromium.launch(**kwargs)
        except BaseException:
            cls.playwright.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    @classmethod
    def button_action(cls, scene, *, wrong=False):
        page = cls.browser.contexts[-1].pages[-1]
        index = (scene[4] + 1) % 4 if wrong else scene[4]
        element = page.locator(
            'button[data-station="' + IDS[index] + '"]')
        bounds = element.bounding_box()
        if bounds is None:
            raise AssertionError("original button not visible")
        x = int(bounds["x"] + bounds["width"] / 2)
        y = int(bounds["y"] + bounds["height"] / 2)
        return json.dumps({"actions": [{
            "kind": "click", "point": [x, y]}]}, separators=(",", ":"))

    def test_all_12_source_native_dynamic_sequences(self):
        for scene in SCENES:
            with self.subTest(scene=scene[0]):
                calls = []
                def fake_fixture(png, task):
                    self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertEqual(task, source_task_prompt(exact_task(scene)))
                    calls.append(1)
                    if len(calls) == 1:
                        return self.button_action(scene)
                    return json.dumps({"actions": [
                        {"kind": "type_text", "text": scene[5]}
                    ]}, separators=(",", ":"))
                result = run_original_dynamic_sequence(
                    self.browser, SOURCE, scene, fake_fixture, role=ROLE)
                self.assertEqual(len(calls), 2)
                self.assertTrue(result["original_four_station_task_success"])
                self.assertEqual(result["native_dispatched_actions"], 2)
                self.assertEqual(result["actions_observed"], 2)
                self.assertFalse(result["model_generations_proven"])
                self.assertFalse(result["official_train_admitted"])
                self.assertFalse(result["independent_blind_ood"])
                self.assertFalse(result["can_release"])

    def test_wrong_button_stops_before_second_model_callback(self):
        for scene in (SCENES[0], SCENES[8]):
            called = []
            def fixture(png, task):
                called.append(1)
                if len(called) > 1:
                    self.fail("second callback after wrong source-owned button")
                return self.button_action(scene, wrong=True)
            result = run_original_dynamic_sequence(
                self.browser, SOURCE, scene, fixture, role=ROLE)
            self.assertEqual(len(called), 1)
            self.assertFalse(result["original_four_station_task_success"])
            self.assertEqual(result["native_dispatched_actions"], 1)
            self.assertEqual(result["categorical_failure"], "wrong_focus_or_state")

    def test_wrong_code_never_dispatches_native_typing(self):
        for scene in (SCENES[1], SCENES[9]):
            called = []
            def fixture(png, task):
                called.append(1)
                return (self.button_action(scene) if len(called) == 1 else
                        '{"actions":[{"kind":"type_text","text":"SIM-99"}]}')
            result = run_original_dynamic_sequence(
                self.browser, SOURCE, scene, fixture, role=ROLE)
            self.assertEqual(len(called), 2)
            self.assertEqual(result["native_dispatched_actions"], 1)
            self.assertEqual(result["categorical_failure"], "wrong_code")
            self.assertFalse(result["original_four_station_task_success"])

    def test_abstain_first_step_refuses(self):
        scene = SCENES[2]
        result = run_original_dynamic_sequence(
            self.browser, SOURCE, scene,
            lambda screen, task: '{"actions":[{"kind":"abstain"}]}',
            role=ROLE)
        self.assertEqual(result["actions_observed"], 1)
        self.assertEqual(result["native_dispatched_actions"], 0)
        self.assertFalse(result["original_four_station_task_success"])

    def test_second_step_abstain_refuses(self):
        scene = SCENES[3]
        seen = []
        def fixture(png, task):
            seen.append(1)
            return (self.button_action(scene) if len(seen) == 1 else
                    '{"actions":[{"kind":"abstain"}]}')
        result = run_original_dynamic_sequence(
            self.browser, SOURCE, scene, fixture, role=ROLE)
        self.assertEqual(result["native_dispatched_actions"], 1)
        self.assertEqual(result["actions_observed"], 2)
        self.assertFalse(result["original_four_station_task_success"])

    def test_source_mutation_during_policy_call_fails_closed(self):
        scene = SCENES[0]
        def hostile(screen, task):
            page = self.browser.contexts[-1].pages[-1]
            page.evaluate("() => document.querySelector('h1').textContent='tampered'")
            return self.button_action(scene)
        with self.assertRaisesRegex(ValueError, "DYNAMIC_SOURCE_OR_POLICY_REFUSED"):
            run_original_dynamic_sequence(
                self.browser, SOURCE, scene, hostile, role=ROLE)


if __name__ == "__main__":
    unittest.main()
