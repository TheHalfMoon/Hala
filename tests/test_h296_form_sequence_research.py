"""H296 original bilingual offline form browser acts: FIXTURE, not Qwen."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import source_task_prompt
from h296.fieldport_sequence_research import (
    BLOB, IDS, ROLE, SCENES, checked_source, exact_task,
    run_original_form_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "h296" / "fieldport.html"


class H296AuthorOwnedSourceTests(unittest.TestCase):
    def test_source_pin_and_balanced_original_bilingual_sequences(self):
        self.assertEqual(BLOB, "52135ef3f3a68922b82be7bcb95a337a89982b18")
        path, url = checked_source(HTML)
        self.assertEqual(path, HTML)
        self.assertTrue(url.startswith("file:"))
        self.assertEqual(len(IDS), len(set(IDS)))
        self.assertEqual(len(SCENES), 12)
        self.assertEqual(len(set(SCENES)), 12)
        self.assertEqual(sum(s[1] == "en" for s in SCENES), 6)
        self.assertEqual(sum(s[1] == "ar" for s in SCENES), 6)
        self.assertEqual({s[2] for s in SCENES}, {0, 1, 2})
        self.assertEqual({s[3] for s in SCENES}, {True, False})
        for s in SCENES:
            self.assertIn(s[4], range(4))
            self.assertEqual(len(s[5]), 6)
            self.assertIn("FIC-", exact_task(s))
            self.assertNotIn("model", exact_task(s).lower())

    def test_original_source_byte_tamper_and_symbolic_drift_refused(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "h296"
            d.mkdir()
            p = d / "fieldport.html"
            p.write_bytes(HTML.read_bytes() + b"<!-- forged -->")
            with self.assertRaisesRegex(ValueError, "Git blob changed"):
                checked_source(p)
        with self.assertRaises(ValueError):
            checked_source(ROOT / "h289" / "routeboard.html")

    def test_role_and_task_not_promoted_to_train_or_blind(self):
        self.assertIn("NOT_ADMITTED_NOT_OOD", ROLE)
        for role in ("TRAIN", "DEV_ADMITTED", "BLIND_OOD", "RELEASE"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                run_original_form_sequence(
                    None, HTML, SCENES[0], lambda png, task: "none",
                    role=role)
        with self.assertRaises(ValueError):
            exact_task(("invented", "en", 0, False, 0, "FIC-00"))
        with self.assertRaises(ValueError):
            run_original_form_sequence(None, HTML, SCENES[0], None, role=ROLE)


@unittest.skipUnless(os.environ.get("HALA_H296_BROWSER_QUALIFY") == "1",
                     "explicit original offline browser qualifier opt-in required")
class H296RealOriginalNativeFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H296_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("H296 unsupported browser")
            kwargs = ({"channel": "msedge", "headless": True}
                      if channel == "msedge" else
                      {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.pw.chromium.launch(**kwargs)
        except BaseException:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    @staticmethod
    def action_click(scene, *, wrong=False):
        page = H296RealOriginalNativeFormTests.browser.contexts[-1].pages[-1]
        target = IDS[(scene[4] + 1) % 4] if wrong else IDS[scene[4]]
        element = page.locator('input[data-field-id="' + target + '"]')
        bounds = element.bounding_box()
        if bounds is None:
            raise ValueError("test only expected visible ORIGINAL input")
        point = [int(bounds["x"] + bounds["width"]/2),
                 int(bounds["y"] + bounds["height"]/2)]
        return json.dumps({"actions": [{"kind": "click", "point": point}]},
                          separators=(",", ":"))

    def test_all_12_real_two_step_focus_and_type_source_states(self):
        ids = []
        first_sha = set()
        for scene in SCENES:
            called = []
            def fixture(png, prompt):
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(prompt, source_task_prompt(exact_task(scene)))
                self.assertLess(len(called), 2)
                if not called:
                    from hashlib import sha256
                    first_sha.add(sha256(png).hexdigest())
                called.append(True)
                if len(called) == 1:
                    return self.action_click(scene)
                return json.dumps({"actions": [
                    {"kind": "type_text", "text": scene[5]}
                ]}, separators=(",", ":"))
            output = run_original_form_sequence(
                self.browser, HTML, scene, fixture, role=ROLE)
            self.assertEqual(len(called), 2)
            self.assertEqual(output["native_actions"], 2)
            self.assertTrue(output["focus_click_dispatched"])
            self.assertTrue(output["type_dispatched"])
            self.assertTrue(output["source_task_success"])
            self.assertTrue(output["screen_lock_each_step"])
            self.assertEqual(output["all_four_fields_checked"], 4)
            self.assertFalse(output["actual_neural_generation_proven"])
            self.assertFalse(output["official_train_dev_admitted"])
            self.assertFalse(output["independent_blind_ood"])
            ids.append(output["scene"])
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(len(first_sha), 12)

    def test_wrong_neighbor_focus_then_type_dispatch_is_task_failure(self):
        scene = SCENES[0]
        calls = []
        def wrong(png, task):
            calls.append(1)
            if len(calls) == 1:
                return self.action_click(scene, wrong=True)
            return json.dumps({"actions": [
                {"kind": "type_text", "text": scene[5]}
            ]})
        result = run_original_form_sequence(
            self.browser, HTML, scene, wrong, role=ROLE)
        self.assertEqual(result["native_actions"], 2)
        self.assertTrue(result["focus_click_dispatched"])
        self.assertTrue(result["type_dispatched"])
        self.assertFalse(result["source_task_success"])

    def test_wrong_code_still_type_dispatches_but_fails_task(self):
        scene = SCENES[1]
        calls = []
        def wrong(png, task):
            calls.append(1)
            if len(calls) == 1:
                return self.action_click(scene)
            return '{"actions":[{"kind":"type_text","text":"FIC-99"}]}'
        result = run_original_form_sequence(
            self.browser, HTML, scene, wrong, role=ROLE)
        self.assertTrue(result["type_dispatched"])
        self.assertFalse(result["source_task_success"])

    def test_abstain_at_first_or_second_step_never_fabricates_success(self):
        scene = SCENES[2]
        for abstain_step in (1, 2):
            called = []
            def policy(png, task):
                called.append(1)
                if len(called) == abstain_step:
                    return '{"actions":[{"kind":"abstain"}]}'
                return self.action_click(scene)
            result = run_original_form_sequence(
                self.browser, HTML, scene, policy, role=ROLE)
            self.assertEqual(result["native_actions"], abstain_step)
            self.assertFalse(result["source_task_success"])
            self.assertFalse(result["type_dispatched"])
            self.assertEqual(len(called), abstain_step)

    def test_type_text_without_focused_field_fails_closed(self):
        with self.assertRaises(ValueError):
            run_original_form_sequence(
                self.browser, HTML, SCENES[3],
                lambda png, prompt: '{"actions":[{"kind":"type_text","text":"FIC-14"}]}',
                role=ROLE)

    def test_screenshot_mutation_during_policy_is_refused(self):
        scene = SCENES[4]
        def mutated(png, task):
            page = self.browser.contexts[-1].pages[-1]
            page.evaluate("() => document.body.style.opacity='0.5'")
            return self.action_click(scene)
        with self.assertRaisesRegex(ValueError, "screenshot changed"):
            run_original_form_sequence(
                self.browser, HTML, scene, mutated, role=ROLE)

    def test_first_step_non_focus_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first native step"):
            run_original_form_sequence(
                self.browser, HTML, SCENES[5],
                lambda png, task: '{"actions":[{"kind":"scroll","point":[300,300],"dy":240}]}',
                role=ROLE)


if __name__ == "__main__":
    unittest.main()
