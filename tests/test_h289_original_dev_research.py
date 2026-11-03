"""H289 source-only tests; actual browser tests are FIXED test oracle clicks, NOT Qwen."""
from __future__ import annotations

from hashlib import sha1
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from h199.native_action_sft import WIDTH, HEIGHT, source_task_prompt
from h289.routeboard_dev_research import (
    BLOB, ROLE, IDS, SCENES, checked_source, exact_task,
    run_original_dev_probe, _initial_dev_screenshot,
)

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "h289" / "routeboard.html"


class H289OriginalDevSafetyTests(unittest.TestCase):
    def test_authored_disjoint_source_and_registered_language_balance(self):
        self.assertEqual(BLOB, "6b7486384c935ff6ea856a0eefeb213a9504fdf2")
        source, url = checked_source(HTML)
        self.assertEqual(source, HTML)
        self.assertTrue(url.startswith("file:"))
        self.assertEqual(len(IDS), 6)
        self.assertEqual(len(set(IDS)), 6)
        self.assertTrue(set(IDS).isdisjoint({
            "S-401", "S-415", "S-428", "S-439",
            "S-452", "S-467", "S-480", "S-493",
        }))
        self.assertEqual(len(SCENES), 12)
        self.assertEqual(len(set(SCENES)), 12)
        self.assertEqual(sum(s[1] == "en" for s in SCENES), 6)
        self.assertEqual(sum(s[1] == "ar" for s in SCENES), 6)
        self.assertTrue(all(s[-1] in IDS for s in SCENES))

    def test_source_tamper_rejected_and_no_train_dev_promotion(self):
        self.assertIn("NOT_BLIND_OOD", ROLE)
        with tempfile.TemporaryDirectory() as scratch:
            parent = Path(scratch) / "h289"
            parent.mkdir()
            copy = parent / "routeboard.html"
            copy.write_bytes(HTML.read_bytes() + b"<!-- changed -->")
            with self.assertRaisesRegex(ValueError, "byte drift"):
                checked_source(copy)
        for illegal in ("TRAIN", "DEV_ADMITTED", "OOD", "BLIND", "RELEASE"):
            with self.subTest(role=illegal), self.assertRaises(ValueError):
                run_original_dev_probe(
                    None, HTML, SCENES[0],
                    lambda png, task: "not-a-model",
                    role=illegal)

    def test_only_exact_predeclared_bilingual_task(self):
        self.assertEqual(exact_task(SCENES[0]),
                         "Assign fictional route R-602")
        self.assertEqual(exact_task(SCENES[6]),
                         "وجّه المسار الخيالي R-602")
        with self.assertRaises(ValueError):
            exact_task(("unregistered", "en", 0, True, False, "R-602"))
        with self.assertRaises(ValueError):
            run_original_dev_probe(None, HTML, SCENES[0], None, role=ROLE)


class H289InitialCaptureTransportTests(unittest.TestCase):
    """Synthetic-only fail-closed one-retry coverage; no browser/model required."""

    @staticmethod
    def fixture():
        _, url = checked_source(HTML)
        return SimpleNamespace(
            url=url, viewport_size={"width": WIDTH, "height": HEIGHT}
        ), url

    @staticmethod
    def capture_error(message="Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot"):
        fake_error = type("Error", (Exception,), {
            "__module__": "playwright._impl._errors"
        })
        return fake_error(message)

    def test_exact_transient_capture_retries_once_before_policy(self):
        page, url = self.fixture()
        with mock.patch("h289.routeboard_dev_research._observed_png",
                        side_effect=[self.capture_error(), b"valid-stub"]) as cap:
            self.assertEqual(_initial_dev_screenshot(page, HTML, url), b"valid-stub")
            self.assertEqual(cap.call_count, 2)

    def test_two_exact_capture_failures_raise_without_third_attempt(self):
        page, url = self.fixture()
        with mock.patch("h289.routeboard_dev_research._observed_png",
                        side_effect=[self.capture_error(), self.capture_error()]) as cap:
            with self.assertRaisesRegex(Exception, "Unable to capture screenshot"):
                _initial_dev_screenshot(page, HTML, url)
            self.assertEqual(cap.call_count, 2)

    def test_unrecognized_screenshot_failure_does_not_retry(self):
        page, url = self.fixture()
        with mock.patch("h289.routeboard_dev_research._observed_png",
                        side_effect=self.capture_error("Page.screenshot: Target closed")) as cap:
            with self.assertRaisesRegex(Exception, "Target closed"):
                _initial_dev_screenshot(page, HTML, url)
            self.assertEqual(cap.call_count, 1)

    def test_page_drift_after_failure_refuses_before_second_capture(self):
        page, url = self.fixture()
        def malicious_capture(_page):
            page.url = "file:///unapproved"
            raise self.capture_error()
        with mock.patch("h289.routeboard_dev_research._observed_png",
                        side_effect=malicious_capture) as cap:
            with self.assertRaisesRegex(ValueError, "source/page"):
                _initial_dev_screenshot(page, HTML, url)
            self.assertEqual(cap.call_count, 1)

    def test_mismatched_source_rejected_during_retry(self):
        page, url = self.fixture()
        with mock.patch("h289.routeboard_dev_research._observed_png",
                        side_effect=self.capture_error()) as cap:
            with self.assertRaisesRegex(ValueError, "source"):
                _initial_dev_screenshot(page, HTML.with_name("other.html"), url)
            self.assertEqual(cap.call_count, 1)


@unittest.skipUnless(os.environ.get("HALA_H289_BROWSER_QUALIFY") == "1",
                     "explicit original DEV-only real browser opt-in required")
class H289OriginalRealBrowserDevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            ch = os.environ.get("HALA_H289_BROWSER_CHANNEL", "chromium")
            if ch not in ("chromium", "msedge"):
                raise ValueError("H289 browser channel unauthorized")
            args = ({"headless": True, "channel": "msedge"}
                    if ch == "msedge" else
                    {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.pw.chromium.launch(**args)
        except BaseException:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_all_12_real_original_source_actions_dev_only(self):
        seen = []
        for scene in SCENES:
            def fixture_policy(png, prompt):
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(prompt, source_task_prompt(exact_task(scene)))
                # TEST-ONLY original DOM oracle: must never be passed to model.
                page = self.browser.contexts[-1].pages[-1]
                target = scene[-1]
                button = page.locator(
                    'article.row[data-route-id="' + target + '"] button')
                bounds = button.bounding_box()
                self.assertIsNotNone(bounds)
                xy = [int(bounds["x"] + bounds["width"] / 2),
                      int(bounds["y"] + bounds["height"] / 2)]
                return json.dumps({"actions": [
                    {"kind": "click", "point": xy}
                ]}, separators=(",", ":"))
            r = run_original_dev_probe(
                self.browser, HTML, scene, fixture_policy, role=ROLE)
            self.assertTrue(r["native_action_dispatched"])
            self.assertTrue(r["firstparty_task_success"])
            self.assertEqual(r["all_six_source_ids_checked"], 6)
            self.assertTrue(r["screenshot_locked"])
            self.assertFalse(r["formal_train_dev_admitted"])
            self.assertFalse(r["independent_blind_ood"])
            self.assertFalse(r["actual_model_generation_proven"])
            seen.append(r["scene"])
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(set(seen)), 12)

    def test_wrong_neighbor_native_click_is_real_but_task_failure(self):
        scene = SCENES[0]
        def wrong_fixture(png, prompt):
            page = self.browser.contexts[-1].pages[-1]
            button = page.locator(
                'article.row[data-route-id="R-625"] button')
            b = button.bounding_box()
            return json.dumps({"actions": [{"kind": "click", "point": [
                int(b["x"] + b["width"] / 2),
                int(b["y"] + b["height"] / 2),
            ]}]}, separators=(",", ":"))
        r = run_original_dev_probe(
            self.browser, HTML, scene, wrong_fixture, role=ROLE)
        self.assertTrue(r["native_action_dispatched"])
        self.assertFalse(r["firstparty_task_success"])

    def test_abstention_never_counts_as_task_success(self):
        r = run_original_dev_probe(
            self.browser, HTML, SCENES[1],
            lambda png, prompt: '{"actions":[{"kind":"abstain"}]}',
            role=ROLE)
        self.assertEqual(r["native_kind"], "abstain")
        self.assertFalse(r["firstparty_task_success"])

    def test_screenshot_mutation_during_prediction_rejected(self):
        def malicious_callback(png, task):
            page = self.browser.contexts[-1].pages[-1]
            page.evaluate("() => document.body.style.opacity='0.7'")
            return '{"actions":[{"kind":"abstain"}]}'
        with self.assertRaisesRegex(ValueError, "screenshot stale"):
            run_original_dev_probe(
                self.browser, HTML, SCENES[2], malicious_callback,
                role=ROLE)


if __name__ == "__main__":
    unittest.main()
