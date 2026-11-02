"""H379: invented-only browser first-paint readiness safety for RailKit.

No screenshots, raw error logs, action gold, model weights, legal TRAIN or OOD.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h325.two_step_train_candidates import (
    SCENES, await_railkit_first_paint, open_original,
)


class FakeRailKitPage:
    url = "file:///public-railkit"

    viewport_size = {"width": 960, "height": 720}

    def __init__(self, result=True):
        self.result = result
        self.calls = []
        self.loaded = False

    def goto(self, url, *, wait_until):
        self.calls.append(("goto", url, wait_until))
        self.loaded = True

    def evaluate(self, script, *args):
        if "configureRailKit" in script:
            self.calls.append(("configure", args))
            return None
        self.calls.append(("first_paint", script))
        return self.result

    def locator(self, selector):
        assert selector in {"html", "button[data-route]"}
        self.calls.append(("locator", selector))
        return self

    def get_attribute(self, name):
        assert name == "dir"
        return "ltr"

    def count(self):
        return 4


class FakeContext:
    def __init__(self, page):
        self.page = page
        self.closed = False
        self.routes = 0

    def route(self, _glob, _callback):
        self.routes += 1

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class H379FirstPaintTests(unittest.TestCase):
    def test_exact_two_raf_callbacks_one_call_and_timeout(self):
        page = FakeRailKitPage()
        self.assertIsNone(await_railkit_first_paint(page))
        self.assertEqual(len(page.calls), 1)
        name, js = page.calls[0]
        self.assertEqual(name, "first_paint")
        self.assertEqual(js.count("requestAnimationFrame("), 2)
        self.assertIn("setTimeout(() => finish(false), 3000)", js)
        self.assertIn("document.readyState === 'complete'", js)
        self.assertIn("document.querySelectorAll('button[data-route]').length === 4", js)
        self.assertNotIn("screenshot(", js)
        self.assertNotIn("click(", js)

    def test_fail_closed_timeout_and_nonliteral_boolean(self):
        for result in (False, None, 1, "true", [], {}):
            with self.subTest(result=result):
                with self.assertRaisesRegex(
                    ValueError, "^H325 ORIGINAL_TRAIN_CANDIDATE_REFUSED$"
                ):
                    await_railkit_first_paint(FakeRailKitPage(result))

    def test_no_exception_retry_and_no_raw_payload_logging(self):
        class FailingPage:
            calls = 0

            def evaluate(self, _):
                self.calls += 1
                raise TimeoutError("PRIVATE CANDIDATE SCREENSHOT")
        page = FailingPage()
        with self.assertRaises(TimeoutError):
            await_railkit_first_paint(page)
        self.assertEqual(page.calls, 1)

    def test_source_scene_uses_barrier_after_original_checks(self):
        page = FakeRailKitPage()
        context = FakeContext(page)
        with patch("h325.two_step_train_candidates.new_offline_context",
                   return_value=context):
            with open_original(object(), page.url, SCENES[0]) as actual:
                self.assertIs(actual, page)
                self.assertEqual(context.routes, 1)
                names = [name for name, *_ in page.calls]
                self.assertEqual(names.count("first_paint"), 1)
                self.assertLess(names.index("configure"), names.index("first_paint"))
        self.assertTrue(context.closed)

    def test_failed_barrier_closes_context_without_yielding_page(self):
        page = FakeRailKitPage(False)
        context = FakeContext(page)
        with patch("h325.two_step_train_candidates.new_offline_context",
                   return_value=context):
            with self.assertRaisesRegex(
                ValueError, "^H325 ORIGINAL_TRAIN_CANDIDATE_REFUSED$"
            ):
                with open_original(object(), page.url, SCENES[0]):
                    self.fail("Fail-closed source gate MUST not yield page")
        self.assertTrue(context.closed)


if __name__ == "__main__":
    unittest.main()
