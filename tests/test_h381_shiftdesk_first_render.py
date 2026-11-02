"""H381 original ShiftDesk readiness: fake source-only tests, no TRAIN/OOD.

No browser pixels, source gold or labels are produced by these tests.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h285.train_only_browser_candidates import (
    SCENES, _open_scene, await_shiftdesk_first_render,
)


class FakePage:
    url = "file:///public-shiftdesk"
    viewport_size = {"width": 960, "height": 720}

    def __init__(self, readiness=True):
        self.readiness = readiness
        self.events = []

    def goto(self, url, *, wait_until):
        self.events.append(("goto", url, wait_until))

    def evaluate(self, js, *args):
        if "configureShiftDesk" in js:
            self.events.append(("configure", args))
            return None
        self.events.append(("first_render", js))
        return self.readiness

    def locator(self, selector):
        if selector not in ("article.card", "html"):
            raise AssertionError("unexpected fake source locator")
        self.events.append(("locator", selector))
        return FakeLocator(selector)


class FakeLocator:
    def __init__(self, selector):
        self.selector = selector

    def count(self):
        return 8 if self.selector == "article.card" else 0

    def get_attribute(self, name):
        if name != "dir" or self.selector != "html":
            raise AssertionError("unexpected attribute request")
        return "ltr"


class FakeContext:
    def __init__(self, page):
        self.page = page
        self.closed = False
        self.routes = 0

    def route(self, glob, callback):
        if glob != "**/*" or not callable(callback):
            raise AssertionError("untrusted route")
        self.routes += 1

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class H381FirstRenderTests(unittest.TestCase):
    def test_single_bounded_double_raf_with_source_eight_cards(self):
        page = FakePage()
        await_shiftdesk_first_render(page)
        self.assertEqual(len(page.events), 1)
        event, js = page.events[0]
        self.assertEqual(event, "first_render")
        self.assertEqual(js.count("requestAnimationFrame("), 2)
        self.assertIn("setTimeout(() => finish(false), 3000)", js)
        self.assertIn("document.readyState === 'complete'", js)
        self.assertIn("document.querySelectorAll('article.card').length === 8", js)
        self.assertNotIn("screenshot(", js)
        self.assertNotIn("click(", js)

    def test_timeout_and_nonliteral_success_fail_closed(self):
        for value in (False, None, 1, "true", {}, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "first render not ready"):
                    await_shiftdesk_first_render(FakePage(value))

    def test_browser_exception_does_not_trigger_retry(self):
        class ErrorPage:
            count = 0

            def evaluate(self, js):
                self.count += 1
                raise RuntimeError("synthetic browser context closed")

        page = ErrorPage()
        with self.assertRaisesRegex(RuntimeError, "context closed"):
            await_shiftdesk_first_render(page)
        self.assertEqual(page.count, 1)

    def test_original_scene_waits_after_checks_then_yields(self):
        page = FakePage(True)
        context = FakeContext(page)
        with patch("h285.train_only_browser_candidates.new_offline_context",
                   return_value=context):
            with _open_scene(object(), object(), page.url, SCENES[0]) as actual:
                self.assertIs(actual, page)
                self.assertEqual(context.routes, 1)
                actions = [x[0] for x in page.events]
                self.assertEqual(actions.count("first_render"), 1)
                self.assertLess(actions.index("configure"),
                                actions.index("first_render"))
        self.assertTrue(context.closed)

    def test_failed_render_never_yields_and_closes_context(self):
        page = FakePage(False)
        context = FakeContext(page)
        with patch("h285.train_only_browser_candidates.new_offline_context",
                   return_value=context):
            with self.assertRaisesRegex(ValueError, "first render not ready"):
                with _open_scene(object(), object(), page.url, SCENES[0]):
                    self.fail("refused first frame cannot yield to screenshot")
        self.assertTrue(context.closed)


if __name__ == "__main__":
    unittest.main()
