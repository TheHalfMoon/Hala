"""H375: invented-only first-paint barrier contract and fail-closed tests.

No images, browser calls, private source cases, model or sealed benchmark.
Actual H319/H337 original-browser QA runs in separate inherited CI jobs.
"""
from __future__ import annotations

import unittest

from h319.beaconboard_dynamic_research import await_initial_render


class H375PaintBarrierContract(unittest.TestCase):
    def test_exact_one_browser_render_wait_and_no_screenshot_retries(self):
        class FakePage:
            def __init__(self):
                self.calls = []
            def evaluate(self, script):
                self.calls.append(script)
                return True
        page = FakePage()
        self.assertIsNone(await_initial_render(page))
        self.assertEqual(len(page.calls), 1)
        script = page.calls[0]
        self.assertEqual(script.count("requestAnimationFrame("), 2)
        self.assertIn("setTimeout(() => finish(false), 3000)", script)
        self.assertIn("document.readyState === 'complete'", script)
        self.assertIn("document.querySelectorAll('button[data-station]').length === 4", script)
        self.assertNotIn("screenshot", script)
        self.assertNotIn("document.body.innerHTML =", script)

    def test_denies_missing_or_nonboolean_render_ack(self):
        class Page:
            def __init__(self, result):
                self.result = result
            def evaluate(self, _):
                return self.result
        for result in (None, False, 1, 0, "true", [], {}):
            with self.subTest(result=result), self.assertRaisesRegex(
                ValueError, "^H319 PUBLIC_DYNAMIC_RESEARCH_REFUSED$"
            ):
                await_initial_render(Page(result))

    def test_browser_error_surfaces_without_retry_or_payload_logging(self):
        class Page:
            calls = 0
            def evaluate(self, _):
                self.calls += 1
                raise TimeoutError("PRIVATE SOURCE NO PHOTO")
        page = Page()
        with self.assertRaises(TimeoutError):
            await_initial_render(page)
        self.assertEqual(page.calls, 1)


if __name__ == "__main__":
    unittest.main()
