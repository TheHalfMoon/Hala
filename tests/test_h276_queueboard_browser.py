"""H276 real Chromium qualification of ORIGINAL offline Queueboard.

No GUI-agent model involved. Read only the committed H275 first-party
file. These tests use an isolated browser and invented tickets only.
Screenshots exist in memory for pixel-stability checks and are never
saved, uploaded, logged or published.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import struct
import unittest

try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    # The broad Hala CPU job intentionally has no browser dependency.
    # Dedicated H276 browser CI explicitly installs Playwright and
    # refuses a skipped test run.
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "h275" / "queueboard.html"
TICKET_IDS = ("Q-104", "Q-118", "Q-126", "Q-137", "Q-142", "Q-153")
WIDTH, HEIGHT = 960, 720


@unittest.skipIf(sync_playwright is None, "optional Playwright; dedicated H276 workflow required")
class QueueboardOriginalBrowserQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not HTML.is_file() or HTML.is_symlink():
            raise AssertionError("H276 expected original committed H275 HTML file")
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
        except BaseException:
            cls.playwright.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.browser.close()
        finally:
            cls.playwright.stop()
        super().tearDownClass()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            locale="en-US",
            color_scheme="light",
            reduced_motion="reduce",
            java_script_enabled=True,
            service_workers="block",
        )
        self.denied_network = []
        self.page_errors = []

        def offline_only(route):
            url = route.request.url
            if url.startswith(("http:", "https:", "ws:", "wss:")):
                self.denied_network.append("external request blocked")
                route.abort()
            else:
                route.continue_()

        self.context.route("**/*", offline_only)
        self.page = self.context.new_page()
        self.page.set_default_timeout(5000)
        self.page.on("pageerror", lambda err: self.page_errors.append(str(err)))
        self.page.goto(HTML.as_uri(), wait_until="load")
        self.assertEqual(self.page.url, HTML.as_uri())
        self.assertEqual(self.page.viewport_size, {"width": WIDTH, "height": HEIGHT})
        self.assertEqual(self.page.locator("#queueboard").get_attribute(
            "data-source-family"), "hala-original-queueboard-h275")
        self.assertEqual(self.page.locator(".ticket-row").count(), 6)

    def tearDown(self):
        try:
            self.assertFalse(self.denied_network, "Queueboard attempted network")
            self.assertEqual(self.page_errors, [], "Queueboard JS failed")
        finally:
            self.context.close()

    def row(self, ticket):
        if ticket not in TICKET_IDS:
            raise ValueError("H276 does not accept arbitrary ticket IDs")
        return self.page.locator(f'.ticket-row[data-ticket-id="{ticket}"]')

    def button(self, ticket):
        return self.page.locator(
            f'button[data-action="complete"][data-ticket-id="{ticket}"]'
        )

    def png_digest(self):
        # Screenshot pixels only in RAM; dimensions verified from PNG header.
        png = self.page.screenshot(
            type="png", animations="disabled", full_page=False
        )
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", png[16:24]), (WIDTH, HEIGHT))
        self.assertGreater(len(png), 2000)
        digest = sha256(png).digest()
        del png
        return digest

    def assert_control_on_screen(self, button):
        box = button.bounding_box()
        self.assertIsNotNone(box, "Synthetic ticket button not rendered")
        self.assertGreaterEqual(box["x"], 0)
        self.assertGreaterEqual(box["y"], 0)
        self.assertGreaterEqual(box["width"], 24)
        self.assertGreaterEqual(box["height"], 20)
        self.assertLessEqual(box["x"] + box["width"], WIDTH)
        self.assertLessEqual(box["y"] + box["height"], HEIGHT)

    def test_en_six_distinct_visible_targets_and_stable_screenshot(self):
        self.assertEqual(self.page.locator("html").get_attribute("dir"), "ltr")
        self.assertEqual(self.page.locator("html").get_attribute("lang"), "en")
        self.assertEqual(self.page.locator("#total-count").inner_text(), "6")
        self.assertEqual(self.page.locator("#urgent-count").inner_text(), "2")
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "0")
        self.assertEqual(
            self.page.locator(".ticket-row").evaluate_all(
                "(rows) => rows.map(row => row.dataset.ticketId)"
            ),
            list(TICKET_IDS),
        )
        for ticket in TICKET_IDS:
            with self.subTest(ticket=ticket):
                self.assert_control_on_screen(self.button(ticket))
        one = self.png_digest()
        self.page.wait_for_timeout(200)
        self.assertEqual(one, self.png_digest(),
                         "Original idle page bytes drifted over time")

    def test_native_complete_is_exactly_one_row_not_neighbor(self):
        target = "Q-137"
        self.assert_control_on_screen(self.button(target))
        self.button(target).click()
        self.assertEqual(self.row(target).get_attribute("data-status"), "done")
        self.assertTrue(self.button(target).is_disabled())
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "1")
        self.assertEqual(self.page.locator("#urgent-count").inner_text(), "1")
        for ticket in TICKET_IDS:
            if ticket != target:
                with self.subTest(unmodified=ticket):
                    self.assertEqual(
                        self.row(ticket).get_attribute("data-status"), "waiting"
                    )
                    self.assertTrue(self.button(ticket).is_enabled())
        # The button cannot count twice after it becomes disabled.
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "1")

    def test_priority_and_search_state_stable_without_network(self):
        self.page.locator("#priority").click()
        self.assertEqual(self.page.locator("#priority").get_attribute(
            "aria-pressed"), "true")
        self.assertEqual(
            self.page.locator(".ticket-row").evaluate_all(
                "(rows) => rows.map(row => row.dataset.ticketId)"
            ),
            ["Q-118", "Q-137"],
        )
        self.page.locator("#search").fill("Q-137")
        self.assertEqual(self.page.locator(".ticket-row").count(), 1)
        self.assertEqual(self.row("Q-137").count(), 1)
        # A blinking caret must not invalidate a deterministic screenshot
        # once user entry has ended. This changes NO target/label/DOM oracle.
        self.page.locator("#search").evaluate("(el) => el.blur()")
        a = self.png_digest()
        self.page.wait_for_timeout(250)
        self.assertEqual(a, self.png_digest(),
                         "Blurred first-party search screenshot drift")
        self.button("Q-137").click()
        self.assertEqual(self.row("Q-137").get_attribute("data-status"), "done")
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "1")
        self.page.locator("#all").click()
        self.page.locator("#search").fill("")
        self.page.locator("#search").evaluate("(el) => el.blur()")
        self.assertEqual(self.page.locator(".ticket-row").count(), 6)

    def test_rtl_arabic_target_and_reset_are_real_native_actions(self):
        self.page.locator("#lang").click()
        self.assertEqual(self.page.locator("html").get_attribute("dir"), "rtl")
        self.assertEqual(self.page.locator("html").get_attribute("lang"), "ar")
        self.assertEqual(self.page.locator("#heading").inner_text(),
                         "طلبات الخدمة اليوم")
        self.assertIn("إكمال", self.button("Q-118").inner_text())
        for ticket in TICKET_IDS:
            with self.subTest(rtl_visible=ticket):
                self.assert_control_on_screen(self.button(ticket))
        self.assertEqual(self.png_digest(), self.png_digest(),
                         "RTL scene screenshot changed without action")
        self.button("Q-118").click()
        self.assertEqual(self.row("Q-118").get_attribute("data-status"), "done")
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "1")
        self.assertEqual(self.page.locator("#urgent-count").inner_text(), "1")
        self.assertEqual(self.row("Q-137").get_attribute("data-status"), "waiting")
        self.page.locator("#reset").click()
        self.assertEqual(self.page.locator("#completed-count").inner_text(), "0")
        self.assertEqual(self.page.locator("#urgent-count").inner_text(), "2")
        self.assertEqual(self.row("Q-118").get_attribute("data-status"), "waiting")
        self.assertTrue(self.button("Q-118").is_enabled())
        self.page.locator("#lang").click()
        self.assertEqual(self.page.locator("html").get_attribute("dir"), "ltr")

    def test_visible_original_language_scenes_are_distinct_in_memory(self):
        initial = self.png_digest()
        self.page.locator("#priority").click()
        priority = self.png_digest()
        self.page.locator("#lang").click()
        priority_ar = self.png_digest()
        self.page.locator("#all").click()
        full_ar = self.png_digest()
        self.assertEqual(len({initial, priority, priority_ar, full_ar}), 4,
                         "Original GUI scenes must differ in actual pixels")
        # No PNG written to disk and no screenshot/model artifact is uploaded.


if __name__ == "__main__":
    unittest.main()
