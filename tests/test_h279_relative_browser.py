"""H279 real first-party browser bridge tests; synthetic fixed-action policy only.

No pretrained model is loaded, no OOD score, no screenshot or coordinate is
saved. The test-only source DOM is consulted to select a fixture point.
The actual H279 private Qwen model must NOT receive this DOM/fixture point.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from h264.screenshot_locked_research import checked_stockroom, screenshot_locked_turn
from h273.relative_native_bridge import (
    FIRST_STOCK_TASK, normalize_strict_action, NORMALIZED_REGEX,
)
from h199.native_action_sft import WIDTH, HEIGHT
from h263.native_browser_executor import new_offline_context
from h279.private_relative_native_probe import RelativePrevalidationProcessor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h262" / "stockroom.html"


@unittest.skipUnless(os.environ.get("HALA_H279_BROWSER_QUALIFY") == "1",
                     "real browser qualification requires dedicated opt-in runner")
class H279RealNativeFrameBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H279_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("only first-party Chromium/Edge allowed")
            opts = ({"headless": True, "channel": "msedge"} if channel == "msedge"
                    else {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.pw.chromium.launch(**opts)
        except BaseException:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        path, url = checked_stockroom(SOURCE)
        self.url = url
        self.context = new_offline_context(self.browser)
        self.network = []
        self.page_errors = []

        def offline_only(route):
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                self.network.append("external request refused")
                route.abort()
            else:
                route.continue_()

        self.context.route("**/*", offline_only)
        self.page = self.context.new_page()
        self.page.on("pageerror", lambda err: self.page_errors.append(str(err)))
        self.page.goto(url, wait_until="load")
        self.page.locator("#sort").click()
        self.page.locator("#lang").click()
        self.assertEqual(self.page.locator("html").get_attribute("dir"), "rtl")
        self.assertEqual(self.page.viewport_size, {"width": WIDTH, "height": HEIGHT})

    def tearDown(self):
        try:
            self.assertEqual(self.network, [])
            self.assertEqual(self.page_errors, [])
        finally:
            self.context.close()

    def stock_by_id(self):
        """First-party TEST oracle, never passed to pretrained policy.

        Browser rendering/sorting changes row positions; correct state
        evidence must bind each visible stock count to its stable ID.
        """
        rows = self.page.locator(".item")
        state = {}
        for index in range(rows.count()):
            row = rows.nth(index)
            item_id = row.get_attribute("data-id")
            self.assertIsNotNone(item_id)
            self.assertNotIn(item_id, state, "duplicate stock identity")
            state[item_id] = int(row.locator(".count").inner_text())
        self.assertGreaterEqual(len(state), 2)
        return state

    def fixture_relative_action(self, row_index):
        """Private test fixture: inverse coordinate for a source-owned button.

        This is a stub-only browser unit test, NEVER actual Qwen or independent
        OOD. No target data enters a genuine pretrained model.
        """
        button = self.page.locator("button[data-restock]").nth(row_index)
        bounds = button.bounding_box()
        self.assertIsNotNone(bounds)
        center = (int(bounds["x"] + bounds["width"]/2),
                  int(bounds["y"] + bounds["height"]/2))
        self.assertTrue(0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT)
        relative = [int(center[0]*1000/WIDTH), int(center[1]*1000/HEIGHT)]
        return json.dumps({"actions": [{"kind": "click", "point": relative}]},
                          separators=(",", ":"))

    def do_fixed_native_action(self, relative):
        native = normalize_strict_action(relative)
        return screenshot_locked_turn(
            self.page, SOURCE, self.url, FIRST_STOCK_TASK,
            lambda png, task: native,
        )

    def test_correct_first_row_fixture_native_click_changes_only_first(self):
        before = self.stock_by_id()
        intended_id = self.page.locator(".item").first.get_attribute("data-id")
        self.assertIn(intended_id, before)
        relative = self.fixture_relative_action(0)
        result = self.do_fixed_native_action(relative)
        after = self.stock_by_id()
        self.assertEqual(result["kind"], "click")
        self.assertTrue(result["executed"])
        expected = before.copy()
        expected[intended_id] += 1
        self.assertEqual(after, expected)

    def test_wrong_neighbor_fixture_remains_wrong_not_oracle_retargeted(self):
        before = self.stock_by_id()
        first_id = self.page.locator(".item").first.get_attribute("data-id")
        neighbor_id = self.page.locator(".item").nth(1).get_attribute("data-id")
        self.assertNotEqual(neighbor_id, first_id)
        relative = self.fixture_relative_action(1)
        result = self.do_fixed_native_action(relative)
        after = self.stock_by_id()
        self.assertTrue(result["executed"])
        expected = before.copy()
        expected[neighbor_id] += 1
        self.assertEqual(after, expected)
        self.assertEqual(after[first_id], before[first_id])

    def test_invalid_relative_coordinate_rejected_before_native_click(self):
        before = self.stock_by_id()
        invalid = '{"actions":[{"kind":"click","point":[1000,999]}]}'
        with self.assertRaisesRegex(ValueError, "invalid strict normalized"):
            self.do_fixed_native_action(invalid)
        self.assertEqual(self.stock_by_id(), before)

    def test_abstain_does_not_change_native_gui(self):
        before = self.stock_by_id()
        result = self.do_fixed_native_action('{"actions":[{"kind":"abstain"}]}')
        self.assertEqual(result["kind"], "abstain")
        self.assertFalse(result["executed"])
        self.assertEqual(self.stock_by_id(), before)

    def test_stale_screenshot_blocks_click_after_mutated_scene(self):
        relative = self.fixture_relative_action(0)
        native = normalize_strict_action(relative)
        before = self.stock_by_id()
        def changed_scene_policy(png, task):
            self.page.locator("#sort").click()
            return native
        with self.assertRaisesRegex(ValueError, "stale screenshot"):
            screenshot_locked_turn(
                self.page, SOURCE, self.url, FIRST_STOCK_TASK, changed_scene_policy)
        # The intentional filter/sort mutation changes visible row ORDER.
        # Guard must block action, but preserve inventory values per ID.
        self.assertEqual(self.stock_by_id(), before)


    def test_identity_oracle_catches_swapped_counts_with_same_multiset(self):
        """Sorting value multisets can silently miss two swapped items."""
        before = self.stock_by_id()
        self.page.evaluate("""() => {
            const a = inventory[0], b = inventory[1];
            const original = a.count;
            a.count = b.count;
            b.count = original;
            renderItems();
        }""")
        after = self.stock_by_id()
        self.assertEqual(sorted(after.values()), sorted(before.values()))
        self.assertNotEqual(after, before)



if __name__ == "__main__":
    unittest.main()
