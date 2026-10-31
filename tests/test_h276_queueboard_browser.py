"""H276 live Chromium qualification of PUBLIC invented first-party Queueboard.

This suite is deliberately opt-in (HALA_H276_BROWSER_QUALIFY=1).
Existing CPU CI does not claim browser coverage when Playwright is absent.
The policy in these tests is an invented fixed-action stub, NOT Qwen.
No screenshots, model responses, private media or target points are logged.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest

from h263.native_browser_executor import new_offline_context
from h264.screenshot_locked_research import _observed_png
from h276.queueboard_native_research import (
    TICKETS, _ticket_state, checked_queueboard,
    run_firstparty_queueboard_probe, screenshot_locked_queueboard_turn,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h275" / "queueboard.html"


class H276RealBrowserQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("HALA_H276_BROWSER_QUALIFY") != "1":
            raise unittest.SkipTest("H276 live browser explicitly not run in CPU-only CI")
        from playwright.sync_api import sync_playwright
        cls.playwright = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H276_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("H276 unsupported browser channel")
            options = ({"headless": True, "channel": "msedge"}
                       if channel == "msedge"
                       else {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.playwright.chromium.launch(**options)
        except Exception:
            cls.playwright.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def fresh(self, *, language="en", priority=False):
        context = new_offline_context(self.browser)
        context.route("**/*", lambda route: route.abort()
                      if route.request.url.startswith(("http:", "https:", "ws:", "wss:"))
                      else route.continue_())
        page = context.new_page()
        page.goto(checked_queueboard(SOURCE)[1], wait_until="load")
        if language == "ar":
            page.locator("#lang").click()
        if priority:
            page.locator("#priority").click()
        page.evaluate("document.activeElement && document.activeElement.blur()")
        return context, page

    def invented_point(self, ticket, *, language="en", priority=False):
        """Fixture reads public DOM BEFORE policy creation, never in policy."""
        context, page = self.fresh(language=language, priority=priority)
        try:
            box = page.locator(
                'button.complete-btn[data-ticket-id="' + ticket + '"]').bounding_box()
            self.assertIsNotNone(box)
            point = [int(box["x"] + box["width"] / 2),
                     int(box["y"] + box["height"] / 2)]
            self.assertTrue(0 <= point[0] < 960 and 0 <= point[1] < 720)
            return point
        finally:
            context.close()

    @staticmethod
    def invented_policy(point):
        def predict(png, prompt):
            if not png[:4] == bytes((137, 80, 78, 71)):
                raise ValueError("missing real screenshot")
            if "Task:" not in prompt:
                raise ValueError("missing screenshot-only instruction")
            return json.dumps({"actions": [{"kind": "click", "point": point}]},
                              separators=(",", ":"))
        return predict

    def test_live_english_correct_row_only(self):
        point = self.invented_point("Q-137")
        result = run_firstparty_queueboard_probe(
            self.browser, SOURCE, "Complete fictional ticket Q-137",
            "Q-137", self.invented_policy(point))
        self.assertTrue(result["native_action_dispatched"])
        self.assertTrue(result["firstparty_task_success"])
        self.assertFalse(result["real_model_inference_attested"])
        self.assertFalse(result["independent_blind_ood"])

    def test_live_wrong_neighbor_fails_oracle(self):
        wrong_point = self.invented_point("Q-118")
        result = run_firstparty_queueboard_probe(
            self.browser, SOURCE, "Complete fictional ticket Q-137",
            "Q-137", self.invented_policy(wrong_point))
        self.assertTrue(result["native_action_dispatched"])
        self.assertFalse(result["firstparty_task_success"])

    def test_live_arabic_rtl_priority_target(self):
        point = self.invented_point("Q-137", language="ar", priority=True)
        result = run_firstparty_queueboard_probe(
            self.browser, SOURCE, "أكمل الطلب الخيالي Q-137",
            "Q-137", self.invented_policy(point),
            language="ar", priority_only=True)
        self.assertTrue(result["firstparty_task_success"])

    def test_live_abstain_is_noop(self):
        result = run_firstparty_queueboard_probe(
            self.browser, SOURCE, "Complete fictional ticket Q-137",
            "Q-137", lambda png, prompt: '{"actions":[{"kind":"abstain"}]}')
        self.assertEqual(result["native_action_kind"], "abstain")
        self.assertFalse(result["native_action_dispatched"])
        self.assertFalse(result["firstparty_task_success"])

    def test_live_stale_screenshot_blocks_click(self):
        context, page = self.fresh()
        try:
            source, url = checked_queueboard(SOURCE)
            point = self.invented_point("Q-137")
            def mutate_after_observation(png, prompt):
                page.locator("#priority").click()
                return json.dumps({"actions": [{"kind": "click", "point": point}]})
            with self.assertRaisesRegex(ValueError, "stale screenshot"):
                screenshot_locked_queueboard_turn(
                    page, source, url, "Complete fictional ticket Q-137",
                    mutate_after_observation)
            self.assertEqual(page.locator("#completed-count").inner_text(), "0")
        finally:
            context.close()

    def test_live_stable_decoded_png_and_no_external_requests(self):
        context, page = self.fresh(language="ar")
        try:
            requests = []
            page.on("request", lambda request: requests.append(request.url)
                    if request.url.startswith(("http:", "https:", "ws:", "wss:"))
                    else None)
            before = _observed_png(page)
            after = _observed_png(page)
            self.assertEqual(sha256(before).digest(), sha256(after).digest())
            self.assertEqual(page.locator("html").get_attribute("dir"), "rtl")
            self.assertEqual(page.locator('article.ticket-row').count(), 6)
            self.assertEqual(requests, [])
        finally:
            context.close()

    def test_source_identity_fails_closed_on_changed_bytes(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "h275"
            parent.mkdir()
            modified = parent / "queueboard.html"
            modified.write_bytes(SOURCE.read_bytes().replace(
                b"Queueboard | Original Hala GUI Research",
                b"Queueboard | Changed Test Source"))
            with self.assertRaisesRegex(ValueError, "source identity drift"):
                checked_queueboard(modified)


    def test_full_state_oracle_sees_priority_hidden_tickets_and_fails_closed(self):
        """Source oracle inspects all six synthetic tickets, not just two rows."""
        context, page = self.fresh(priority=True)
        try:
            self.assertEqual(page.locator("article.ticket-row").count(), 2)
            self.assertEqual(
                _ticket_state(page),
                {ticket: "waiting" for ticket in TICKETS},
            )
            # Test-only mutation to an invisible ticket, no render; the
            # pinned first-party internal state is intentionally probed.
            page.evaluate("tickets.find(x => x.id === 'Q-104').done = true")
            self.assertEqual(page.locator("article.ticket-row").count(), 2)
            self.assertEqual(_ticket_state(page)["Q-104"], "done")
            page.evaluate(
                "tickets.push({id: 'Q-999', done: false})"
            )
            with self.assertRaisesRegex(ValueError, "incomplete first-party"):
                _ticket_state(page)
        finally:
            context.close()

    def test_full_state_oracle_rejects_duplicate_visible_identity(self):
        context, page = self.fresh(priority=True)
        try:
            # Exactly six entries, but one fictional identity duplicated.
            # A visible-row-only oracle would silently miss this anomaly.
            page.evaluate("tickets[0].id = 'Q-118'")
            with self.assertRaisesRegex(ValueError, "invalid first-party"):
                _ticket_state(page)
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
