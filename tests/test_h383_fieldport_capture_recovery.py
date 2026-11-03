"""H383: strict one-retry FieldPort screenshot recovery, source-only CPU tests.

Only invented PNG-like bytes and fake pages; no private browser, source gold,
real Qwen inference, legal TRAIN, adapted weights or blinded OOD.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h296.fieldport_sequence_research import _screen_locked_predict
from h322.source_capture import SourceCaptureRefused


class Error(Exception):
    """Fake Playwright Error with exactly the type name used by H322."""


KNOWN_PROTOCOL_ERROR = (
    "Page.screenshot: Protocol error (Page.captureScreenshot): "
    "Unable to capture screenshot"
)
PNG = b"\x89PNG\r\n\x1a\n" + b"INVENTED-SOURCE-ONLY"
ABSTAIN = '{"actions":[{"kind":"abstain"}]}'


class FakePublicPage:
    def __init__(self):
        self.url = "file:///invented-first-party-only.html"


class H383FieldPortCaptureTests(unittest.TestCase):
    def test_single_known_first_capture_error_recovers_before_policy(self):
        page = FakePublicPage()
        invocations = []
        calls = []
        def screenshot(_page):
            calls.append(len(calls))
            if len(calls) == 1:
                raise Error(KNOWN_PROTOCOL_ERROR)
            return PNG

        def predict(image, text):
            invocations.append((image, text))
            return ABSTAIN

        with patch("h296.fieldport_sequence_research._observed_png",
                   side_effect=screenshot):
            self.assertEqual(_screen_locked_predict(page, predict, "Click Save"),
                             ABSTAIN)
        # First capture: one bounded retry. Then one fresh lock recheck.
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0][0], PNG)
        self.assertIn("Task: Click Save", invocations[0][1])

    def test_both_known_cdp_capture_errors_refuse_before_model_call(self):
        page = FakePublicPage()
        seen = []
        with patch("h296.fieldport_sequence_research._observed_png",
                   side_effect=Error(KNOWN_PROTOCOL_ERROR)) as screenshot:
            with self.assertRaisesRegex(SourceCaptureRefused,
                                        "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED"):
                _screen_locked_predict(page, lambda *_: seen.append(True),
                                       "Click Save")
        self.assertEqual(screenshot.call_count, 2)
        self.assertEqual(seen, [])

    def test_unrecognized_error_is_not_retried_or_logged(self):
        for message in ("Browser context closed", "Timeout", "Wrong screenshot"):
            with self.subTest(message=message):
                seen = []
                with patch("h296.fieldport_sequence_research._observed_png",
                           side_effect=Error(message)) as screenshot:
                    with self.assertRaisesRegex(SourceCaptureRefused,
                                                "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED"):
                        _screen_locked_predict(
                            FakePublicPage(), lambda *_: seen.append(True),
                            "Click Save")
                self.assertEqual(screenshot.call_count, 1)
                self.assertEqual(seen, [])

    def test_stale_model_visible_frame_still_refused(self):
        reads = []
        def screenshot(_page):
            reads.append(True)
            return PNG if len(reads) == 1 else PNG + b"-CHANGED"

        policy_calls = []
        def predict(*_args):
            policy_calls.append(True)
            return ABSTAIN

        with patch("h296.fieldport_sequence_research._observed_png",
                   side_effect=screenshot):
            with self.assertRaisesRegex(ValueError, "screenshot changed"):
                _screen_locked_predict(FakePublicPage(), predict,
                                       "Click Save")
        self.assertEqual(len(reads), 2)
        self.assertEqual(len(policy_calls), 1)

    def test_original_page_url_change_during_policy_refuses_without_recheck(self):
        page = FakePublicPage()
        def change_page(image, text):
            page.url = "file:///different-original.html"
            return ABSTAIN

        with patch("h296.fieldport_sequence_research._observed_png",
                   return_value=PNG) as screenshot:
            with self.assertRaisesRegex(ValueError, "screenshot changed"):
                _screen_locked_predict(page, change_page, "Click Save")
        self.assertEqual(screenshot.call_count, 1)

    def test_malformed_screen_does_not_reach_policy(self):
        seen = []
        with patch("h296.fieldport_sequence_research._observed_png",
                   return_value=b"NOT-PNG"):
            with self.assertRaisesRegex(ValueError, "genuine browser screenshot"):
                _screen_locked_predict(
                    FakePublicPage(), lambda *_: seen.append(True),
                    "Click Save")
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
