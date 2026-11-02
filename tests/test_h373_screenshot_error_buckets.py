"""H373 source-only adversarial screenshot error classification tests.

No actual source screen, gold/task/coordinates, model weights or browser APIs.
"""
from __future__ import annotations

import unittest

from h370.browser_failure_stage import browser_failure_code
from h371.run_browser_repeats import scrub_failure
from h373.screenshot_error_buckets import CAPTURE_BUCKETS, safe_screenshot_bucket

# No Playwright dependency needed in source-only CI. Simulate the public,
# fixed exception classes without invoking a real browser or private data.
PlaywrightError = type("Error", (Exception,),
                       {"__module__": "playwright._impl._errors"})
PlaywrightTimeout = type("TimeoutError", (Exception,),
                         {"__module__": "playwright._impl._errors"})


class H373CaptureSignatureTests(unittest.TestCase):
    def test_public_protocol_capture_signature_no_payload(self):
        exc = PlaywrightError(
            "Page.screenshot: Protocol error (Page.captureScreenshot): "
            "Unable to capture screenshot; PATH=C:/PRIVATE/Sim-Gold.png")
        self.assertEqual(safe_screenshot_bucket("first_image_capture", exc),
                         "protocol_capture")
        msg = browser_failure_code("h319", "first_image_capture", exc)
        self.assertEqual(
            msg, "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED "
            "stage=first_image_capture kind=browser_error "
            "capture=protocol_capture")
        self.assertNotIn("PRIVATE", msg)
        self.assertNotIn("Sim-Gold", msg)
        self.assertEqual(
            scrub_failure("Traceback: PRIVATE GOLD\n" + msg, "h319"),
            "stage=first_image_capture kind=browser_error "
            "capture=protocol_capture")

    def test_closed_context_and_destroyed_context_signatures(self):
        for text, expected in (
            ("Target page, context or browser has been closed; SECRET",
             "target_closed"),
            ("Execution context was destroyed, most likely because "
             "of a navigation; SCREENSHOT_SECRET", "execution_context"),
            ("Cannot find context with specified id 7 ; GOLD_LABEL",
             "execution_context"),
        ):
            with self.subTest(expected=expected):
                code = browser_failure_code(
                    "h337", "first_image_capture", PlaywrightError(text))
                self.assertTrue(code.endswith("capture=" + expected))
                for secret in ("SECRET", "SCREENSHOT", "GOLD", "id 7"):
                    self.assertNotIn(secret, code)

    def test_font_wait_indeterminate_and_other_timeout(self):
        unresolved = PlaywrightTimeout(
            "Page.screenshot: Timeout 30000ms exceeded; "
            "waiting for fonts to load... PRIVATE_X")
        loaded = PlaywrightTimeout(
            "Page.screenshot: Timeout 30000ms exceeded; "
            "waiting for fonts to load... fonts loaded; PRIVATE_X")
        self.assertEqual(
            safe_screenshot_bucket("second_image_capture", unresolved),
            "font_wait_unresolved")
        self.assertEqual(
            safe_screenshot_bucket("first_post_policy_frame", loaded),
            "screenshot_timeout")

    def test_unmatched_error_is_bounded_other_and_untrusted_types_not_parsed(self):
        for secret in ("D:/PRIVATE/GOLD/SIM-21",
                       "https://user-secret.example/record/999"):
            val = PlaywrightError(secret)
            self.assertEqual(safe_screenshot_bucket("first_image_capture", val),
                             "browser_other")
            msg = browser_failure_code("h319", "first_image_capture", val)
            self.assertNotIn(secret, msg)
        self.assertIsNone(safe_screenshot_bucket(
            "first_image_capture", ValueError("SECRET")))
        self.assertIsNone(safe_screenshot_bucket(
            "first_policy_callback", PlaywrightError("PRIVATE")))
        self.assertIsNone(safe_screenshot_bucket(
            "native_click", PlaywrightTimeout("PRIVATE")))
        self.assertIsNone(safe_screenshot_bucket(
            "first_image_capture", RuntimeError("PRIVATE")))
        self.assertEqual(
            browser_failure_code("h319", "first_policy_callback",
                                 PlaywrightError("SECRET")),
            "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED "
            "stage=first_policy_callback kind=browser_error")

    def test_soak_filter_accepts_only_fixed_signatures(self):
        expected = "stage=first_image_capture kind=browser_error"
        prefix = "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED " + expected
        for suffix, result in (
            (" capture=PRIVATE_GOLD", expected),
            (" capture=protocol_capture", expected + " capture=protocol_capture"),
            (" capture=target_closed", expected + " capture=target_closed"),
        ):
            with self.subTest(suffix=suffix):
                self.assertEqual(scrub_failure(prefix + suffix, "h319"), result)
        self.assertTrue({"protocol_capture", "browser_other"}.issubset(CAPTURE_BUCKETS))

    def test_no_promoted_release_or_training_evidence(self):
        # Diagnostic labels cannot be mistaken for any model metric.
        for s in CAPTURE_BUCKETS:
            self.assertNotIn("train", s)
            self.assertNotIn("release", s)
            self.assertNotIn("blind", s)


if __name__ == "__main__":
    unittest.main()
