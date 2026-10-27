"""H159 fail-closed screenshot stability gate without private cases."""
import inspect
import unittest
from collections import Counter

from h159.stable_source_qa import (
    COUNT, SCHEMA, START_INDEX, cases, stable_png, capture, verify, scenario,
)


class Page:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.waits = []

    def evaluate(self, expression):
        assert "document.fonts.ready" in expression

    def wait_for_timeout(self, ms):
        self.waits.append(ms)

    def screenshot(self, **kwargs):
        assert kwargs == {"type": "png", "animations": "disabled"}
        return next(self.frames)


class H159SourceIdentityUnitTests(unittest.TestCase):
    def test_requires_two_identical_bytes_after_update(self):
        page = Page([b"frameA", b"frameB", b"frameC", b"frameC"])
        self.assertEqual(stable_png(page), b"frameC")
        self.assertEqual(page.waits, [200, 125, 125, 125])

    def test_fails_closed_if_pixels_never_stabilize(self):
        page = Page([bytes([n]) for n in range(8)])
        with self.assertRaisesRegex(ValueError, "unstable"):
            stable_png(page)
        self.assertEqual(len(page.waits), 8)

    def test_fresh_balanced_fixed_source_cases(self):
        fresh = cases()
        self.assertEqual(len(fresh), COUNT)
        self.assertEqual(len({x["index"] for x in fresh}), COUNT)
        self.assertEqual(fresh[0]["index"], START_INDEX)
        self.assertEqual(fresh[-1]["index"], START_INDEX + COUNT - 1)
        self.assertEqual(
            Counter((x["task"], x["language"]) for x in fresh),
            Counter({
                ("complete_item", "en"): 4,
                ("complete_item", "ar"): 4,
                ("filter_active", "en"): 2,
                ("filter_active", "ar"): 2,
                ("clear_completed", "en"): 2,
                ("clear_completed", "ar"): 2,
            }),
        )

    def test_no_model_or_earlier_predictions_in_image_replay(self):
        src = "\n".join(inspect.getsource(fn) for fn in (
            stable_png, scenario, capture, verify
        ))
        self.assertNotIn("model_action", src)
        self.assertNotIn("private_predictions", src)
        self.assertNotIn("page.mouse.click(*click)", src)
        self.assertNotIn("H158_DIFFERENT_APP_NEURAL_ONESHOT_CONSUMED.lock", src)
        self.assertIn("screenshot", src)
        self.assertIn("h159", SCHEMA)


if __name__ == "__main__":
    unittest.main()
