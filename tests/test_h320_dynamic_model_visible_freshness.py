"""H320: invented RGB-only adversarial tests for original dynamic GUI.

No real model, private screenshots, coordinates, gold, GPU or admission.
"""
from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, PngImagePlugin

from h319.beaconboard_dynamic_research import _observed_predict


def frame(color=(24, 45, 70), *, comment=None):
    image = Image.new("RGB", (960, 720), color)
    output = BytesIO()
    metadata = None
    if comment is not None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("invented", comment)
    image.save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


class FakePage:
    url = "file:///invented/h319/beaconboard.html"

    def __init__(self, *frames):
        self.frames = iter(frames)
        self.calls = 0

    def screenshot(self, **kwargs):
        assert kwargs == {"animations": "disabled", "full_page": False}
        self.calls += 1
        return next(self.frames)


CLICK = '{"actions":[{"kind":"click","point":[200,200]}]}'
TYPE = '{"actions":[{"kind":"type_text","text":"SIM-21"}]}'


class H320ModelVisibleFrameTests(unittest.TestCase):
    def test_metadata_only_change_during_callback_does_not_fake_mutation(self):
        a, b = frame(comment="first"), frame(comment="second")
        self.assertNotEqual(a, b)
        page = FakePage(a, b)
        calls = []
        result = _observed_predict(
            page, lambda png, prompt: calls.append(1) or CLICK,
            "Open fictional station B-11.", set())
        self.assertEqual(result["kind"], "click")
        self.assertEqual(len(calls), 1)
        self.assertEqual(page.calls, 2)

    def test_identical_rgb_reencoded_between_steps_refused_before_callback(self):
        a, b = frame(comment="first"), frame(comment="second")
        page = FakePage(a, a, b)
        seen = set()
        calls = []
        callback = lambda png, prompt: calls.append(1) or CLICK
        _observed_predict(page, callback, "Open fictional station B-11.", seen)
        with self.assertRaises(ValueError):
            _observed_predict(page, callback, "Open fictional station B-11.", seen)
        self.assertEqual(len(calls), 1)
        self.assertEqual(page.calls, 3)

    def test_new_visible_pixels_allow_second_distinct_prediction(self):
        a, b = frame(), frame((25, 45, 70))
        page = FakePage(a, a, b, b)
        seen = set()
        calls = []
        def predict(png, prompt):
            calls.append(1)
            return CLICK if len(calls) == 1 else TYPE
        first = _observed_predict(page, predict, "Open fictional station B-11.", seen)
        second = _observed_predict(page, predict, "Open fictional station B-11.", seen)
        self.assertEqual((first["kind"], second["kind"]), ("click", "type_text"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(seen), 2)

    def test_visible_mutation_during_callback_refused(self):
        a, b = frame(), frame((26, 45, 70))
        page = FakePage(a, b)
        seen = set()
        with self.assertRaises(ValueError):
            _observed_predict(
                page, lambda png, prompt: CLICK,
                "Open fictional station B-11.", seen)
        self.assertEqual(len(seen), 0)

    def test_invalid_dimensions_refused_without_prediction(self):
        output = BytesIO()
        Image.new("RGB", (959, 720), (1, 2, 3)).save(output, format="PNG")
        page = FakePage(output.getvalue())
        calls = []
        with self.assertRaises(ValueError):
            _observed_predict(
                page, lambda png, prompt: calls.append(1) or CLICK,
                "Open fictional station B-11.", set())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
