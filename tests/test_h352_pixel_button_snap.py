"""H352 invented-only pixel-snap tests and opted-in original-source browser QA.

Model output and source oracle are NEVER read by h352.pixel_button_snap.
Opt-in source checks require Playwright and are NOT model-time evidence.
"""
import io
import json
import os
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from h199.native_action_sft import parse_actions
from h352.pixel_button_snap import pixel_button_rectangles, snap_native_first_click


def invented_image(*, dark=False, ambiguous=False):
    background = (22, 27, 47) if dark else (244, 239, 223)
    color = (29, 110, 118) if dark else (58, 98, 140)
    im = Image.new("RGB", (960, 720), background)
    pen = ImageDraw.Draw(im)
    pen.rectangle((0, 0, 959, 104), fill=(17, 18, 41) if dark else (50, 44, 83))
    right_start = 450 if ambiguous else 535
    for y in (255, 455):
        for left in (45, right_start):
            pen.rectangle((left, y, left + 375, y + 52), fill=color)
    dest = io.BytesIO()
    im.save(dest, "PNG")
    return dest.getvalue()


def click(x, y):
    return json.dumps({"actions": [{"kind": "click", "point": [x, y]}]},
                      separators=(",", ":"))


class H352InventedPixelOnlyTests(unittest.TestCase):
    def test_four_pure_pixel_boxes_light_dark(self):
        for dark in (False, True):
            with self.subTest(dark=dark):
                boxes = pixel_button_rectangles(invented_image(dark=dark))
                self.assertEqual(len(boxes), 4)
                self.assertTrue(all(350 <= b[2] - b[0] <= 450 for b in boxes))

    def test_click_outside_button_snaps_to_unambiguous_center(self):
        for dark in (False, True):
            raw = click(438, 282)
            result = snap_native_first_click(invented_image(dark=dark), raw)
            parsed = parse_actions(result)["actions"][0]["point"]
            self.assertGreater(parsed[0], 190)
            self.assertLess(parsed[0], 300)
            self.assertGreater(parsed[1], 255)
            self.assertLess(parsed[1], 310)
            self.assertNotEqual(result, raw)

    def test_inside_also_snaps_to_center(self):
        result = parse_actions(snap_native_first_click(
            invented_image(), click(100, 270)))["actions"][0]["point"]
        self.assertTrue(200 <= result[0] <= 255)
        self.assertTrue(260 <= result[1] <= 305)

    def test_already_center_click_is_true_noop_despite_json_formatting(self):
        png = invented_image()
        left, top, right, bottom = pixel_button_rectangles(png)[0]
        center = [(left + right) // 2, (top + bottom) // 2]
        raw = json.dumps({"actions": [{"kind": "click", "point": center}]},
                         indent=2)
        self.assertEqual(snap_native_first_click(png, raw), raw)

    def test_ambiguous_near_two_buttons_keeps_original_point(self):
        image = invented_image(ambiguous=True)
        raw = click(435, 280)
        self.assertEqual(snap_native_first_click(image, raw), raw)

    def test_far_away_model_click_stays_unmodified(self):
        raw = click(930, 660)
        self.assertEqual(snap_native_first_click(invented_image(), raw), raw)

    def test_abstention_remains_untouched(self):
        raw = '{"actions":[{"kind":"abstain"}]}'
        self.assertEqual(snap_native_first_click(b"", raw), raw)

    def test_unrelated_image_and_wrong_aspect_fail_closed(self):
        from PIL import Image
        def png_of(w, h):
            buf = io.BytesIO()
            Image.new("RGB", (w, h), (255, 255, 255)).save(buf, "PNG")
            return buf.getvalue()
        for data in (b"nonpng", png_of(960, 720), png_of(800, 600)):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                snap_native_first_click(data, click(50, 50))

    def test_native_invalid_action_rejected(self):
        for raw in ('{"actions":[{"kind":"type_text","text":"secret"}]}',
                    '{"actions":[{"kind":"click","point":[1,1]},{"kind":"click","point":[2,2]}]}',
                    'hi', '{"actions": []}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                snap_native_first_click(invented_image(), raw)


@unittest.skipUnless(os.environ.get("HALA_H352_ORIGINAL_BROWSER") == "1",
                     "Real source browser opted in separately; CPU tests are not model scores")
class H352OriginalBrowserPixelOnlyTests(unittest.TestCase):
    def test_all_twelve_author_owned_original_images_detect_four_real_buttons(self):
        from playwright.sync_api import sync_playwright
        from h319.beaconboard_dynamic_research import (
            SCENES, checked_source, original_scene,
        )
        source, url = checked_source(Path("h319/beaconboard.html"))
        from h319.beaconboard_dynamic_research import _observed_png
        with sync_playwright() as pw:
            import sys
            kwargs = {"channel": "msedge"} if sys.platform == "win32" else {}
            browser = pw.chromium.launch(headless=True, **kwargs)
            try:
                for scene in SCENES:
                    with original_scene(browser, source, url, scene) as page:
                        pixels = _observed_png(page)
                        detected = pixel_button_rectangles(pixels)
                        self.assertEqual(len(detected), 4)
                        # Source DOM is an EVALUATOR ONLY, never a policy input.
                        source_buttons = [
                            button.bounding_box() for button in
                            page.locator("button[data-station]").all()
                        ]
                        self.assertEqual(len(source_buttons), 4)
                        matches = set()
                        for box in detected:
                            center = ((box[0] + box[2]) / 2,
                                      (box[1] + box[3]) / 2)
                            matched = [
                                i for i, original in enumerate(source_buttons)
                                if original["x"] <= center[0] <= original["x"] + original["width"]
                                and original["y"] <= center[1] <= original["y"] + original["height"]
                            ]
                            self.assertEqual(len(matched), 1)
                            matches.add(matched[0])
                        self.assertEqual(len(matches), 4)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
