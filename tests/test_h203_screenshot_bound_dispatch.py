"""H203 screenshot-bound GUI dispatch; synthetic determinism and tamper tests.

Only generated RGB PNG fixtures. No owner gold, old OOD, GPU or model calls.
"""
import json
import struct
import unittest
import zlib

from h201.original_native_action_executor import (
    MAX_PREACTION_PNG_BYTES,
    execute_attested_native_action,
    exact_original_loopback_url,
)


def _chunk(name: bytes, data: bytes) -> bytes:
    body = name + data
    return struct.pack(">I", len(data)) + body + struct.pack(
        ">I", zlib.crc32(body) & 0xffffffff
    )


def generated_960_png(fill: int = 0, *, width: int = 960) -> bytes:
    """Valid stdlib-only synthetic RGB PNG; no real/source screenshot."""
    row = b"\x00" + bytes([fill]) * (width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, 720, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(row * 720, level=9))
        + _chunk(b"IEND", b"")
    )


class Mouse:
    def __init__(self, page):
        self.page = page

    def click(self, x, y):
        self.page.events.append(("click", x, y))

    def move(self, x, y, **kw):
        self.page.events.append(("move", x, y))

    def down(self):
        self.page.events.append(("down",))

    def up(self):
        self.page.events.append(("up",))


class Keyboard:
    def __init__(self, page):
        self.page = page

    def press(self, key):
        self.page.events.append(("key", key))


class Page:
    def __init__(self, image: object):
        self.url = exact_original_loopback_url(8201)
        self.viewport_size = {"width": 960, "height": 720}
        self.image = image
        self.redirect_on_screenshot = False
        self.resize_on_screenshot = False
        self.fail_screenshot = False
        self.screenshots = 0
        self.events = []
        self.mouse = Mouse(self)
        self.keyboard = Keyboard(self)

    def screenshot(self, *, animations, full_page):
        self.screenshots += 1
        if animations != "disabled" or full_page is not False:
            raise AssertionError("unexpected screenshot acquisition")
        if self.fail_screenshot:
            raise RuntimeError("browser screenshot failed")
        if self.redirect_on_screenshot:
            self.url = "https://unregistered.invalid/"
        if self.resize_on_screenshot:
            self.viewport_size = {"width": 720, "height": 960}
        return self.image


def one(kind="click", **kwargs):
    if not kwargs and kind == "click":
        kwargs = {"point": [200, 170]}
    return json.dumps({"actions": [{"kind": kind, **kwargs}]})


class H203ScreenshotBindingTests(unittest.TestCase):
    def test_matching_model_visible_image_executes_native_click(self):
        img = generated_960_png()
        p = Page(img)
        r = execute_attested_native_action(p, one(), p.url, img)
        self.assertTrue(r["preaction_png_verified"])
        self.assertEqual(r["native_events"], 1)
        self.assertEqual(p.events, [("click", 200, 170)])
        self.assertEqual(p.screenshots, 1)

    def test_changed_gui_pixels_rejected_without_input(self):
        p = Page(generated_960_png(10))
        with self.assertRaisesRegex(ValueError, "stale or changed"):
            execute_attested_native_action(p, one(), p.url, generated_960_png(0))
        self.assertEqual(p.events, [])

    def test_changed_gui_pixels_rejected_before_native_keyboard(self):
        p = Page(generated_960_png(42))
        with self.assertRaisesRegex(ValueError, "stale or changed"):
            execute_attested_native_action(
                p, one("key_press", point=[200, 170], key="ArrowRight"),
                p.url, generated_960_png(0),
            )
        self.assertEqual(p.events, [])

    def test_invalid_model_action_rejected_without_a_screenshot(self):
        p = Page(generated_960_png())
        for raw in (
            '{"actions":[{"kind":"click","point":[1,2],"point":[3,4]}]}',
            '{"actions":[{"kind":"click","point":[NaN,2]}]}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    execute_attested_native_action(p, raw, p.url, p.image)
        self.assertEqual(p.screenshots, 0)
        self.assertEqual(p.events, [])

    def test_invalid_or_oversize_expected_image_rejected_without_screen(self):
        img = generated_960_png()
        for bad in (
            None, b"", bytearray(img), b"arbitrary media",
            img[:12], generated_960_png(width=959),
            img + b"\x00" * (MAX_PREACTION_PNG_BYTES + 1),
        ):
            with self.subTest(kind=type(bad).__name__, size=len(bad) if bad else 0):
                p = Page(img)
                with self.assertRaises(ValueError):
                    execute_attested_native_action(p, one(), p.url, bad)
                self.assertEqual(p.screenshots, 0)
                self.assertEqual(p.events, [])

    def test_malformed_current_browser_png_rejected_no_input(self):
        img = generated_960_png()
        for bad in (b"not png", generated_960_png(width=959), None):
            with self.subTest(bad=type(bad).__name__):
                p = Page(bad)
                with self.assertRaises(ValueError):
                    execute_attested_native_action(p, one(), p.url, img)
                self.assertEqual(p.events, [])

    def test_wrong_original_route_rejected_before_screen(self):
        img = generated_960_png()
        p = Page(img)
        with self.assertRaisesRegex(ValueError, "only registered"):
            execute_attested_native_action(
                p, one(), "http://127.0.0.1:8201/other.html", img)
        self.assertEqual(p.screenshots, 0)
        self.assertEqual(p.events, [])

    def test_redirect_while_taking_screenshot_denies_action(self):
        img = generated_960_png()
        p = Page(img)
        p.redirect_on_screenshot = True
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_attested_native_action(p, one(), p.url, img)
        self.assertEqual(p.events, [])

    def test_resize_while_taking_screenshot_denies_action(self):
        img = generated_960_png()
        p = Page(img)
        p.resize_on_screenshot = True
        with self.assertRaisesRegex(ValueError, "viewport"):
            execute_attested_native_action(p, one(), p.url, img)
        self.assertEqual(p.events, [])

    def test_screenshot_engine_error_propagates_without_input(self):
        img = generated_960_png()
        p = Page(img)
        p.fail_screenshot = True
        with self.assertRaisesRegex(RuntimeError, "screenshot failed"):
            execute_attested_native_action(p, one(), p.url, img)
        self.assertEqual(p.events, [])

    def test_matching_abstain_keeps_no_input(self):
        img = generated_960_png()
        p = Page(img)
        result = execute_attested_native_action(
            p, one("abstain"), p.url, img
        )
        self.assertEqual(result["native_events"], 0)
        self.assertTrue(result["preaction_png_verified"])
        self.assertEqual(p.events, [])


if __name__ == "__main__":
    unittest.main()
