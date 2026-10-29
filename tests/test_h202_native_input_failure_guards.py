"""H202 fail-closed native input transition tests; synthetic pages only.

This never measures model performance, opens owner images, or touches OOD.
"""
import json
import unittest

from h201.original_native_action_executor import (
    execute_one_native_action, exact_original_loopback_url,
)


class FakeMouse:
    def __init__(self, page):
        self.page = page
        self.move_number = 0

    def click(self, x, y):
        self.page.events.append(("click", x, y))
        if self.page.redirect_on == "click":
            self.page.url = "https://unregistered.invalid/"

    def move(self, x, y, **kwargs):
        self.move_number += 1
        self.page.events.append(("move", x, y))
        if self.page.redirect_on == "move" and self.move_number == 2:
            self.page.url = "https://unregistered.invalid/"
        if self.page.fail_on == "move" and self.move_number == 2:
            raise RuntimeError("injected pointer move failure")

    def down(self):
        self.page.events.append(("down",))
        if self.page.redirect_on == "down":
            self.page.url = "https://unregistered.invalid/"
        if self.page.fail_on == "down":
            raise RuntimeError("injected pointer down failure")

    def up(self):
        self.page.events.append(("up",))


class FakeKeyboard:
    def __init__(self, page):
        self.page = page

    def press(self, key):
        self.page.events.append(("key", key))
        if self.page.redirect_on == "key":
            self.page.url = "https://unregistered.invalid/"


class FakePage:
    def __init__(self, redirect_on=None, fail_on=None):
        self.url = exact_original_loopback_url(8201)
        self.viewport_size = {"width": 960, "height": 720}
        self.events = []
        self.redirect_on = redirect_on
        self.fail_on = fail_on
        self.mouse = FakeMouse(self)
        self.keyboard = FakeKeyboard(self)


def action(kind, **kwargs):
    return json.dumps({"actions": [{"kind": kind, **kwargs}]})


class H202NativeTransitionSafety(unittest.TestCase):
    def test_only_exact_canonical_source_url_is_accepted(self):
        for bad in (
            "http://127.0.0.1:0/snap/index.html",
            "http://127.0.0.1:65536/snap/index.html",
            "http://127.0.0.1:08201/snap/index.html",
            "http://127.0.0.1:8201/snap/index.html?tracking=1",
            "http://127.0.0.1:8201/snap/index.html#note",
            "http://127.0.0.1:8201@evil.invalid/snap/index.html",
            "http://127.0.0.1:8201/snap/indexXhtml",
            "http://127x0x0x1:8201/snap/index.html",
            None,
        ):
            with self.subTest(bad=bad):
                p = FakePage()
                with self.assertRaisesRegex(ValueError, "only registered"):
                    execute_one_native_action(
                        p, action("click", point=[10, 20]), bad
                    )
                self.assertEqual(p.events, [])

    def test_click_does_not_report_success_after_unregistered_navigation(self):
        p = FakePage(redirect_on="click")
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_one_native_action(p, action("click", point=[10, 20]),
                                      exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events], ["click"])

    def test_focus_click_navigation_never_sends_keyboard(self):
        p = FakePage(redirect_on="click")
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_one_native_action(
                p, action("key_press", point=[120, 130], key="ArrowRight"),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events], ["click"])

    def test_keyboard_navigation_cannot_be_reported_as_source_success(self):
        p = FakePage(redirect_on="key")
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_one_native_action(
                p, action("key_press", point=[120, 130], key="ArrowRight"),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events], ["click", "key"])

    def test_drag_midmove_exception_still_releases_pressed_button(self):
        p = FakePage(fail_on="move")
        with self.assertRaisesRegex(RuntimeError, "pointer move failure"):
            execute_one_native_action(
                p, action("drag", start=[10, 10], end=[180, 220]),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events],
                         ["move", "down", "move", "up"])

    def test_drag_down_exception_still_attempts_release(self):
        p = FakePage(fail_on="down")
        with self.assertRaisesRegex(RuntimeError, "pointer down failure"):
            execute_one_native_action(
                p, action("drag", start=[10, 10], end=[180, 220]),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events], ["move", "down", "up"])

    def test_drag_redirect_after_down_stops_move_but_releases(self):
        p = FakePage(redirect_on="down")
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_one_native_action(
                p, action("drag", start=[10, 10], end=[180, 220]),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events], ["move", "down", "up"])

    def test_drag_redirect_on_end_move_still_releases(self):
        p = FakePage(redirect_on="move")
        with self.assertRaisesRegex(ValueError, "browser left"):
            execute_one_native_action(
                p, action("drag", start=[10, 10], end=[180, 220]),
                exact_original_loopback_url(8201))
        self.assertEqual([e[0] for e in p.events],
                         ["move", "down", "move", "up"])

    def test_unchanged_native_success_and_abstain(self):
        p = FakePage()
        res = execute_one_native_action(
            p, action("drag", start=[10, 10], end=[180, 220]),
            exact_original_loopback_url(8201))
        self.assertEqual(res["native_events"], 4)
        self.assertEqual([e[0] for e in p.events],
                         ["move", "down", "move", "up"])
        another = FakePage()
        res = execute_one_native_action(
            another, action("abstain"), another.url)
        self.assertEqual(res["native_events"], 0)
        self.assertEqual(another.events, [])


if __name__ == "__main__":
    unittest.main()
