"""H201 original native JSON-to-pointer executor hardening and receipt tests.

Fake page unit tests NEVER represent real browser proof. The separate
pinned original interact.js Playwright workflow tests actual native reward.
"""
import json
import unittest
from dataclasses import replace

from h193.interact_snap_preaction import SPECS
from h201.original_native_action_executor import (
    SnapNativeActionReceipt, execute_one_native_action,
    exact_original_loopback_url, validate_original_execution,
)


class Mouse:
    def __init__(self, log):
        self.log = log
    def click(self, x, y):
        self.log.append(("click", x, y))
    def move(self, x, y, **kwargs):
        self.log.append(("move", x, y, kwargs))
    def down(self):
        self.log.append(("down",))
    def up(self):
        self.log.append(("up",))


class Keyboard:
    def __init__(self, log):
        self.log = log
    def press(self, key):
        self.log.append(("press", key))


class Page:
    def __init__(self):
        self.url = exact_original_loopback_url(8201)
        self.viewport_size = {"width":960, "height":720}
        self.events = []
        self.mouse = Mouse(self.events)
        self.keyboard = Keyboard(self.events)


def one(action):
    return json.dumps({"actions":[action]}, separators=(",", ":"))


def receipts():
    return tuple(SnapNativeActionReceipt(
        task_id=s["id"], language=s["language"],
        before_input="300", after_input="301", other_input="499",
        before_png_sha256=("1" if i == 0 else "2")*64,
        before_rgba_sha256=("a" if i == 0 else "b")*64,
        after_rgba_sha256=("c" if i == 0 else "d")*64,
        before_grid_sha256=("e" if i == 0 else "f")*64,
        after_grid_sha256=("3" if i == 0 else "4")*64,
        target_center=(500.0+i, 230.0+i),
        native_events=2,
    ) for i, s in enumerate(SPECS) for _ in range(2))


class H201ActionExecutorTests(unittest.TestCase):
    def test_keypress_is_real_click_then_keyboard_and_no_dom(self):
        p=Page()
        result=execute_one_native_action(p, one({
            "kind":"key_press","point":[500,230],"key":"ArrowRight",
        }),p.url)
        self.assertEqual(result["native_events"],2)
        self.assertEqual(p.events,[("click",500,230),("press","ArrowRight")])
        self.assertEqual(result["kind"],"key_press")

    def test_click_goes_to_native_browser_mouse(self):
        p=Page()
        result=execute_one_native_action(p,one({
            "kind":"click","point":[5,6]
        }),p.url)
        self.assertEqual(result["native_events"],1)
        self.assertEqual(p.events,[("click",5,6)])

    def test_drag_is_native_pointer_sequence_not_dom_value_setter(self):
        p=Page()
        result=execute_one_native_action(p,one({
            "kind":"drag","start":[12,16],"end":[312,218],
        }),p.url)
        self.assertEqual(result["native_events"],4)
        self.assertEqual([e[0] for e in p.events],
                         ["move","down","move","up"])
        self.assertEqual(p.events[2][3],{"steps":12})

    def test_abstain_never_moves_pointer_or_sends_keyboard(self):
        p=Page()
        x=execute_one_native_action(p,one({"kind":"abstain"}),p.url)
        self.assertEqual(x["status"],"ABSTAIN_NO_BROWSER_INPUT")
        self.assertEqual(x["native_events"],0)
        self.assertEqual(p.events,[])

    def test_unregistered_external_navigation_aborts_before_browser_events(self):
        for url in ("https://malicious.invalid",
                    "http://127.0.0.1:8201/other.html",
                    "http://127.0.0.1:8202/snap/index.html"):
            with self.subTest(url=url):
                p=Page()
                p.url=url
                with self.assertRaisesRegex(ValueError,"browser left"):
                    execute_one_native_action(p,one({
                        "kind":"click","point":[1,3]
                    }),exact_original_loopback_url(8201))
                self.assertEqual(p.events,[])

    def test_unregistered_expected_url_rejected_first(self):
        p=Page()
        for expected in ("http://evil.invalid/snap/index.html",
                         "https://127.0.0.1:8201/snap/index.html",
                         "http://127.0.0.1:8201/not-snap/index.html"):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    execute_one_native_action(p,one({
                        "kind":"click","point":[5,5]
                    }),expected)
                self.assertEqual(p.events,[])

    def test_incorrect_screenshot_resolution_fails_before_input(self):
        p=Page()
        p.viewport_size={"width":1920,"height":1080}
        with self.assertRaisesRegex(ValueError,"viewport"):
            execute_one_native_action(p,one({
                "kind":"click","point":[960,300]
            }),p.url)
        self.assertEqual(p.events,[])

    def test_model_duplicate_keys_and_nonstandard_constants_rejected_no_input(self):
        for raw in (
            '{"actions":[{"kind":"click","kind":"drag","point":[10,20]}]}',
            '{"actions":[{"kind":"click","point":[10,NaN]}]}',
            '{"actions":[{"kind":"key_press","point":[1,2],"key":"Ctrl+A"}]}',
            '{"actions":[{"kind":"click","point":[10,20],"selector":"#private"}]}',
        ):
            with self.subTest(raw=raw):
                p=Page()
                with self.assertRaises(ValueError):
                    execute_one_native_action(p,raw,p.url)
                self.assertEqual(p.events,[])

    def test_native_source_receipt_two_bilingual_scenes_positive(self):
        r=validate_original_execution(receipts())
        self.assertEqual(r["original_source_tasks"],2)
        self.assertEqual(r["fresh_contexts"],4)
        self.assertEqual(r["real_browser_input_events"],8)
        self.assertEqual(r["changed_real_original_grid_canvases"],4)
        self.assertEqual(r["media_saved"],0)
        self.assertEqual(r["model_calls"],0)
        self.assertFalse(r["old_ood_read"])

    def test_missing_context_or_wrong_native_slider_reward_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing fresh"):
            validate_original_execution(receipts()[:-1])
        for key,value in (("after_input","300"),("other_input","300"),
                          ("native_events",1)):
            with self.subTest(key=key):
                rs=list(receipts())
                rs[0]=replace(rs[0],**{key:value})
                with self.assertRaisesRegex(ValueError,"reward"):
                    validate_original_execution(tuple(rs))

    def test_spoofed_focus_only_pixels_without_grid_redraw_rejected(self):
        rs=list(receipts())
        rs[0]=replace(rs[0],
            after_grid_sha256=rs[0].before_grid_sha256)
        with self.assertRaisesRegex(ValueError,"grid canvas"):
            validate_original_execution(tuple(rs))

    def test_preaction_raster_diff_between_fresh_contexts_is_rejected(self):
        for key,val in (
            ("before_png_sha256","9"*64),
            ("before_rgba_sha256","8"*64),
            ("target_center",(490.0,205.0)),
        ):
            with self.subTest(key=key):
                rs=list(receipts())
                rs[1]=replace(rs[1],**{key:val})
                with self.assertRaisesRegex(ValueError,"differs across fresh"):
                    validate_original_execution(tuple(rs))

    def test_two_original_languages_cannot_use_same_before_pixels(self):
        for key,val in (
            ("before_png_sha256","1"*64),
            ("before_rgba_sha256","a"*64),
        ):
            with self.subTest(key=key):
                rs=list(receipts())
                for n in (2,3):
                    rs[n]=replace(rs[n],**{key:val})
                with self.assertRaisesRegex(ValueError,"identical GUI"):
                    validate_original_execution(tuple(rs))


if __name__ == "__main__":
    unittest.main()
