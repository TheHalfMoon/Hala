"""H263 invented action-contract tests; no media, model or private gold."""
import json
import unittest
from unittest.mock import Mock

from h263.native_browser_executor import execute_one, new_offline_context


def action(kind, **fields):
    return json.dumps({"actions": [{"kind": kind, **fields}]}, ensure_ascii=False)


def fake_page():
    page = Mock()
    page.viewport_size = {"width": 960, "height": 720}
    page.evaluate.return_value = True
    return page


class NativeBrowserExecutorTests(unittest.TestCase):
    def test_offline_fixed_viewport_context(self):
        browser = Mock()
        result = new_offline_context(browser)
        self.assertIs(result, browser.new_context.return_value)
        self.assertEqual(browser.new_context.call_args.kwargs, {
            "viewport": {"width": 960, "height": 720},
            "offline": True, "service_workers": "block",
            "accept_downloads": False, "permissions": [],
        })

    def test_click_is_executed_once(self):
        page = fake_page()
        result = execute_one(page, action("click", point=[45, 67]))
        page.mouse.click.assert_called_once_with(45, 67)
        self.assertEqual(result, {"kind": "click", "executed": True})

    def test_drag_has_both_endpoints_and_release(self):
        page = fake_page()
        result = execute_one(page, action("drag", start=[30, 40], end=[200, 300]))
        page.mouse.move.assert_any_call(30, 40)
        page.mouse.move.assert_any_call(200, 300, steps=12)
        page.mouse.down.assert_called_once_with()
        page.mouse.up.assert_called_once_with()
        self.assertEqual(result["kind"], "drag")

    def test_keyboard_focus_then_press(self):
        page = fake_page()
        execute_one(page, action("key_press", point=[55, 55], key="Escape"))
        page.mouse.click.assert_called_once_with(55, 55)
        page.keyboard.press.assert_called_once_with("Escape")

    def test_unicode_text_exact(self):
        page = fake_page()
        result = execute_one(page, action("type_text", text="مخزون 42"))
        page.keyboard.insert_text.assert_called_once_with("مخزون 42")
        self.assertNotIn("text", result)
        self.assertNotIn("point", result)

    def test_type_text_rejects_unfocused_editable(self):
        page = fake_page()
        page.evaluate.return_value = False
        with self.assertRaisesRegex(ValueError, "focused editable"):
            execute_one(page, action("type_text", text="Arabic"))
        page.keyboard.insert_text.assert_not_called()

    def test_scroll_is_bounded_and_exact(self):
        page = fake_page()
        execute_one(page, action("scroll", point=[120, 250], dy=240))
        page.mouse.move.assert_called_once_with(120, 250)
        page.mouse.wheel.assert_called_once_with(0, 240)

    def test_abstain_is_noop(self):
        page = fake_page()
        result = execute_one(page, action("abstain"))
        self.assertEqual(result, {"kind": "abstain", "executed": False})
        page.mouse.click.assert_not_called()
        page.keyboard.press.assert_not_called()

    def test_wrong_viewport_rejected_without_action(self):
        page = fake_page()
        page.viewport_size = {"width": 800, "height": 600}
        with self.assertRaisesRegex(ValueError, "960x720"):
            execute_one(page, action("click", point=[20, 20]))
        page.mouse.click.assert_not_called()

    def test_duplicate_json_keys_rejected(self):
        page = fake_page()
        with self.assertRaises(ValueError):
            execute_one(page, '{"actions":[{"kind":"click","kind":"drag","point":[2,3]}]}')
        page.mouse.click.assert_not_called()

    def test_out_of_bounds_action_rejected(self):
        page = fake_page()
        with self.assertRaises(ValueError):
            execute_one(page, action("click", point=[960, 0]))
        page.mouse.click.assert_not_called()

    def test_nonstandard_number_rejected(self):
        page = fake_page()
        with self.assertRaises(ValueError):
            execute_one(page, '{"actions":[{"kind":"click","point":[NaN,2]}]}')
        page.mouse.click.assert_not_called()

    def test_multiple_actions_rejected(self):
        page = fake_page()
        with self.assertRaises(ValueError):
            execute_one(page, '{"actions":[{"kind":"abstain"},{"kind":"abstain"}]}')
        page.mouse.click.assert_not_called()

    def test_no_action_payload_in_return(self):
        page = fake_page()
        result = execute_one(page, action("scroll", point=[18, 26], dy=-180))
        self.assertEqual(set(result), {"kind", "executed"})


if __name__ == "__main__":
    unittest.main()
