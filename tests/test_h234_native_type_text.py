"""H234 synthetic contract and real offline browser QA; no private media/gold."""
import json
from pathlib import Path
import tempfile
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h234.native_type_text import (
    parse_actions_v4, replay_local_type_text, source_task_prompt_v4,
    synthetic_sft_messages_v4,
)


def payload(action):
    return json.dumps({"actions": [action]}, ensure_ascii=False)


class H234SyntheticTypeTextTests(unittest.TestCase):
    def test_bilingual_one_action_and_h199_backwards_compatibility(self):
        for phrase in ("Enter the name", "أدخل اسم المريض التجريبي"):
            action = {"kind": "type_text", "point": [140.5, 225], "text": phrase}
            self.assertEqual(parse_actions_v4(payload(action))["actions"], [action])
            self.assertEqual(
                parse_actions_v4(payload(action), training_target=True)["actions"],
                [action],
            )
        for action in (
            {"kind": "click", "point": [15, 20]},
            {"kind": "drag", "start": [20, 25], "end": [70, 85]},
            {"kind": "key_press", "point": [20, 25], "key": "ArrowRight"},
            {"kind": "abstain"},
        ):
            result = parse_actions_v4(payload(action))
            self.assertEqual(result["actions"], [action])
        with self.assertRaises(ValueError):
            parse_actions_v4(payload({"kind": "abstain"}), training_target=True)

    def test_bounded_and_validated_text(self):
        for text in (
            "", " ", "  hello", "hello ", "line\nbreak", "tab\t",
            "null\x00", "hidden\u200btext", "direction\u202etext",
            "private\ue000", "a\u0301", "a" * 161, "م" * 161, "😀" * 100,
        ):
            with self.subTest(value=repr(text)), self.assertRaises(ValueError):
                parse_actions_v4(payload({
                    "kind": "type_text", "point": [100, 120], "text": text,
                }))
        self.assertEqual(
            parse_actions_v4(payload({
                "kind": "type_text", "point": [100, 120],
                "text": "الفاتورة ١٢٣ — paid",
            }))["actions"][0]["text"],
            "الفاتورة ١٢٣ — paid",
        )

    def test_wrong_coordinates_and_metadata_denied(self):
        for point in ([960, 10], [10, 720], [-1, 5], [True, 5], [5], [5, "8"]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                parse_actions_v4(payload({
                    "kind": "type_text", "point": point, "text": "Hello",
                }))
        for key in ("dom_id", "selector", "reference_gold", "shell", "after_action"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_actions_v4(payload({
                    "kind": "type_text", "point": [100, 100],
                    "text": "Hello", key: "hidden",
                }))

    def test_malformed_duplicate_and_multiple_actions_denied(self):
        for raw in (
            '{"actions":[{"kind":"type_text","point":[20,30],"text":"a","text":"b"}]}',
            '{"actions":[{"kind":"type_text","point":[20,30],"text":"a"}],"actions":[]}',
            '{"actions":[{"kind":"type_text","point":[20,30],"text":"a"},'
            '{"kind":"click","point":[10,10]}]}',
            '{"actions":[{"kind":"type_text","point":[20,NaN],"text":"a"}]}',
            '{"actions":[{"kind":"type_text","point":[20,30],"text":"a"}],"gold":1}',
            "",
            "a" * 1300,
        ):
            with self.subTest(raw=raw[:50]), self.assertRaises(ValueError):
                parse_actions_v4(raw)

    def test_screenshot_only_multimodal_message_and_training_denied(self):
        task = "اكتب مرحبا في الحقل الظاهر"
        action = payload({
            "kind": "type_text", "point": [220, 160], "text": "مرحبا",
        })
        before, with_answer = synthetic_sft_messages_v4(task, action)
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["content"][0], {"type": "image"})
        self.assertIn(task, before[0]["content"][1]["text"])
        self.assertIn("type_text", before[0]["content"][1]["text"])
        self.assertIn("screenshot", before[0]["content"][1]["text"])
        self.assertNotIn("selector", before[0]["content"][1]["text"])
        self.assertEqual(with_answer[-1]["role"], "assistant")
        self.assertEqual(
            json.loads(with_answer[-1]["content"][0]["text"])["actions"][0]["text"],
            "مرحبا",
        )
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())

    def test_invalid_user_task_refused_by_h199_gate(self):
        for task in ("", "x" * 501, None, "\x00"):
            with self.subTest(task=repr(task)), self.assertRaises(ValueError):
                source_task_prompt_v4(task)

    def test_unadmitted_actions_never_rewritten_to_legacy(self):
        with self.assertRaises(ValueError):
            parse_actions_v4(payload({
                "kind": "type", "point": [100, 100], "text": "not typed",
            }))


class H234OfflineBrowserTests(unittest.TestCase):
    def test_real_native_bilingual_type_without_dom_model_action(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Playwright not installed on hosted CPU")
        with sync_playwright() as p:
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            kwargs = {"headless": True}
            if edge.is_file():
                kwargs["executable_path"] = str(edge)
            elif not Path(p.chromium.executable_path).is_file():
                self.skipTest("No preinstalled Chromium/Edge on hosted CPU")
            with tempfile.TemporaryDirectory(prefix="h234-synthetic-browser-") as temp:
                html = Path(temp) / "synthetic.html"
                html.write_text(
                    '<!doctype html><meta charset="utf-8">'
                    '<style>body{margin:0}input{position:absolute;top:130px;'
                    'left:100px;width:330px;height:42px;font:22px Arial}</style>'
                    '<input id="field" aria-label="synthetic demo field">'
                    '<div id="state" data-edited="no"></div>'
                    '<script>document.querySelector("#field").addEventListener('
                    '"input",()=>{document.querySelector("#state").dataset.edited='
                    '"yes"})</script>',
                    encoding="utf-8",
                )
                browser = p.chromium.launch(**kwargs)
                try:
                    context = browser.new_context(
                        viewport={"width": 960, "height": 720},
                        service_workers="block",
                    )
                    attempted = []
                    context.route(
                        "**/*",
                        lambda route: route.continue_()
                        if route.request.url.startswith(html.parent.as_uri() + "/")
                        else (attempted.append(route.request.url), route.abort()),
                    )
                    page = context.new_page()
                    for text in ("Hello synthetic", "مرحبا بالعالم"):
                        page.goto(html.as_uri(), wait_until="load")
                        result = replay_local_type_text(page, payload({
                            "kind": "type_text", "point": [220, 155], "text": text,
                        }))
                        self.assertEqual(page.locator("#field").input_value(), text)
                        self.assertEqual(
                            page.locator("#state").get_attribute("data-edited"),
                            "yes",
                        )
                        self.assertFalse(result["training_approved"])
                        self.assertEqual(result["model_calls"], 0)
                        self.assertEqual(result["optimizer_steps"], 0)
                    self.assertEqual(attempted, [])
                    context.close()
                finally:
                    browser.close()

    def test_browser_replay_rejects_non_local_url(self):
        class FakePage:
            url = "https://example.com/test"
        with self.assertRaisesRegex(ValueError, "offline file"):
            replay_local_type_text(FakePage(), payload({
                "kind": "type_text", "point": [100, 100], "text": "Hi",
            }))


if __name__ == "__main__":
    unittest.main()
