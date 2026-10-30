"""H249 safe bilingual text + vertical scroll proposal contract; synthetic only."""
import json
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, MAX_JSON_BYTES, parse_actions,
    source_task_prompt, training_messages_for_synthetic_contract,
)

def encode(action):
    return json.dumps({"actions": [action]}, ensure_ascii=False)


class H249BilingualTextScrollTests(unittest.TestCase):
    def test_english_and_arabic_literal_text_are_single_actions(self):
        for literal in ("Project notes", "أضف مهمة جديدة", "تَحْرِيرُ نص عربي", "الأربعاء 2026"):
            with self.subTest(literal=literal):
                action = {"kind": "type_text", "text": literal}
                self.assertEqual(
                    parse_actions(encode(action), training_target=True),
                    {"actions": [action]},
                )
                prompt, messages = training_messages_for_synthetic_contract(
                    "Enter the requested words in the already focused field.",
                    encode(action),
                )
                self.assertEqual(prompt[0]["content"][0], {"type": "image"})
                self.assertEqual(
                    json.loads(messages[-1]["content"][0]["text"])["actions"][0],
                    action,
                )
                self.assertIn("SEPARATE step", prompt[0]["content"][1]["text"])

    def test_type_never_supplies_dom_or_fake_focus_or_extra_execution(self):
        for action in (
            {"kind": "type_text", "text": "hi", "selector": "#password"},
            {"kind": "type_text", "text": "hi", "point": [10, 20]},
            {"kind": "type_text", "text": "hi", "key": "Enter"},
            {"kind": "type_text", "text": "hi", "focused": True},
            {"kind": "type_text", "text": "hi", "shell": "cmd"},
        ):
            with self.subTest(action=action), self.assertRaises(ValueError):
                parse_actions(encode(action))

    def test_control_bidi_surrogate_private_use_and_long_text_denied(self):
        for bad in (
            "", "x" * 161, "🎨" * 129,
            "line\nnext", "one\rnext", "x\ty",
            "text\x00", "abc\u202edef", "abc\u2066def",
            "abc\u200ddef", "\ue000", "\ud800",
            "\u2028", 32, None, True, ["text"],
        ):
            with self.subTest(bad=repr(bad)[:25]), self.assertRaises(ValueError):
                parse_actions(encode({"kind": "type_text", "text": bad}))
        with self.assertRaises(ValueError):
            parse_actions('{"actions":[{"kind":"type_text","text":"\\ud800"}]}')

    def test_vertical_scroll_exact_bounded_schema(self):
        for point in ([0, 0], [959.5, 719.5]):
            for dy in (-720, -1, 1, 240, 720):
                with self.subTest(point=point, dy=dy):
                    action = {"kind": "scroll", "point": point, "dy": dy}
                    self.assertEqual(
                        parse_actions(encode(action), training_target=True),
                        {"actions": [action]},
                    )

    def test_scroll_requires_visible_point_and_nonzero_int_delta(self):
        for bad in (
            {"kind": "scroll", "point": [0, 0], "dy": 0},
            {"kind": "scroll", "point": [0, 0], "dy": 721},
            {"kind": "scroll", "point": [0, 0], "dy": -721},
            {"kind": "scroll", "point": [0, 0], "dy": 1.5},
            {"kind": "scroll", "point": [0, 0], "dy": True},
            {"kind": "scroll", "point": [960, 0], "dy": 5},
            {"kind": "scroll", "point": [-1, 0], "dy": 5},
            {"kind": "scroll", "point": [0, 0], "pixels": 5},
            {"kind": "scroll", "point": [0, 0], "dy": 5, "dom": ".list"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_actions(encode(bad))

    def test_still_exactly_one_action_and_no_abstain_as_train_gold(self):
        for actions in (
            [
                {"kind": "click", "point": [1, 2]},
                {"kind": "type_text", "text": "hello"},
            ],
            [
                {"kind": "scroll", "point": [30, 40], "dy": 120},
                {"kind": "click", "point": [1, 2]},
            ],
            [],
            [{"kind": "abstain"}],
        ):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                parse_actions(
                    json.dumps({"actions": actions}), training_target=True
                )

    def test_duplicate_json_and_max_bytes_remain_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_actions(
                '{"actions":[{"kind":"type_text","text":"safe","text":"override"}]}'
            )
        with self.assertRaises(ValueError):
            parse_actions(" " * (MAX_JSON_BYTES + 1))

    def test_no_dataset_admission_or_model_claim(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        self.assertIn("screenshot-only", source_task_prompt("اكتب كلمة مرحبا"))


if __name__ == "__main__":
    unittest.main()
