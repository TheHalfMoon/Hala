"""H155 bounded typed action parser, negative/positive admission tests."""
import json
import unittest

from h155.action_contract import (
    action_from_dict, from_neural_response, model_capability_status,
)


class H155ActionProtocolTests(unittest.TestCase):
    def test_click_and_submit_points_must_be_finite_in_viewport(self):
        self.assertEqual(action_from_dict(
            {"kind": "click", "point": [300, 180]}).point, (300.0, 180.0))
        for point in ([-1, 100], [960, 100], [10, 721], [float("nan"), 5],
                      ["4", 8], None, [1], [1, 2, 3]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                action_from_dict({"kind": "click", "point": point})

    def test_text_action_escapes_invalid_or_unbounded_content(self):
        self.assertEqual(action_from_dict(
            {"kind": "type", "text": "A test"}).text, "A test")
        for value in ("", "x" * 129, "one\ntwo", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                action_from_dict({"kind": "type", "text": value})
        with self.assertRaises(ValueError):
            action_from_dict({"kind": "select", "text": "Italy", "dom": "#id"})

    def test_explicit_abstention_only_and_fail_closed(self):
        raw = json.dumps({"actions": [{"kind": "abstain"}]})
        self.assertEqual(from_neural_response(
            raw, task="click-test", source="h155-new-trained-model")[0].kind,
            "abstain")
        for value in ("not json", "{}", '{"actions":[]}',
                      '{"actions":[{"kind":"eval","text":"x"}]}'):
            with self.subTest(raw=value), self.assertRaises(ValueError):
                from_neural_response(
                    value, task="click-test", source="h155-new-trained-model")
        with self.assertRaises(ValueError):
            from_neural_response(json.dumps(
                {"actions":[{"kind":"click", "point":[2,2]},
                            {"kind":"abstain"}]}),
                task="click-test", source="h155-new-trained-model")

    def test_native_action_sequence_not_dom_gold(self):
        types = [
            {"kind":"click","point":[100,140]},
            {"kind":"type","text":"Nora"},
            {"kind":"submit","point":[200,160]},
        ]
        actions = from_neural_response(json.dumps({"actions":types}),
            task="enter-text", source="h155-new-trained-model")
        self.assertEqual([x.kind for x in actions],
                         ["click","type","submit"])
        with self.assertRaises(ValueError):
            from_neural_response(json.dumps({"actions":types[::-1]}),
                task="enter-text", source="h155-new-trained-model")
        with self.assertRaises(ValueError):
            from_neural_response(json.dumps({"actions":types}),
                task="enter-text", source="h142-frozen-research-lora")

    def test_prior_lo_ra_roles_abstain_not_claim_12_of_12(self):
        for role in ("start", "visible_column_heading"):
            result = model_capability_status(role, "choose-list")
            self.assertEqual(result["status"],
                             "UNSUPPORTED_ACTION_ROLE_ABSTAIN")
            self.assertIsNone(result["model_task_success_claim"])
            self.assertEqual(result["model_actions_attempted"], 0)


if __name__ == "__main__":
    unittest.main()
