"""H164 strict typed screenshot-space action contract tests."""
import json
import unittest
from h164.native_action_contract import parse_action,supervision,visible_prompt

class H164NativeActionContractTests(unittest.TestCase):
    def test_exact_click_drag_and_type_schema(self):
        for item,kind in [
            ({"kind":"click","point":[123,234]},"click"),
            ({"kind":"drag","start":[24,37],"end":[440,430]},"drag"),
            ({"kind":"type","point":[120,140],"text":"Ana"},"type"),
            ({"kind":"abstain"},"abstain")
        ]:
            self.assertEqual(parse_action(json.dumps({"actions":[item]}))["kind"],kind)

    def test_no_dom_or_extra_model_instructions_accepted(self):
        forbidden=[
            {"actions":[{"kind":"click","point":[1,2],"selector":"button"}]},
            {"actions":[{"kind":"click","point":[-1,10]}]},
            {"actions":[{"kind":"click","point":[960,720]}]},
            {"actions":[{"kind":"drag","start":[8,8],"end":[15,720]}]},
            {"actions":[{"kind":"type","point":[4,4],"text":"\n"}]},
            {"actions":[{"kind":"type","point":[4,4],"text":"x"*129}]},
            {"actions":[{"kind":"click","point":[2,3]},{"kind":"abstain"}]},
            {"actions":[{"kind":"abstain","reason":"internal DOM"}]},
        ]
        for item in forbidden:
            self.assertFalse(parse_action(json.dumps(item))["valid"])

    def test_gold_available_for_train_only(self):
        row={"split":"TRAIN","action_type":"type","instruction":"اكتب Ana",
             "training_supervision":{"actions":[
                {"kind":"type","point":[100,120],"text":"Ana"}]}}
        self.assertEqual(parse_action(supervision(row))["kind"],"type")
        self.assertIn("اكتب Ana",visible_prompt(row))
        self.assertIn('"kind":"type"',visible_prompt(row))
        without=dict(row)
        del without["training_supervision"]
        with self.assertRaisesRegex(ValueError,"TRAIN gold"):
            visible_prompt(without)
        with self.assertRaisesRegex(ValueError,"gold supervision"):
            supervision(without)
