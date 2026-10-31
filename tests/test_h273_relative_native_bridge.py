"""H273 deterministic normalized-point bridge invented-only CPU tests."""
import re
import unittest
from h273.relative_native_bridge import (
    WIDTH,HEIGHT,NMAX,NORMALIZED_REGEX,RELATIVE_PROMPT,
    FIRST_STOCK_TASK,normalized_to_pixel,normalize_strict_action,
    bind_relative_policy,
)
from h199.native_action_sft import parse_actions,source_task_prompt

class H273NormalizedContracts(unittest.TestCase):
    def test_only_exact_normalized_actions_convert_to_native_pixels(self):
        self.assertEqual((WIDTH,HEIGHT,NMAX),(960,720,999))
        self.assertEqual(normalized_to_pixel(0,WIDTH),0)
        self.assertEqual(normalized_to_pixel(500,WIDTH),480)
        self.assertEqual(normalized_to_pixel(500,HEIGHT),360)
        self.assertEqual(normalized_to_pixel(999,WIDTH),959)
        self.assertEqual(normalized_to_pixel(999,HEIGHT),719)
        for x,y in [(0,0),(500,500),(999,999),(750,100),(123,845)]:
            raw=f'{{"actions":[{{"kind":"click","point":[{x},{y}]}}]}}'
            self.assertIsNotNone(re.fullmatch(NORMALIZED_REGEX,raw))
            mapped=normalize_strict_action(raw)
            parsed=parse_actions(mapped)
            self.assertEqual(parsed["actions"][0]["point"],
                [normalized_to_pixel(x,960),normalized_to_pixel(y,720)])
        self.assertEqual(normalize_strict_action(
            '{"actions":[{"kind":"abstain"}]}'),
            '{"actions":[{"kind":"abstain"}]}')

    def test_reject_loose_json_or_unnatural_oracle_corrections(self):
        invalid=[
            '{"actions":[{"kind":"click","point":[1000,500]}]}',
            '{"actions":[{"kind":"click","point":[500,-1]}]}',
            '{"actions":[{"kind":"click","point":[500.0,500]}]}',
            '{"actions":[{"kind":"click","point":[050,10]}]}',
            '{"actions":[{"kind":"click","point":[true,40]}]}',
            '{"actions":[{"kind":"click","point":[500,500]}],"gold":[480,360]}',
            '{"actions":[{"kind":"click","point":[500,500]}]} trailing',
            "fenced_response_not_json",
            '{"actions":[{"kind":"drag","point":[40,50]}]}',
            '{"actions":[{"kind":"abstain"},{"kind":"abstain"}]}',
            '{"actions": [{"kind":"abstain"}]}',
        ]
        for item in invalid:
            with self.subTest(raw=item),self.assertRaises(ValueError):
                normalize_strict_action(item)
        for bad in [-1,1000,999.0,True,"500",None]:
            with self.subTest(coord=bad),self.assertRaises(ValueError):
                normalized_to_pixel(bad,WIDTH)
        self.assertNotIn("gold",RELATIVE_PROMPT.lower())
        self.assertIn("RELATIVE coordinate grid",RELATIVE_PROMPT)
        self.assertIn("FIRST visible inventory item",FIRST_STOCK_TASK)

    def test_predict_wrapper_checks_original_task_no_raw_storage(self):
        observed=[]
        raw='{"actions":[{"kind":"click","point":[500,500]}]}'
        def policy(png,prompt):
            observed.append((png,prompt))
            return raw
        wrapped=bind_relative_policy(policy)
        png=b"PRIVATE_NONIMAGE_FAKE"
        output=wrapped(png,source_task_prompt(FIRST_STOCK_TASK))
        self.assertEqual(output,
            '{"actions":[{"kind":"click","point":[480,360]}]}')
        self.assertEqual(len(observed),1)
        self.assertIs(observed[0][0],png)
        self.assertEqual(observed[0][1],RELATIVE_PROMPT)
        with self.assertRaises(ValueError):
            wrapped(b"not-an-image",source_task_prompt("other task"))
        self.assertEqual(len(observed),1)
        abstain=bind_relative_policy(lambda *_:'{"actions":[{"kind":"abstain"}]}')
        self.assertEqual(abstain(b"png",source_task_prompt(FIRST_STOCK_TASK)),
                         '{"actions":[{"kind":"abstain"}]}')


if __name__=="__main__":
    unittest.main()
