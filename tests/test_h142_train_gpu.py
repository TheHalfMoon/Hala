"""H142 old H141 hidden END failure addressed only with original VISIBLE boxes."""
import unittest
from collections import Counter
from h142.train_gpu import (
    role_prompt_and_gold,schedule,TRAIN_SHA,DEV_SHA,STEPS,ACCUM,
    HEADING_ROLE,START_ROLE,PRETRAINED_V2_ADAPTER_SHA
)

class H142RealVisibleHeadingTrainingContracts(unittest.TestCase):
    def rows(self):
        cases=[]
        for i in range(64):
            goal="done" if i%4 in (2,3) else "doing"
            cases.append({
                "instruction":("اسحب البطاقة H142 إلى عمود مكتمل"
                               if i%2 else "Drag H142 task to the "+goal+" column"),
                "language":"ar" if i%2 else "en",
                "destination_status":goal,
                "start_bbox":{"x":160,"y":300,"width":80,"height":40},
                "visible_destination_column_heading_bbox":{
                    "x":720 if goal=="done" else 420,
                    "y":170,"width":120,"height":30},
                "hidden_exact_drop_point_is_not_output_target":True,
            })
        return cases

    def test_exact_source_and_visible_heading_role_distribution(self):
        rows=self.rows()
        examples=schedule(rows)
        self.assertEqual(STEPS*ACCUM,192)
        self.assertEqual(Counter(role for _,role in examples),
                         {START_ROLE:64,HEADING_ROLE:128})
        self.assertEqual(examples,schedule(rows))
        self.assertEqual(len(TRAIN_SHA),64)
        self.assertEqual(len(DEV_SHA),64)
        self.assertEqual(len(PRETRAINED_V2_ADAPTER_SHA),64)

    def test_goal_is_visible_column_heading_not_hidden_target(self):
        row=self.rows()[3]
        _,gold=role_prompt_and_gold(row,HEADING_ROLE)
        self.assertEqual(__import__("json").loads(gold),{
            "point_2d":[812,257]})
        prompt,_=role_prompt_and_gold(row,START_ROLE)
        self.assertIn("BEGIN",prompt)
        with self.assertRaises(ValueError):
            role_prompt_and_gold(row,"end")
        with self.assertRaises(ValueError):
            role_prompt_and_gold(row,"hidden_drop")
if __name__=="__main__":
    unittest.main()
