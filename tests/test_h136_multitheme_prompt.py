import unittest
from h136.prompt import role_prompt

class H136MultithemePrompts(unittest.TestCase):
    def test_start_end_prompts_do_not_assume_specific_palette(self):
        sentence="Drag the purple divider into the orange outlined region."
        start=role_prompt(sentence,"start")
        end=role_prompt(sentence,"end")
        self.assertIn("BEGIN",start)
        self.assertIn("FINISH",end)
        self.assertIn(sentence,start)
        self.assertNotEqual(start,end)
        self.assertNotIn("blue HANDLE",start)
        with self.assertRaises(ValueError):
            role_prompt(sentence,"anything")

if __name__=="__main__":
    unittest.main()
