"""H144 staged action pilot only consumes visible source/heading, not DOM drop oracle."""
import unittest
from collections import Counter
from h144.paired_action_pilot import (
    ATTEMPT_OFFSET,CASES,ARMS,case,staged_visible_sweep
)

class Mouse:
    def __init__(self):self.calls=[]
    def move(self,*a,**kw):self.calls.append(("move",a))
    def down(self):self.calls.append(("down",))
    def up(self):self.calls.append(("up",))
class Page:
    def __init__(self):self.mouse=Mouse();self.shots=0
    def wait_for_timeout(self,n):pass
    def screenshot(self,**kw):self.shots+=1;return b"static screenshot"
    def locator(self,*a,**kw):
        raise AssertionError("H144 controller cannot access hidden DOM")

class H144SourceIndependentControllerContracts(unittest.TestCase):
    def test_new_unseen_bilingual_destinations_and_layouts(self):
        self.assertEqual(ATTEMPT_OFFSET,82000)
        self.assertEqual(CASES,16)
        self.assertEqual(ARMS,("baseline","staged_sweep"))
        examples=[case(i) for i in range(82000,82016)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in examples),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in examples}),4)
        with self.assertRaises(ValueError):case(73000)

    def test_only_model_representable_heading_and_start_control_sweep(self):
        fake=Page()
        dest=staged_visible_sweep(
            fake,(211,280),
            {"x":430,"y":170,"width":120,"height":30})
        self.assertEqual(dest,[490,232])
        self.assertEqual(fake.shots,3)
        self.assertEqual(fake.mouse.calls[0],("move",(211,280)))
        self.assertEqual(fake.mouse.calls[1],("down",))
        self.assertEqual(fake.mouse.calls[-1],("up",))
        with self.assertRaises(ValueError):
            staged_visible_sweep(Page(),(1000,200),
                                 {"x":400,"y":180,"width":120,"height":30})
        with self.assertRaises(ValueError):
            staged_visible_sweep(Page(),(200,100),
                                 {"x":400,"y":675,"width":120,"height":30})
if __name__=="__main__":
    unittest.main()
