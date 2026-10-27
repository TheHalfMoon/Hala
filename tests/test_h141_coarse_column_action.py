"""H141 original coarse column controller cannot see hidden release DOM."""
import unittest
from h141.coarse_column_action import approach_point_from_visible_column_heading,move_card_by_coarse_visible_goal

class FakeMouse:
    def __init__(self):self.calls=[]
    def move(self,*args,**kwargs):self.calls.append(("move",args))
    def down(self):self.calls.append(("down",))
    def up(self):self.calls.append(("up",))

class FakePage:
    def __init__(self):self.mouse=FakeMouse()
    def wait_for_timeout(self,ms):self.waited=ms
    def locator(self,*a,**k):raise AssertionError("H141 controller must not inspect DOM")

class H141VisibleOriginalColumnActionTests(unittest.TestCase):
    def test_only_visible_start_and_heading_control_drag_action(self):
        heading={"x":420,"y":165,"width":120,"height":31}
        self.assertEqual(approach_point_from_visible_column_heading(heading),
                         (480,208))
        page=FakePage()
        proposed=move_card_by_coarse_visible_goal(page,(230,350),heading)
        self.assertEqual(proposed,(480,208))
        self.assertEqual(page.mouse.calls[0],("move",(230,350)))
        self.assertEqual(page.mouse.calls[1],("down",))
        self.assertEqual(page.mouse.calls[-1],("up",))
        self.assertEqual(page.waited,150)

    def test_invalid_out_of_view_oracle_positions_rejected(self):
        with self.assertRaises(ValueError):
            approach_point_from_visible_column_heading(
                {"x":480,"y":700,"width":140,"height":35})
        with self.assertRaises(ValueError):
            move_card_by_coarse_visible_goal(
                FakePage(),(-100,350),
                {"x":420,"y":165,"width":120,"height":31})

if __name__=="__main__":
    unittest.main()
