"""H148 frozen new owner-only release normalizing hypothesis."""
import unittest
from collections import Counter
from h148.snapback_action_ab import (
    OFFSET,N,SNAP_TO_HEADING_OFFSET,new_case
)
class H148ReleaseSnapbackContracts(unittest.TestCase):
    def test_new_bilingual_two_goal_source_not_consumed(self):
        self.assertEqual(OFFSET,124000)
        self.assertEqual(N,16)
        rows=[new_case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in rows),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in rows}),4)
        with self.assertRaises(ValueError):new_case(114000)
    def test_snap_offset_frozen_before_new_native_test(self):
        self.assertEqual(SNAP_TO_HEADING_OFFSET,18)
if __name__=="__main__":
    unittest.main()
