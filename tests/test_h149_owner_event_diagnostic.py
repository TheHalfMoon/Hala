"""H149 source-only owner browser event telemetry contract."""
import unittest
from collections import Counter
from h149.owner_event_diagnostic import case,OFFSET,N,EVENT_HOOK
class H149OwnerEventDiagnostics(unittest.TestCase):
    def test_new_bilingual_original_source_and_no_old_holdout(self):
        self.assertEqual(OFFSET,134000)
        self.assertEqual(N,16)
        scenes=[case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((s["language"],s["destination_status"])
                                 for s in scenes),
                         {("en","doing"):4,("ar","doing"):4,
                          ("en","done"):4,("ar","done"):4})
        self.assertEqual(len({s["layout"] for s in scenes}),4)
        with self.assertRaises(ValueError):
            case(124000)
    def test_owner_only_event_observer_is_capture_not_controller(self):
        self.assertIn("dragstart",EVENT_HOOK)
        self.assertIn("dragenter",EVENT_HOOK)
        self.assertIn("drop",EVENT_HOOK)
        self.assertIn(".task_column_heading",EVENT_HOOK)
if __name__=="__main__":
    unittest.main()
