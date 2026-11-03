"""Optional first-party real browser H407 review boundary, no external UI data."""
import os
import unittest
from h407.original_reviewed_browser_smoke import qualify_local_original


@unittest.skipUnless(
    os.environ.get("HALA_H407_BROWSER_QUALIFY")=="1",
    "Real first-party original browser smoke requires explicit opt-in")
class H407RealSourceReviewBrowserTests(unittest.TestCase):
    def test_real_local_browser_click_requires_source_unique_review(self):
        channel=os.environ.get("HALA_H407_BROWSER_CHANNEL","msedge")
        result=qualify_local_original(channel)
        self.assertEqual(result["status"],
                         "H407_FIRST_PARTY_FIXTURE_REVIEW_BOUNDARY_PASS")
        self.assertEqual(result["reviewed_unique_source_actions"],1)
        self.assertEqual(result["ambiguous_source_actions_refused"],1)
        self.assertFalse(result["real_hala_model_used"])
        self.assertFalse(result["externally_authenticated_reviewer"])
        self.assertFalse(result["independent_ood"])
        self.assertFalse(result["model_release_qualified"])

if __name__=="__main__":
    unittest.main()
