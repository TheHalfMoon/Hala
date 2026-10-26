"""Unit gates for owner-private original JavaScript H140 source collector."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h140.vanilla_pilot import SOURCE_SHA,check_source,proposal,visible_box


class H140VanillaOwnerPilotTests(unittest.TestCase):
    def test_proposed_real_layout_variation_stays_bounded_and_balanced(self):
        cases=[proposal(i) for i in range(64)]
        self.assertEqual({c[4] for c in cases},{"en","ar"})
        self.assertEqual(sum(c[4]=="en" for c in cases),32)
        self.assertEqual(len({c[0] for c in cases}),4)
        self.assertGreater(len({(c[1],c[2]) for c in cases}),9)
        for n,s,d,scroll,lang in cases:
            self.assertTrue(5<=n<=8)
            self.assertTrue(0<=s<n and 0<=d<n)
            self.assertTrue(0<=scroll<190)

    def test_source_hash_license_clean_checkout_and_no_remote_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            src=Path(root)
            (src/"LICENSE").write_text("MIT License\n\nCopyright (c) private original owner\n","utf8")
            (src/"index.html").write_text("<html></html>","utf8")
            (src/"script.js").write_text("const original=true","utf8")
            with patch("h140.vanilla_pilot.subprocess.check_output",
                       side_effect=[SOURCE_SHA+"\n",""]):
                self.assertEqual(check_source(src),SOURCE_SHA)
            with patch("h140.vanilla_pilot.subprocess.check_output",
                       side_effect=[SOURCE_SHA+"\n","script.js"]):
                with self.assertRaises(ValueError):
                    check_source(src)

    def test_visible_native_rect_rejects_offscreen_and_undersized(self):
        self.assertTrue(visible_box({"x":140,"y":200,"width":60,"height":70}))
        self.assertFalse(visible_box({"x":140,"y":680,"width":60,"height":70}))
        self.assertFalse(visible_box({"x":140,"y":-50,"width":60,"height":70}))
        self.assertFalse(visible_box({"x":140,"y":200,"width":60,"height":-5}))
        self.assertFalse(visible_box({"x":140,"y":200,"width":60,"height":0}))
        self.assertFalse(visible_box(None))


if __name__=="__main__":
    unittest.main()
