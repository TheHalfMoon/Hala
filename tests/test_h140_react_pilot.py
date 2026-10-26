"""CPU-only source and geometry checks for original React reference harness."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h140.react_pilot import SOURCE_SHA,midpoint,verify_app

class H140OriginalReactPilotContracts(unittest.TestCase):
    def test_pinned_unmodified_mit_app_build_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"LICENSE").write_text("MIT License\nCopyright original creator",encoding="utf8")
            (root/"dist").mkdir()
            (root/"dist"/"index.html").write_text("<html></html>",encoding="utf8")
            with patch("h140.react_pilot.subprocess.check_output",
                       side_effect=[SOURCE_SHA+"\n",""]):
                self.assertEqual(verify_app(root),root/"dist")
            with patch("h140.react_pilot.subprocess.check_output",
                       side_effect=[SOURCE_SHA+"\n","modified tracked app"]):
                with self.assertRaises(ValueError):
                    verify_app(root)

    def test_native_owner_bbox_center_is_not_a_model_prediction(self):
        self.assertEqual(midpoint({"x":20,"y":30,"width":40,"height":60}),[40,60])

if __name__=="__main__":
    unittest.main()
