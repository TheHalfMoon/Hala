"""H141 visible column semantic target never confused with hidden release."""
import json
import tempfile
import unittest
from pathlib import Path

from h141.visible_goal_source import SCHEMA,collect
from h141.interactive_source import original_case

class H141VisibleRestTargetSourceTests(unittest.TestCase):
    def test_genuine_unmodified_source_task_language(self):
        self.assertTrue(SCHEMA.endswith(".v1"))
        for i in range(44000,44016):
            case=original_case(i)
            self.assertEqual(case["language"],"en" if i%2==0 else "ar")
            self.assertGreaterEqual(case["destination_index"],0)
            self.assertIn(case["title"],case["instruction"])

    def test_create_only_and_no_fake_small_odd_split(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)/"private-h141"
            parent.mkdir()
            output=parent/"owner-original-approach"
            with self.assertRaisesRegex(ValueError,"create-only"):
                collect(Path(temp)/"no-upstream",output,count=3)
            with self.assertRaisesRegex(ValueError,"create-only"):
                collect(Path(temp)/"no-upstream",output,count=16,
                        attempt_offset=20000)
            output.mkdir()
            with self.assertRaisesRegex(ValueError,"create-only"):
                collect(Path(temp)/"no-upstream",output,count=16,
                        attempt_offset=44000)
if __name__=="__main__":
    unittest.main()
