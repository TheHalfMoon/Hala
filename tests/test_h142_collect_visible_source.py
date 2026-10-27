"""H142 must use new original source IDs and never an invisible END label."""
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from h142.collect_visible_source import (
    seed_scene,collect,SCHEMA,COUNTS,OFFSETS,APP_SHA,DESTINATIONS
)

class H142NewVisibleSFTSourceContracts(unittest.TestCase):
    def test_fresh_stratified_bilingual_two_goal_source(self):
        self.assertEqual(COUNTS,{"train":64,"dev":16})
        self.assertEqual(OFFSETS,{"train":52000,"dev":62000})
        self.assertEqual(len(APP_SHA),40)
        self.assertTrue(SCHEMA.startswith("hala.h142"))
        for start,count in ((52000,64),(62000,16)):
            scenes=[seed_scene(n) for n in range(start,start+count)]
            quota=Counter((x["language"],x["destination_status"]) for x in scenes)
            self.assertEqual(quota,{(a,b):count//4
                              for a in ("ar","en") for b in DESTINATIONS})
            self.assertEqual(len({x["title"] for x in scenes}),count)
            for case in scenes:
                self.assertEqual(case["destination_column"],
                                 DESTINATIONS[case["destination_status"]])
                if case["destination_status"]=="done":
                    self.assertIn(case["destination_index"],(0,1))
                self.assertIn(case["title"],case["instruction"])
                self.assertNotIn("point_2d",case["instruction"])
                self.assertTrue(any(
                    card["task"]==case["title"] and card["status"]=="todo"
                    for card in case["rows"]))
    def test_old_attempts_and_invalid_splits_fail_closed_without_playwright(self):
        with self.assertRaises(ValueError):
            seed_scene(48000)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"private-h142"
            root.mkdir()
            for role in ("train","dev"):
                with self.assertRaisesRegex(ValueError,"create only"):
                    collect("missing",root/role,role,attempt_offset=48000)
            with self.assertRaises(ValueError):
                collect("missing",root/"no","blind")
            (root/"train").mkdir()
            with self.assertRaisesRegex(ValueError,"create only"):
                collect("missing",root/"train","train")
if __name__=="__main__":
    unittest.main()
