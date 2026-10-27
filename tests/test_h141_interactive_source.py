"""H141 source-integrity and hidden-target dynamic-reveal experiment."""
import unittest
from collections import Counter

from h141.interactive_source import APP,APP_SHA,SCHEMA,language,original_case

class H141OriginalTwoStageSourceTests(unittest.TestCase):
    def test_separate_new_bilingual_owner_attempts(self):
        self.assertEqual(len(APP_SHA),40)
        self.assertEqual(APP,"react-native-board")
        self.assertIn("source-qa",SCHEMA)
        cases=[original_case(i) for i in range(41000,41080)]
        self.assertEqual(Counter(x["language"] for x in cases),{"en":40,"ar":40})
        self.assertEqual(len({x["title"] for x in cases}),80)
        for x in cases:
            self.assertEqual(sum(
                r["task"]==x["title"] and r["status"]=="todo"
                for r in x["rows"]),1)
            self.assertGreater(sum(r["status"]=="doing" for r in x["rows"]),0)
            self.assertNotIn("point_2d",x["instruction"])
            self.assertTrue(0<=x["source_index"]<len(x["rows"]))
            self.assertTrue(0<=x["destination_index"]<8)
            self.assertIn(x["title"],x["instruction"])
    def test_no_history_or_h140_reuse(self):
        with self.assertRaises(ValueError):
            original_case(20000)
        self.assertEqual(language(41000),"en")
        self.assertEqual(language(41001),"ar")

if __name__=="__main__":
    unittest.main()
