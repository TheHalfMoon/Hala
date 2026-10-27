"""H163 original MIT dashboard source splits; no model or gold in prompts."""
import unittest
from collections import Counter
from h163.source import specs, GIT_SHA, NPM_VERSION, NPM_ARCHIVE_SHA, instruction

class H163Tests(unittest.TestCase):
    def test_external_version_pins(self):
        self.assertEqual(len(GIT_SHA),40)
        self.assertEqual(len(NPM_ARCHIVE_SHA),64)
        self.assertEqual(NPM_VERSION,"14.0.0")

    def test_bilingual_two_task_16_scene_balance(self):
        rows=specs()
        self.assertEqual(len(rows),16)
        self.assertEqual(len({(r["layout"],r["pre_column"],r["task"]) for r in rows}),16)
        self.assertEqual(Counter((r["task"],r["language"]) for r in rows),
                         {("set_column","en"):4,("set_column","ar"):4,
                          ("add_widget","en"):4,("add_widget","ar"):4})
        for row in rows:
            if row["task"]=="set_column":
                self.assertNotEqual(row["pre_column"],row["target_column"])

    def test_arabic_visible_tasks(self):
        for row in specs():
            value=instruction(row)
            if row["language"]=="ar":
                self.assertTrue(any("\u0600"<=x<="\u06ff" for x in value))
            else:
                self.assertTrue(value.isascii())

if __name__=="__main__":
    unittest.main()
