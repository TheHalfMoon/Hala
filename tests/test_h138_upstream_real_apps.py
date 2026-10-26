"""H138 immutable upstream real-app benchmark safeguards."""
import unittest
from pathlib import Path
from h138.probe import APPS
from h138.source import SCHEMA,TITLES,task_instruction,seed_columns

class UpstreamFullAppSourceTests(unittest.TestCase):
    def test_pinned_two_distinct_real_MIT_apps(self):
        self.assertEqual(len(APPS),2)
        self.assertEqual(len(set(APPS.values())),2)
        self.assertTrue(all(len(x)==40 for x in APPS.values()))
        self.assertIn("untouched-two-complete-upstream",SCHEMA)

    def test_bilingual_instructions_and_no_oracle_pixels(self):
        for app in APPS:
            for i,title in enumerate(TITLES):
                lang="ar" if i%2 else "en"
                prompt=task_instruction(app,title,lang)
                self.assertIn(title,prompt)
                self.assertNotIn("point_2d",prompt)
                self.assertNotIn("960",prompt)
                state=seed_columns(app,i)
                self.assertTrue(state)
                self.assertNotIn("target_bbox",repr(state))

    def test_live_full_app_source_never_vendored_to_hala_repo(self):
        root=Path(__file__).resolve().parents[1]
        self.assertFalse((root/"h138"/"vendor-apps").exists())

if __name__=="__main__":unittest.main()
