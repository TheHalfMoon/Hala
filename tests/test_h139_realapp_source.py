"""H139 independent real app source and leak-proof split validation."""
import unittest
from h139.registry import UPSTREAM,SCHEMA,check_split_manifest
from h139.source import COUNTS,app_seed,instructions

class H139CrossAppGuards(unittest.TestCase):
    def test_immutable_MIT_apps_partition(self):
        self.assertEqual(set(UPSTREAM),{"js-board","alt-static-board","ood-board"})
        self.assertEqual([n for n,p in UPSTREAM.items() if p["role"]=="blind_test"],
                         ["ood-board"])
        self.assertEqual(COUNTS,{"train":32,"dev":8})
        self.assertEqual(len({p["git_sha"] for p in UPSTREAM.values()}),3)

    def test_en_ar_real_application_titles_and_seed_ids(self):
        for app in ("js-board","alt-static-board"):
            key,initial,id_,title=app_seed(app,1044)
            self.assertEqual(len(initial),3)
            self.assertEqual(initial[0]["id"],id_)
            self.assertIn(title,instructions(title,"ar"))
            self.assertIn(title,instructions(title,"en"))
            self.assertNotIn("point_2d",instructions(title,"en"))
            self.assertNotIn("bbox",instructions(title,"en"))

    def test_explicit_blind_leakage_and_wrong_repo_rejected(self):
        row={"app":"js-board","app_git_sha":UPSTREAM["js-board"]["git_sha"],
             "image_sha256":"a"*64,"role":"train"}
        m={"schema":SCHEMA,"private":True,"cases":[row]}
        self.assertEqual(check_split_manifest(m),[row])
        with self.assertRaises(ValueError):
            check_split_manifest(m,forbidden_sha={"a"*64})
        bad=dict(row,app="ood-board",app_git_sha=UPSTREAM["ood-board"]["git_sha"])
        with self.assertRaises(ValueError):
            check_split_manifest(dict(m,cases=[bad]))
        with self.assertRaises(ValueError):
            check_split_manifest(dict(m,cases=[row,row]))
        with self.assertRaises(ValueError):
            check_split_manifest(dict(m,cases=[dict(row,app_git_sha="fake")]))

if __name__=="__main__":unittest.main()
