"""H245 synthetic only MIT donor, source novelty, and release-denial tests."""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h245.independent_html5sortable_native as qa


def git(root,*args):
    return subprocess.run(["git","-C",str(root),*args],
        check=True,text=True,capture_output=True,timeout=12).stdout.strip()


class H245IndependentOriginalMITSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="hala-h245-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"source"
        self.source.mkdir()
        files={
            "LICENSE.md":b"The MIT License (MIT) SYNTHETIC TEST\n",
            "dist/html5sortable.min.js":b"/* fake test only, NEVER native QA */",
            "package.json":b'{"license":"MIT","version":"synthetic"}',
        }
        for name,blob in files.items():
            p=self.source.joinpath(*name.split("/"))
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(blob)
        git(self.source,"init","-q")
        git(self.source,"remote","add","origin",
            "https://github.com/lukasoppermann/html5sortable.git")
        git(self.source,"add",".")
        git(self.source,"-c","user.name=H245 Synthetic",
            "-c","user.email=h245@example.invalid",
            "commit","-q","-m","test fixture")
        self.sha=git(self.source,"rev-parse","HEAD")
        self.assets={name:sha256(blob).hexdigest() for name,blob in files.items()}
        self.output=self.root/"html5sortable-research-new"
        self.patches=[
            patch.object(qa,"COMMIT",self.sha),
            patch.object(qa,"FILES",self.assets),
            patch.object(qa,"known_public_source_family_floor",
                         return_value=frozenset()),
            patch.object(qa,"inventory_local_original_sources",return_value={
                "_owner_only_repository_shas":frozenset(),
                "historical_source_completeness_proven":False,
            }),
            patch.object(qa,"find_pre_h173_phases",return_value=[self.root]),
            patch.object(qa,"scan_legacy_manifest_sha_fields",return_value={
                "complete_historical_manifest_coverage_proven":False,
                "training_dataset_admitted":False,"old_image_media_opened":False,
                "_owner_only_png_sha_candidates":frozenset({"1"*64}),
                "_owner_only_rgba_sha_candidates":frozenset(),
            }),
            patch.object(qa,"current_20_metadata",return_value=(
                {"2"*64},{"3"*64})),
            patch.object(qa,"historical_h236_research",return_value=(
                {"4"*64},{"5"*64})),
            patch.object(qa,"private_json",return_value={
                "cases":[{
                    "preaction_png_sha256":f"{i+10:064x}",
                    "postaction_png_sha256":f"{i+20:064x}",
                    "preaction_rgba_sha256":f"{i+30:064x}",
                    "postaction_rgba_sha256":f"{i+40:064x}",
                } for i in range(8)]}),
            patch.object(qa,"verify_h242_receipt",return_value={
                "training_approved":False,
            }),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def preflight(self):
        return qa.preflight(self.source,self.output,self.root,
                             self.root,self.root,self.root)

    def test_original_source_identity_and_mit_notice_immutable(self):
        self.assertEqual(qa.REPO,"lukasoppermann/html5sortable")
        self.assertEqual(len(qa.COMMIT),40)
        self.assertEqual(len(qa.FILES),3)
        self.assertIn("LICENSE.md",qa.FILES)
        self.assertEqual(len(qa.FILES["LICENSE.md"]),64)

    def test_new_source_preflight_does_not_admit_data(self):
        result=self.preflight()
        self.assertEqual(result["older_exclusions"],1)
        self.assertEqual(result["current_research"],1)
        self.assertEqual(result["prior_native"],17)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(result["old_png"]),19)
        self.assertEqual(len(result["old_rgba"]),18)

    def test_changed_mit_or_original_asset_hash_fails_closed(self):
        for name in ("LICENSE.md","dist/html5sortable.min.js","package.json"):
            with self.subTest(name=name),patch.object(
                    qa,"FILES",{**self.assets,name:"0"*64}):
                with self.assertRaisesRegex(ValueError,"bytes mismatch"):
                    self.preflight()

    def test_public_or_owner_historical_source_reuse_denied(self):
        with patch.object(qa,"known_public_source_family_floor",
                          return_value=frozenset({qa.REPO})):
            with self.assertRaisesRegex(ValueError,"historical source"):
                self.preflight()
        with patch.object(qa,"inventory_local_original_sources",return_value={
            "_owner_only_repository_shas":frozenset({(qa.REPO,"f"*40)}),
        }):
            with self.assertRaisesRegex(ValueError,"private historical"):
                self.preflight()

    def test_dirty_untrusted_or_wrong_git_sha_fails_closed(self):
        with patch.object(qa,"COMMIT","f"*40):
            with self.assertRaisesRegex(ValueError,"commit mismatch"):
                self.preflight()
        (self.source/"untracked").write_text("untracked",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"dirty"):
            self.preflight()

    def test_never_write_inside_git_or_existing_dir(self):
        with self.assertRaisesRegex(ValueError,"outside every Git tree"):
            qa.preflight(self.source,self.source/"html5sortable-research-unsafe",
                         self.root,self.root,self.root,self.root)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError,"outside every Git tree"):
            self.preflight()

    def test_eight_original_source_scenes_are_distinct_and_bilingual(self):
        self.assertEqual(qa.COUNT,8)
        self.assertEqual(len(qa.CASES),8)
        self.assertEqual(len({x["id"] for x in qa.CASES}),8)
        self.assertEqual({x["number"] for x in qa.CASES},set(range(1,9)))
        self.assertEqual(sum(x["lang"]=="en" for x in qa.CASES),4)
        self.assertEqual(sum(x["lang"]=="ar" for x in qa.CASES),4)
        for row in qa.CASES:
            html=qa.self_authored_html(row)
            self.assertIn("class='sortable'",html)
            self.assertIn("id='source'",html)
            self.assertIn("CARD "+str(row["number"]),html)
            self.assertNotIn("http:",html)
            self.assertNotIn("https:",html)
            self.assertNotIn("<script src=",html)

    def test_prior_quarantined_scene_duplicate_rejected(self):
        previous={
            "cases":[{
                "preaction_png_sha256":"a"*64,
                "postaction_png_sha256":"a"*64,
                "preaction_rgba_sha256":"b"*64,
                "postaction_rgba_sha256":"b"*64,
            } for _ in range(8)]
        }
        with patch.object(qa,"private_json",return_value=previous):
            with self.assertRaisesRegex(ValueError,"distinct native frames"):
                self.preflight()


if __name__=="__main__":
    unittest.main()
