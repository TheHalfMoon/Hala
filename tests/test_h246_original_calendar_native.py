"""H246 exact original-MIT calendar source admission-denial, synthetic-only QA."""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h246.original_calendar_native as calendar


def git(root,*args):
    return subprocess.run(["git","-C",str(root),*args],text=True,
        capture_output=True,check=True,timeout=12).stdout.strip()


def fake_h242():
    return {"cases":[{
        "preaction_png_sha256":f"{i+20:064x}",
        "postaction_png_sha256":f"{i+40:064x}",
        "preaction_rgba_sha256":f"{i+100:064x}",
        "postaction_rgba_sha256":f"{i+120:064x}",
    } for i in range(8)]}


def fake_h245():
    return {
        "status":"H245_INDEPENDENT_NATIVE_SOURCE_PRIVATE_RESEARCH_NOT_ADMITTED",
        "source_repository":calendar.H245_REPO,"source_git_sha":calendar.H245_COMMIT,
        "original_source_sha256":calendar.H245_FILES,
        "unmodified_original_mit_js":True,"self_authored_ui_fixture":True,
        "historical_source_and_visual_completeness":False,
        "third_party_and_transitive_media_rights_audited":False,
        "train_approved":False,"dev_approved":False,"ood_approved":False,
        "model_calls":0,"optimizer_steps":0,
        "cases":[{
            "case_id":f"html5sortable-research-{i+1:02}",
            "language":"en" if i%2==0 else "ar",
            "real_native_reorder_verified":True,"network_attempts":0,
            "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
            "preaction_png_sha256":f"{i+200:064x}",
            "postaction_png_sha256":f"{i+230:064x}",
            "preaction_rgba_sha256":f"{i+300:064x}",
            "postaction_rgba_sha256":f"{i+330:064x}",
        } for i in range(8)],
    }


class H246OriginalCalendarTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="hala-h246-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"source"
        self.source.mkdir()
        samples={
            "LICENSE":b"MIT License SYNTHETIC ORIGINAL TEST ONLY",
            "dist/mc-calendar.min.js":b"/* fabricated JS - never native */",
            "dist/mc-calendar.min.css":b"/* fabricated CSS - never native */",
            "package.json":b'{"license":"MIT"}',
        }
        for name,blob in samples.items():
            p=self.source.joinpath(*name.split("/"))
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(blob)
        git(self.source,"init","-q")
        git(self.source,"remote","add","origin",
            "https://github.com/mikecoj/MCDatepicker.git")
        git(self.source,"add",".")
        git(self.source,"-c","user.name=H246 synthetic",
            "-c","user.email=h246@example.invalid",
            "commit","-q","-m","invented")
        self.sha=git(self.source,"rev-parse","HEAD")
        self.assets={name:sha256(blob).hexdigest() for name,blob in samples.items()}
        self.output=self.root/"mcdatepicker-private-research-test"
        mocks=[
            patch.object(calendar,"COMMIT",self.sha),
            patch.object(calendar,"FILES",self.assets),
            patch.object(calendar,"known_public_source_family_floor",
                         return_value=frozenset()),
            patch.object(calendar,"inventory_local_original_sources",return_value={
                "_owner_only_repository_shas":frozenset(),
            }),
            patch.object(calendar,"find_pre_h173_phases",return_value=[self.root]),
            patch.object(calendar,"scan_legacy_manifest_sha_fields",return_value={
                "complete_historical_manifest_coverage_proven":False,
                "training_dataset_admitted":False,
                "old_image_media_opened":False,
                "_owner_only_png_sha_candidates":frozenset({"1"*64}),
                "_owner_only_rgba_sha_candidates":frozenset(),
            }),
            patch.object(calendar,"current_20_metadata",return_value=(
                {"2"*64},{"3"*64})),
            patch.object(calendar,"historical_h236_research",return_value=(
                {"4"*64},{"5"*64})),
            patch.object(calendar,"private_json",side_effect=lambda p:
                fake_h245() if p.name=="PRIVATE_RESEARCH_NOT_ADMITTED.json"
                else fake_h242()),
            patch.object(calendar,"verify_h242_receipt",return_value={
                "training_approved":False,
            }),
        ]
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def check(self):
        return calendar.preflight(
            self.source,self.output,self.root,self.root,
            self.root,self.root,self.root)

    def test_eight_calendar_tasks_bilingual_and_not_drag(self):
        self.assertEqual(calendar.COUNT,8)
        self.assertEqual([x["day"] for x in calendar.CASES],list(range(8,16)))
        self.assertEqual(sum(x["lang"]=="ar" for x in calendar.CASES),4)
        self.assertEqual(sum(x["lang"]=="en" for x in calendar.CASES),4)
        self.assertEqual(len({x["id"] for x in calendar.CASES}),8)
        for item in calendar.CASES:
            html=calendar.self_authored_html(item)
            self.assertIn("id='datepicker'",html)
            self.assertIn(item["instruction"],html)
            self.assertNotIn("https:",html)
            self.assertNotIn("<script src",html)

    def test_pinned_license_and_original_files_required(self):
        result=self.check()
        self.assertEqual(result["old_png_count"],1)
        self.assertEqual(result["other_known_research_count"],1)
        self.assertEqual(result["prior_native_frames"],33)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(result["old_png"]),35)
        self.assertEqual(len(result["old_rgba"]),34)

    def test_any_original_file_modified_denied(self):
        for name in self.assets:
            with self.subTest(asset=name),patch.object(
                calendar,"FILES",{**self.assets,name:"0"*64}):
                with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                    self.check()

    def test_already_historical_original_source_denied(self):
        with patch.object(calendar,"known_public_source_family_floor",
                          return_value=frozenset({calendar.REPO})):
            with self.assertRaisesRegex(ValueError,"historical original"):
                self.check()
        with patch.object(calendar,"inventory_local_original_sources",return_value={
            "_owner_only_repository_shas":frozenset({(calendar.REPO,"f"*40)}),
        }):
            with self.assertRaisesRegex(ValueError,"consumed owner-only"):
                self.check()

    def test_wrong_sha_and_dirty_original_denied(self):
        with patch.object(calendar,"COMMIT","f"*40):
            with self.assertRaisesRegex(ValueError,"SHA or clean worktree"):
                self.check()
        (self.source/"staged-secret.tmp").write_text("fake",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"SHA or clean worktree"):
            self.check()

    def test_private_research_output_must_not_exist_or_enter_git(self):
        with self.assertRaisesRegex(ValueError,"outside all Git"):
            calendar.preflight(self.source,self.source/"mcdatepicker-private-research-unsafe",
                               self.root,self.root,self.root,self.root,self.root)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError,"outside all Git"):
            self.check()

    def test_h245_prior_source_bad_role_and_forged_sha_denied(self):
        for variant in ("train","repo","dupe","locale"):
            p=fake_h245()
            if variant=="train":p["cases"][1]["train_eligible"]=True
            if variant=="repo":p["source_repository"]="fabricated/stolen"
            if variant=="dupe":
                p["cases"][7]["postaction_png_sha256"]=p["cases"][0]["preaction_png_sha256"]
            if variant=="locale":p["cases"][0]["language"]="ar"
            with self.subTest(case=variant),patch.object(
                calendar,"private_json",return_value=p):
                with self.assertRaises(ValueError):
                    calendar._h245_exclusions(self.root)

    def test_prior_h245_metadata_returns_16_distinct_frames(self):
        p,v=calendar._h245_exclusions(self.root)
        self.assertEqual(len(p),16)
        self.assertEqual(len(v),16)

    def test_prior_two_native_source_duplicate_fails_closed(self):
        forged=fake_h245()
        forged["cases"][0]["preaction_png_sha256"]=f"{20:064x}"
        with patch.object(calendar,"private_json",side_effect=lambda p:
                forged if p.name=="PRIVATE_RESEARCH_NOT_ADMITTED.json"
                else fake_h242()):
            with self.assertRaisesRegex(ValueError,"visually overlap"):
                self.check()


if __name__=="__main__":
    unittest.main()
