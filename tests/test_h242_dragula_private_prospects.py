"""H242 synthetic-only source provenance and offline native-scene contract.

No real images, no historical OOD, no model forward, no owner-private data.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h242.private_dragula_prospects import (
    CASES, CASE_COUNT, REPO, _strict_local_source, fixture,
)


def mocked_source(root: Path, out: Path, repos=frozenset()):
    return (
        patch("h242.private_dragula_prospects.preflight",
              return_value={"original_source": root,"asset":root/"public.js","output":out}),
        patch("h242.private_dragula_prospects.inventory_local_original_sources",
              return_value={"source_checkouts_inspected":7,
                            "_owner_only_repository_shas":
                            frozenset((repo,"a"*40) for repo in repos)}),
        patch("h242.private_dragula_prospects.find_pre_h173_phases",
              return_value=[root]),
        patch("h242.private_dragula_prospects.scan_legacy_manifest_sha_fields",
              return_value={
                  "complete_historical_manifest_coverage_proven":False,
                  "training_dataset_admitted":False,"old_image_media_opened":False,
                  "manifest_files_scanned":7,"distinct_png_sha_candidates":130,
                  "distinct_decoded_rgba_sha_candidates":0,
                  "_owner_only_png_sha_candidates":frozenset({"1"*64}),
                  "_owner_only_rgba_sha_candidates":frozenset(),
              })
    )


class H242OriginalSourcePrivateResearchTests(unittest.TestCase):
    def test_eight_unique_author_created_contexts_and_balanced_languages(self):
        self.assertEqual(CASE_COUNT, 8)
        self.assertEqual(len(CASES), 8)
        self.assertEqual(len({row["case_id"] for row in CASES}), 8)
        self.assertEqual(sum(row["lang"] == "en" for row in CASES), 4)
        self.assertEqual(sum(row["lang"] == "ar" for row in CASES), 4)
        self.assertEqual({x["card"] for x in CASES}, set(range(1,9)))

    def test_html_contains_only_an_authored_offline_screenshot_canvas(self):
        for row in CASES:
            with self.subTest(case=row["case_id"]):
                html=fixture(row)
                self.assertIn("H242 PRIVATE SOURCE RESEARCH - NOT TRAIN",html)
                self.assertIn("id='moving'",html)
                self.assertIn("id='right'",html)
                self.assertIn("CARD "+str(row["card"]),html)
                self.assertNotIn("http:",html)
                self.assertNotIn("https:",html)
                self.assertNotIn("<script src=",html)
                self.assertNotIn("OWNER_PRIVATE_REFERENCE",html)

    def test_no_ood_or_training_image_published_in_source_constants(self):
        self.assertEqual(REPO,"bevacqua/dragula")
        for row in CASES:
            self.assertEqual(set(row),{
                "case_id","lang","accent","target","card","task"})
            self.assertNotIn("sha256",str(row))
            self.assertNotIn("gold",str(row).lower())

    def test_runtime_provenance_preflight_keeps_old_history_unapproved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            output=root/"private-output"
            p=mocked_source(root,output)
            with p[0],p[1],p[2],p[3]:
                result=_strict_local_source(root,output,root)
                self.assertEqual(result["old"]["manifest_files_scanned"],7)
                self.assertFalse(result["old"]["training_dataset_admitted"])
                self.assertFalse(output.exists())

    def test_preflight_rejects_used_owner_original_family_even_if_public_new(self):
        for repo in ("bevacqua/dragula","BEVACQUA/DRAGULA"):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                mocks=mocked_source(root,root/"capture",frozenset({repo.lower()}))
                with mocks[0],mocks[1],mocks[2],mocks[3]:
                    with self.subTest(repo=repo),self.assertRaisesRegex(
                            ValueError,"known locally consumed"):
                        _strict_local_source(root,root/"capture",root)

    def test_incomplete_or_false_historical_metadata_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            patched=mocked_source(root,root/"capture")
            for edited in ({"training_dataset_admitted":True},
                           {"old_image_media_opened":True},
                           {"complete_historical_manifest_coverage_proven":True}):
                value=patched[3].new_kwargs if hasattr(patched[3],"new_kwargs") else {}
                original={
                    "complete_historical_manifest_coverage_proven":False,
                    "training_dataset_admitted":False,
                    "old_image_media_opened":False,
                }
                with patched[0], patched[1],patched[2],patch(
                    "h242.private_dragula_prospects.scan_legacy_manifest_sha_fields",
                    return_value={**original,**edited}):
                    with self.subTest(edited=edited),self.assertRaises(ValueError):
                        _strict_local_source(root,root/"capture",root)

    def test_no_data_capture_in_tests_without_browser_or_original_source(self):
        self.assertTrue(all("http" not in row["task"] for row in CASES))


if __name__=="__main__":
    unittest.main()
