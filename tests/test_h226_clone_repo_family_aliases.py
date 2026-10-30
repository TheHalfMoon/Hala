"""H226: reproduce and close Git clone .git source-family alias bypasses.

Synthetic metadata ONLY. No model, gold, native browser, private media,
history pixels or consumed H173 OOD accessed. No TRAIN/DEV admission.
"""
from __future__ import annotations

from copy import deepcopy
import unittest

from h174.split_firewall import (
    SCHEMA, HISTORY_SCHEMA, BANNED, BANNED_GIT, BANNED_HTML,
    canonical_repository_family, validate,
)
from h174.source_history_gate import (
    SCHEMA as SOURCE_HISTORY_SCHEMA, KNOWN_H173_REPOSITORIES,
    validate_with_source_history,
)


def fixture():
    metadata = {
        "schema": SCHEMA,
        "sources": [
            {"app":"freshclick","split":"TRAIN","repo":"fresh/demo",
             "git_sha":"a"*40,"html_sha256":"a"*64,"license":"MIT"},
            {"app":"freshdrag","split":"DEV","repo":"fresh/drag",
             "git_sha":"b"*40,"html_sha256":"b"*64,"license":"MIT"},
            {"app":"freshood","split":"OOD_RESERVED","repo":"fresh/holdout",
             "git_sha":"c"*40,"html_sha256":"c"*64,"license":"MIT"},
        ],
        "cases": [
            {"index":1,"app":"freshclick","split":"TRAIN",
             "image_sha256":"1"*64,"visual_rgba_sha256":"2"*64,
             "language":"en","action_type":"click",
             "instruction":"Click the blue button"},
            {"index":2,"app":"freshdrag","split":"DEV",
             "image_sha256":"3"*64,"visual_rgba_sha256":"4"*64,
             "language":"ar","action_type":"drag",
             "instruction":"اسحب العنصر إلى الهدف"},
        ],
    }
    image_shas={"schema":HISTORY_SCHEMA,"image_shas":["8"*64],
                "visual_shas":["9"*64]}
    historical={
        "schema": SOURCE_HISTORY_SCHEMA,
        "historical_app_ids": sorted(BANNED),
        "historical_repositories": sorted(
            KNOWN_H173_REPOSITORIES | {"history/old-donor"}),
        "source_git_shas": sorted(BANNED_GIT),
        "source_html_shas": sorted(BANNED_HTML),
    }
    return metadata,image_shas,historical


class GithubCloneSuffixAliasRegression(unittest.TestCase):
    def test_positive_unique_original_families_remain_metadata_only(self):
        meta,pixels,hist=fixture()
        a=validate(meta,pixels)
        b=validate_with_source_history(meta,pixels,hist)
        self.assertEqual(a["train_cases"],1)
        self.assertEqual(a["dev_cases"],1)
        self.assertEqual(a["reserved_ood_app_count"],1)
        self.assertFalse(b["training_approved"])
        self.assertFalse(b["source_history_completeness_proven"])

    def test_same_real_repository_across_train_dev_or_reserved_denied(self):
        for i in (1,2):
            for alias in ("FRESH/DEMO.git","FRESH/DEMO.GIT",
                          "fresh/demo.git.git", "FRESH/DEMO"):
                with self.subTest(split=i,alias=alias):
                    meta,pixels,_=fixture()
                    meta["sources"][i]["repo"]=alias
                    with self.assertRaisesRegex(
                        ValueError,"duplicate upstream repository"
                    ):
                        validate(meta,pixels)

    def test_same_real_repo_with_clone_suffix_in_train_denied(self):
        for source_idx in (1,2):
            meta,pixels,_=fixture()
            meta["sources"][0]["repo"]="FRESH/DRAG.git" if source_idx==1 else "FRESH/HOLDOUT.git"
            with self.subTest(idx=source_idx), self.assertRaisesRegex(
                ValueError,"duplicate upstream repository"
            ):
                validate(meta,pixels)

    def test_different_repository_families_still_accepted(self):
        meta,pixels,_=fixture()
        meta["sources"][0]["repo"]="FRESH/DEMO.git"
        meta["sources"][1]["repo"]="FRESH/DRAG.GIT"
        meta["sources"][2]["repo"]="fresh/holdout.git.git"
        result=validate(meta,pixels)
        self.assertEqual(result["train_cases"],1)

    def test_old_repo_under_clone_alias_rejected_for_any_new_split(self):
        for i in (0,1,2):
            for alias in ("HISTORY/OLD-DONOR.git",
                          "history/old-donor.GIT",
                          "history/old-donor.git.git",
                          "history/old-donor"):
                with self.subTest(split=i,repo=alias):
                    meta,pixels,hist=fixture()
                    meta["sources"][i]["repo"]=alias
                    with self.assertRaisesRegex(
                        ValueError,"historical upstream repository family reused"
                    ):
                        validate_with_source_history(meta,pixels,hist)

    def test_known_consumed_h173_repo_alias_denied(self):
        for frozen in KNOWN_H173_REPOSITORIES:
            for alias in (frozen+".git", frozen.upper()+".GIT"):
                with self.subTest(alias=alias):
                    meta,pixels,hist=fixture()
                    meta["sources"][0]["repo"]=alias
                    # H238 rejects consumed H173 sources at the base H174
                    # preflight, before the historical SHA-only crosscheck.
                    with self.assertRaisesRegex(
                        ValueError,"historical upstream repository family reused|public historical original source family reused"
                    ):
                        validate_with_source_history(meta,pixels,hist)

    def test_duplicate_historical_clone_alias_rejected(self):
        for alias in ("HISTORY/OLD-DONOR.GIT",
                      "history/old-donor.git.git"):
            with self.subTest(alias=alias):
                meta,pixels,hist=fixture()
                hist["historical_repositories"].append(alias)
                with self.assertRaisesRegex(
                    ValueError,"case-variant aliases"
                ):
                    validate_with_source_history(meta,pixels,hist)

    def test_bad_clone_suffix_only_empty_repository_rejected(self):
        for invalid in ("valid/.git","valid/.GIT.GIT","valid/..git"):
            with self.subTest(repo=invalid):
                with self.assertRaises(ValueError):
                    canonical_repository_family(invalid)

    def test_repo_identity_is_casefolded_and_git_suffix_normalized(self):
        keys=("Owner/Repo","OWNER/REPO.git","owner/repo.GIT",
              "owner/repo.git.git")
        self.assertEqual({canonical_repository_family(r) for r in keys},
                         {"owner/repo"})

    def test_invalid_metadata_rejected_as_value_error(self):
        for invalid in (None, [], "owner/repo/extra",
                        "../outside", "owner\\repo", 10):
            with self.subTest(repo=repr(invalid)):
                with self.assertRaises(ValueError):
                    canonical_repository_family(invalid)


if __name__ == "__main__":
    unittest.main()
