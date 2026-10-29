"""H176 source-family attribution: strictly synthetic metadata-only tests.

All hashes and pseudo-owner repositories in these tests are fabricated except
the already-published H173 immutable source pins. Never load consumed media.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from h173.source_preflight import SOURCE_PINS
from h174.source_history_gate import SCHEMA as HISTORY_SCHEMA
from h174.split_firewall import (
    BANNED, HISTORY_SCHEMA as PIXEL_HISTORY_SCHEMA, SCHEMA as SPLIT_SCHEMA,
)
from h176.source_family_attribution import (
    SCHEMA, KNOWN_EXTRA_PREVIOUS, KNOWN_NPM_SOURCES, FROZEN_OTHER_GIT,
    FROZEN_H173_REPOS,
    audit_attribution,
)
from h176.audit_owner_history import run as local_cli_run


def _fixture():
    metadata = {
        "schema": SPLIT_SCHEMA,
        "sources": [
            {"app": "freshclick", "split": "TRAIN", "repo": "newowner/click",
             "git_sha": "a"*40, "html_sha256": "a"*64, "license": "MIT"},
            {"app": "freshdrag", "split": "DEV", "repo": "newowner/drag",
             "git_sha": "b"*40, "html_sha256": "b"*64, "license": "MIT"},
            {"app": "freshood", "split": "OOD_RESERVED", "repo": "newowner/ood",
             "git_sha": "c"*40, "html_sha256": "c"*64, "license": "MIT"},
        ],
        "cases": [
            {"index": 11, "app": "freshclick", "split": "TRAIN",
             "image_sha256": "1"*64, "visual_rgba_sha256": "2"*64,
             "language": "en", "action_type": "click",
             "instruction": "Click the original button"},
            {"index": 12, "app": "freshdrag", "split": "DEV",
             "image_sha256": "3"*64, "visual_rgba_sha256": "4"*64,
             "language": "ar", "action_type": "drag",
             "instruction": "اسحب العنصر إلى الهدف"},
        ],
    }
    pixels = {
        "schema": PIXEL_HISTORY_SCHEMA, "image_shas": ["8"*64],
        "visual_shas": ["9"*64],
    }
    rows = []
    for index, app in enumerate(sorted(BANNED | KNOWN_EXTRA_PREVIOUS), start=1):
        if app == "splitjs":
            # Historical H137 npm vendor fixture, not a fabricated Git SHA.
            rows.append({
                "app": app, "package": "split.js", "version": "1.6.5",
                "artifact_sha256": f"{index:064x}",
            })
            continue
        frozen = SOURCE_PINS.get(app)
        rows.append({
            "app": app, "repo": (FROZEN_H173_REPOS[app] if frozen else
                                  f"fictional-owner/old-original-{index}"),
            "git_sha": (frozen["git_sha"] if frozen else
                        FROZEN_OTHER_GIT.get(app, f"{index:040x}")),
            "html_sha256": (frozen["original_html_sha256"] if frozen
                            else f"{index:064x}"),
        })
    history = {
        "schema": HISTORY_SCHEMA,
        "historical_app_ids": [r["app"] for r in rows],
        "historical_repositories": [r["repo"] for r in rows if "repo" in r],
        "source_git_shas": [r["git_sha"] for r in rows if "git_sha" in r],
        "source_html_shas": [r["html_sha256"] for r in rows if "html_sha256" in r],
    }
    attribution = {"schema": SCHEMA, "families": rows}
    return metadata, pixels, history, attribution


class SourceFamilyAttributionTests(unittest.TestCase):
    def test_valid_paired_metadata_never_approves_training_or_ood(self):
        result = audit_attribution(*_fixture())
        self.assertEqual(result["paired_source_families"], len(BANNED | KNOWN_EXTRA_PREVIOUS))
        self.assertEqual(result["earlier_source_git_pins_matched"], len(FROZEN_OTHER_GIT))
        self.assertEqual(result["known_npm_packages_paired"], 1)
        self.assertEqual(result["frozen_h173_pairs_matched"], len(SOURCE_PINS))
        self.assertFalse(result["source_history_completeness_proven"])
        self.assertFalse(result["provenance_authenticity_proven"])
        self.assertFalse(result["model_training_approved"])
        self.assertFalse(result["ood_evaluation_approved"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["native_rewards"], 0)

    def test_pair_order_and_repo_case_do_not_change_fingerprint(self):
        args = _fixture()
        baseline = audit_attribution(*args)
        updated = deepcopy(args)
        updated[3]["families"].reverse()
        updated[3]["families"][0]["repo"] = updated[3]["families"][0]["repo"].upper()
        result = audit_attribution(*updated)
        self.assertEqual(result["attribution_sha256"], baseline["attribution_sha256"])

    def test_legacy_placeholder_repository_cannot_fake_family_coverage(self):
        meta, pixels, history, attribution = _fixture()
        history["historical_repositories"] = ["fictional-owner/placeholder"]
        with self.assertRaisesRegex(ValueError, "repository attribution"):
            audit_attribution(meta, pixels, history, attribution)

    def test_missing_known_source_family_refused(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"].pop()
        with self.assertRaisesRegex(ValueError, "missing known|disagree"):
            audit_attribution(meta, pixels, history, attribution)

    def test_known_h173_pair_must_not_be_substituted(self):
        for app in SOURCE_PINS:
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            row["git_sha"] = "e"*40
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "H173 frozen original app"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_h173_original_repo_cannot_be_spoofed_even_with_matching_sha_pair(self):
        # The legacy unpaired repository roster can be rewritten alongside a
        # fake row. Exact Git and HTML hashes alone do not establish family.
        for app in SOURCE_PINS:
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            old_repo = row["repo"]
            row["repo"] = f"fictional-owner/renamed-{app}"
            history["historical_repositories"].remove(old_repo)
            history["historical_repositories"].append(row["repo"])
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "H173 frozen original upstream repository"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_h173_original_repo_case_variant_remains_same_family(self):
        meta, pixels, history, attribution = _fixture()
        row = next(r for r in attribution["families"] if r["app"] == "tingle")
        row["repo"] = row["repo"].upper()
        report = audit_attribution(meta, pixels, history, attribution)
        self.assertFalse(report["source_history_completeness_proven"])

    def test_newly_recovered_prior_source_cannot_disappear_from_both_manifests(self):
        meta, pixels, history, attribution = _fixture()
        # The old independent H174 arrays alone would accept removal of
        # this H138 eval app because it is not in the old H173 minimum.
        app = "react-html5-board"
        row = next(r for r in attribution["families"] if r["app"] == app)
        attribution["families"].remove(row)
        history["historical_app_ids"].remove(row["app"])
        history["historical_repositories"].remove(row["repo"])
        history["source_git_shas"].remove(row["git_sha"])
        history["source_html_shas"].remove(row["html_sha256"])
        with self.assertRaisesRegex(ValueError, "missing known historical"):
            audit_attribution(meta, pixels, history, attribution)

    def test_older_source_app_git_revision_cannot_be_relabelled(self):
        for app in FROZEN_OTHER_GIT:
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            row["git_sha"] = "e"*40
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "earlier source pilot/eval frozen Git"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_h137_npm_package_has_no_fabricated_git_or_html(self):
        meta, pixels, history, attribution = _fixture()
        row = next(r for r in attribution["families"] if r["app"] == "splitjs")
        self.assertEqual(set(row), {"app", "package", "version", "artifact_sha256"})
        self.assertFalse(any("split.js" in r for r in history["historical_repositories"]))
        report = audit_attribution(meta, pixels, history, attribution)
        self.assertEqual(report["known_npm_packages_paired"], 1)

    def test_h137_npm_version_and_binary_digest_are_mandatory(self):
        for key, invalid in (
            ("package", "different-name"), ("version", "9.0.0"),
            ("artifact_sha256", None), ("artifact_sha256", "0" * 40),
        ):
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == "splitjs")
            row[key] = invalid
            with self.subTest(field=key), self.assertRaisesRegex(
                ValueError, "npm package or version|artifact SHA-256"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_h137_consumed_npm_upstream_rejects_new_git_alias_in_any_split(self):
        # The H174 legacy repo arrays lack the npm-only family. H176
        # catches a new Git checkout of the exact npm package's upstream.
        for index in range(3):
            meta, pixels, history, attribution = _fixture()
            meta["sources"][index]["repo"] = "NATHANCAHILL/SPLIT"
            with self.subTest(split=meta["sources"][index]["split"]), self.assertRaisesRegex(
                ValueError, "historical npm upstream Git repository"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_h137_npm_cannot_masquerade_as_unpinned_git_source(self):
        meta, pixels, history, attribution = _fixture()
        row = next(r for r in attribution["families"] if r["app"] == "splitjs")
        row.update({"repo": "invented/splitjs", "git_sha": "a"*40,
                    "html_sha256": "f"*64})
        with self.assertRaisesRegex(ValueError, "historical npm source requires"):
            audit_attribution(meta, pixels, history, attribution)

    def test_historical_sha_must_be_attributed_to_a_family(self):
        meta, pixels, history, attribution = _fixture()
        history["source_git_shas"].append("d"*40)
        with self.assertRaisesRegex(ValueError, "Git SHA attribution"):
            audit_attribution(meta, pixels, history, attribution)

    def test_duplicate_historical_app_cannot_hide_alternate_revision(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"].append(deepcopy(attribution["families"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate historical application"):
            audit_attribution(meta, pixels, history, attribution)

    def test_case_variant_repo_alias_same_family(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"][0]["repo"] = "FICTIONAL-OWNER/OLD-ORIGINAL-1"
        result = audit_attribution(meta, pixels, history, attribution)
        self.assertEqual(result["distinct_historical_repositories"],
                         len(BANNED | KNOWN_EXTRA_PREVIOUS) - len(KNOWN_NPM_SOURCES))

    def test_no_images_gold_or_private_fields_in_attribution(self):
        for payload in ({"gold": "fake"}, {"image_sha256": "f"*64},
                        {"coordinates": [1, 2]}):
            meta, pixels, history, attribution = _fixture()
            attribution["families"][0].update(payload)
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "only app/repo/Git/HTML"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_type_confusion_and_malformed_repos_fail(self):
        for bad in (None, [], ["owner", "repo"], "owner/../repo", "owner/."):
            meta, pixels, history, attribution = _fixture()
            attribution["families"][0]["repo"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit_attribution(meta, pixels, history, attribution)

    def test_non_metadata_root_fields_rejected(self):
        meta, pixels, history, attribution = _fixture()
        attribution["raw_images"] = ["/private/path"]
        with self.assertRaisesRegex(ValueError, "exact metadata-only"):
            audit_attribution(meta, pixels, history, attribution)

    def test_known_historical_app_reuse_still_blocked(self):
        meta, pixels, history, attribution = _fixture()
        meta["sources"][0]["repo"] = attribution["families"][0]["repo"]
        with self.assertRaisesRegex(ValueError, "historical upstream"):
            audit_attribution(meta, pixels, history, attribution)

    def test_invalid_legacy_history_still_blocked(self):
        meta, pixels, history, attribution = _fixture()
        history["source_html_shas"].pop()
        with self.assertRaisesRegex(ValueError, "original HTML SHA attribution"):
            audit_attribution(meta, pixels, history, attribution)


    def test_owner_cli_succeeds_without_printing_identity_values(self):
        files = ("candidate.json", "pixel_hashes.json", "source_history.json",
                 "paired_history.json")
        with tempfile.TemporaryDirectory() as root:
            for name, payload in zip(files, _fixture()):
                (Path(root) / name).write_text(json.dumps(payload), encoding="utf-8")
            args = [
                "--candidates", str(Path(root) / files[0]),
                "--prior-pixel-hashes", str(Path(root) / files[1]),
                "--prior-source-history", str(Path(root) / files[2]),
                "--paired-source-history", str(Path(root) / files[3]),
            ]
            output, errors = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                status = local_cli_run(args)
            self.assertEqual(status, 0)
            self.assertEqual(errors.getvalue(), "")
            self.assertIn("HISTORY_COMPLETENESS_UNPROVEN", output.getvalue())
            self.assertNotIn("fictional-owner", output.getvalue())
            self.assertNotIn("freshclick", output.getvalue())
            self.assertNotIn(str(Path(root)), output.getvalue())

    def test_owner_cli_rejects_duplicate_json_keys_without_leaking_data(self):
        files = ("candidate.json", "pixel_hashes.json", "source_history.json",
                 "paired_history.json")
        with tempfile.TemporaryDirectory() as root:
            for name, payload in zip(files, _fixture()):
                (Path(root) / name).write_text(json.dumps(payload), encoding="utf-8")
            (Path(root) / files[3]).write_text(
                '{"schema": "sensitive-donor", "schema": "duplicate"}',
                encoding="utf-8",
            )
            args = [
                "--candidates", str(Path(root) / files[0]),
                "--prior-pixel-hashes", str(Path(root) / files[1]),
                "--prior-source-history", str(Path(root) / files[2]),
                "--paired-source-history", str(Path(root) / files[3]),
            ]
            output, errors = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                status = local_cli_run(args)
            self.assertEqual(status, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("H176_FAIL_CLOSED_OWNER_METADATA_CHECK", errors.getvalue())
            self.assertNotIn("sensitive-donor", errors.getvalue())
            self.assertNotIn(str(Path(root)), errors.getvalue())

    def test_owner_cli_rejects_nonfinite_json_and_oversized_inputs(self):
        from h176.audit_owner_history import _read_metadata_json, MAX_MANIFEST_BYTES
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "untrusted.json"
            path.write_text('{"field": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_metadata_json(str(path))
            path.write_bytes(b"x" * (MAX_MANIFEST_BYTES + 1))
            with self.assertRaises(ValueError):
                _read_metadata_json(str(path))


if __name__ == "__main__":
    unittest.main()
