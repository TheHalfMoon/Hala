"""H241 metadata-only pre-H173 discovery regression with synthetic fixtures."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h204.training_readiness_gap import training_readiness_gap
from h241.expanded_pre173_metadata import (
    audit_expanded_owner_metadata, find_pre_h173_phases, main,
)
from test_h204_training_readiness_gap import fixtures


def create_environment(root: Path):
    owner, first = fixtures(root)
    extra = root / "Hala-H141-Private" / "metadata" / "manifest.json"
    extra.parent.mkdir(parents=True)
    extra.write_text(json.dumps({"image_sha256": "a"*64}),
                     encoding="utf-8")
    # The consumed H173 can never be opened for H241.
    forbidden = root / "Hala-H173-Private" / "metadata" / "manifest.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"CONSUMED_H173_FORBIDDEN")
    # Legacy-looking but H185-UNSUPPORTED names are not silently
    # represented as scanned phases; H241 reports their aggregate count.
    for legacy_alias in ("Hala-H142", "private-h142-Private"):
        unsupported = root / legacy_alias / "metadata" / "manifest.json"
        unsupported.parent.mkdir(parents=True)
        unsupported.write_bytes(b"UNSUPPORTED_OLD_NAME_NEVER_OPEN")
    # H241 must also never use new H174+ research for old exclusions.
    newer = root / "Hala-H188-Private" / "metadata" / "manifest.json"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"NEWER_RESEARCH_FORBIDDEN")
    return owner, first, extra, forbidden, newer


class H241MetadataOnlyTests(unittest.TestCase):
    def test_discovers_all_and_only_pre_h173_immediate_phase_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _owner, first, extra, _consumed, _newer = create_environment(root)
            roots = find_pre_h173_phases([root])
            self.assertEqual({x.resolve() for x in roots},
                             {first.resolve(), extra.parent.parent.resolve()})
            self.assertEqual(len(roots), 2)

    def test_expanded_history_compares_20_candidates_without_admitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner, _first, _extra, _consumed, _newer = create_environment(root)
            value = audit_expanded_owner_metadata(owner, [root])
            self.assertEqual(value["pre_h173_phases_scanned"], 2)
            self.assertEqual(value["potential_old_phase_name_variants_unscanned"], 2)
            self.assertEqual(value["historical_manifests_scanned"], 2)
            self.assertEqual(value["distinct_prior_png_sha_exclusions"], 2)
            self.assertEqual(value["distinct_prior_rgba_sha_exclusions"], 0)
            self.assertEqual(value["historical_manifests_missing_rgba_sha"], 2)
            self.assertEqual(value["new_unadmitted_cases_compared"], 20)
            self.assertFalse(value["historical_history_complete"])
            self.assertFalse(value["visual_history_complete"])
            self.assertFalse(value["training_approved"])
            self.assertFalse(value["dev_approved"])
            self.assertFalse(value["ood_approved"])
            self.assertFalse(value["h173_ood_accessed"])
            self.assertEqual(value["model_calls"], 0)
            self.assertEqual(value["optimizer_steps"], 0)

    def test_never_opens_old_media_gold_consumed_ood_or_newer_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner, _old, _extra, consumed, newer = create_environment(root)
            original = Path.read_bytes
            observed = []
            def safe_read(path):
                observed.append(path.name)
                if path.name not in ("manifest.json",
                                     "source-prospects-not-admitted.json"):
                    raise AssertionError("unapproved private file read")
                if (path in (consumed, newer) or path.parent.parent.name in
                        ("Hala-H142", "private-h142-Private")):
                    raise AssertionError("H173/modern manifest forbidden")
                return original(path)
            with patch.object(Path, "read_bytes", safe_read):
                result = audit_expanded_owner_metadata(owner, [root])
            self.assertEqual(result["historical_manifests_scanned"], 2)
            self.assertEqual(observed.count("manifest.json"), 2)
            self.assertEqual(observed.count("source-prospects-not-admitted.json"), 15)

    def test_old_png_collision_in_any_discovered_phase_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner, _old, extra, _consumed, _newer = create_environment(root)
            # The new prospective case 0 uses SHA f'{1:064x}' in H204's fixture.
            extra.write_text(json.dumps({"image_sha256": f"{1:064x}"}),
                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                audit_expanded_owner_metadata(owner, [root])

    def test_no_old_phase_or_only_consumed_h173_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Hala-H173-Private").mkdir()
            (root / "Hala-H188-Private").mkdir()
            with self.assertRaisesRegex(ValueError, "no eligible pre-H173"):
                find_pre_h173_phases([root])

    def test_duplicate_parent_and_invalid_owner_root_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner, *_ = create_environment(root)
            with self.assertRaisesRegex(ValueError, "duplicate parent"):
                find_pre_h173_phases([root, root])
            with self.assertRaises(ValueError):
                audit_expanded_owner_metadata(root / "missing", [root])
            with self.assertRaises(ValueError):
                find_pre_h173_phases([])
            self.assertEqual(
                training_readiness_gap(owner, [root/"private-h140"])[
                    "legacy_manifests_scanned"
                ], 1)

    def test_untrusted_pre173_symlink_fails_closed_if_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "plain-target"
            target.mkdir()
            try:
                (root / "private-h138").symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("OS does not allow non-admin directory symlinks")
            with self.assertRaisesRegex(ValueError, "reparse"):
                find_pre_h173_phases([root])

    def test_cli_emits_only_aggregate_metadata_not_paths_or_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner, _old, _extra, _consumed, _newer = create_environment(root)
            out, err = StringIO(), StringIO()
            args = ["h241", "--owner-root", str(owner),
                    "--historical-parent", str(root)]
            with patch("sys.argv", args), redirect_stdout(out), redirect_stderr(err):
                main()
            x = json.loads(out.getvalue())
            self.assertEqual(x["status"],
                             "H241_EXPANDED_PRE_H173_OWNER_SHA_EXCLUSIONS_ONLY_NO_ADMISSION")
            self.assertFalse(x["training_approved"])
            self.assertNotIn(str(root), out.getvalue())
            self.assertNotIn("a"*64, out.getvalue())
            self.assertNotIn("CONSUMED_H173_FORBIDDEN", out.getvalue())
            self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
