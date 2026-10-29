"""H204 strict training gap gate: synthetic manifests, never image/old OOD."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h174.source_candidates import CANDIDATES
from h195.original_vanilla_datepicker_native import PIN
from h198.owner_source_metadata_readiness import COHORTS
from h204.training_readiness_gap import (
    training_readiness_gap, _safe_legacy_phases, main,
)


def fixtures(root: Path, *, old_rgba: bool = False) -> tuple[Path, Path]:
    owner = root / "prospective"
    owner.mkdir()
    for i, (cohort, size, family, status) in enumerate(COHORTS):
        folder = owner / cohort
        folder.mkdir()
        pin = PIN if family == "vanilla-datepicker" else CANDIDATES[family]
        cases = [{
            "case_id": f"synthetic-source-{i}-{j}",
            "source_repo": pin["repo"],
            "source_git_sha": pin["git_sha"],
            "language": "en" if j % 2 == 0 else "ar",
            "instruction": "Select original visible source control.",
            "image_sha256": f"{i*100+j+1:064x}",
            "visual_rgba_sha256": f"{i*100+j+1001:064x}",
            "train_eligible": False, "dev_eligible": False, "ood_eligible": False,
        } for j in range(size)]
        (folder / "source-prospects-not-admitted.json").write_text(
            json.dumps({
                "status": status,
                "official_train_dev_dataset_admitted": False,
                "cases": cases,
            }), encoding="utf-8"
        )
        # Must never be opened or JSON parsed; deliberately invalid text.
        (folder / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
            "INVALID-SEALED-GOLD-NEVER-READ", encoding="utf-8"
        )
    phase = root / "private-h140"
    manifest = phase / "metadata" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    old = {"image_sha256": "f"*64}
    if old_rgba:
        old["visual_rgba_sha256"] = "e"*64
    manifest.write_text(json.dumps(old), encoding="utf-8")
    # Owner image bytes must not be needed by any test.
    (manifest.parent / "old-image.png").write_bytes(b"never read media")
    return owner, phase


class H204TrainingReadinessGapTests(unittest.TestCase):
    def test_actual_metadata_fields_audit_denies_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            result = training_readiness_gap(owner, [phase])
            self.assertEqual(result["new_unadmitted_original_gui_cases"], 20)
            self.assertEqual(result["new_source_families"], 3)
            self.assertEqual(result["languages"], {"en": 10, "ar": 10})
            self.assertEqual(result["legacy_manifests_scanned"], 1)
            self.assertEqual(result["distinct_known_legacy_png_sha_candidates"], 1)
            self.assertEqual(result["distinct_known_legacy_rgba_sha_candidates"], 0)
            self.assertEqual(result["legacy_manifests_lacking_rgba_sha"], 1)
            self.assertFalse(result["train_admission_approved"])
            self.assertFalse(result["complete_historical_old_rgba_proven"])
            self.assertFalse(result["complete_historical_source_history_proven"])
            self.assertFalse(result["official_train_sha_allowlist_nonempty"])
            self.assertEqual(result["private_media_opened"], 0)

    def test_legacy_rgba_fields_do_not_prove_full_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp), old_rgba=True)
            result = training_readiness_gap(owner, [phase])
            self.assertEqual(result["legacy_manifests_lacking_rgba_sha"], 0)
            self.assertFalse(result["complete_historical_old_rgba_proven"])
            self.assertFalse(result["train_admission_approved"])

    def test_known_old_png_collision_is_a_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            old = phase / "metadata" / "manifest.json"
            old.write_text(json.dumps({"image_sha256": f"{1:064x}"}))
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                training_readiness_gap(owner, [phase])

    def test_known_old_rgba_collision_is_a_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp), old_rgba=True)
            old = phase / "metadata" / "manifest.json"
            old.write_text(json.dumps({
                "image_sha256": "f"*64,
                "visual_rgba_sha256": f"{1001:064x}",
            }))
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                training_readiness_gap(owner, [phase])

    def test_never_read_gold_or_any_png_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            previous = Path.read_bytes
            paths = []
            def guarded(path):
                paths.append(path.name)
                if path.name not in (
                    "manifest.json", "source-prospects-not-admitted.json",
                ):
                    raise AssertionError("H204/H205 forbidden media/gold read")
                return previous(path)
            with patch.object(Path, "read_bytes", guarded):
                result = training_readiness_gap(owner, [phase])
            self.assertTrue(result["known_old_png_rgba_overlap_excluded_for_all_new_candidates"])
            self.assertTrue(result["current_source_metadata_stable_across_audit"])
            self.assertEqual(paths.count("manifest.json"), 1)
            self.assertEqual(
                paths.count("source-prospects-not-admitted.json"), 15
            )

    def test_consumed_h173_legacy_root_never_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            old_ood = phase.parent / "Hala-H173-Private"
            old_ood.mkdir()
            with self.assertRaisesRegex(ValueError, "pre-H173"):
                training_readiness_gap(owner, [old_ood])
            self.assertEqual(_safe_legacy_phases([phase]), [phase])

    def test_newer_case_cannot_masquerade_as_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            new = phase.parent / "Hala-H190-Private"
            new.mkdir()
            with self.assertRaisesRegex(ValueError, "pre-H173"):
                training_readiness_gap(owner, [new])

    def test_missing_or_duplicate_phases_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            for roots in ([], [phase, phase], [phase.parent / "private-h141"]):
                with self.subTest(roots=len(roots)):
                    with self.assertRaises(ValueError):
                        training_readiness_gap(owner, roots)

    def test_false_source_admission_blocks_legacy_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            source = owner / COHORTS[0][0] / "source-prospects-not-admitted.json"
            data = json.loads(source.read_text(encoding="utf-8"))
            data["official_train_dev_dataset_admitted"] = True
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "admission flag changed"):
                training_readiness_gap(owner, [phase])

    def test_cli_only_aggregate_counts_and_no_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            args = ["--owner-root", str(owner), "--legacy-phase", str(phase)]
            with patch("sys.argv", ["h204"] + args), redirect_stdout(StringIO()) as o:
                main()
            output = o.getvalue()
            self.assertIn("H204_TRAINING_RELEASE_BLOCKED", output)
            self.assertIn("legacy_rgba=0", output)
            self.assertNotIn(str(phase), output)
            self.assertNotIn("f"*64, output)
            self.assertNotIn("source-", output)
            self.assertIn("NO_OLD_OOD_NO_MODEL", output)


if __name__ == "__main__":
    unittest.main()
