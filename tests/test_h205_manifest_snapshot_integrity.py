"""H205: source cohort snapshots must be the SAME bytes throughout H204.

Mutation tests are entirely synthetic; never read owner images, labels, OOD.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_h204_training_readiness_gap import fixtures
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h198.owner_source_metadata_readiness import COHORTS, census
from h204.training_readiness_gap import training_readiness_gap
from h199.native_action_sft import verify_real_training_release


def change_instruction(owner: Path, marker: str) -> None:
    path = owner / COHORTS[0][0] / "source-prospects-not-admitted.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    # This mutation preserves all case/PNG/RGBA counts and no source splits;
    # the old H204 check accepted it because no snapshot equality existed.
    record["cases"][0]["instruction"] = marker
    path.write_text(json.dumps(record), encoding="utf-8")


def change_source_sha(owner: Path) -> None:
    path = owner / COHORTS[0][0] / "source-prospects-not-admitted.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    # Valid 64-character source metadata hash, not an old SHA collision.
    record["cases"][0]["image_sha256"] = "a"*64
    path.write_text(json.dumps(record), encoding="utf-8")


class H205SourceSnapshotTests(unittest.TestCase):
    def test_stable_consistent_metadata_passes_only_as_denial(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            proof = training_readiness_gap(owner, [phase])
            self.assertTrue(proof["current_source_metadata_stable_across_audit"])
            self.assertFalse(proof["train_admission_approved"])
            self.assertFalse(proof["complete_historical_old_rgba_proven"])
            self.assertEqual(proof["private_gold_opened"], 0)
            self.assertEqual(proof["private_media_opened"], 0)

    def test_instruction_change_after_h198_census_is_rejected(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            def tampered_census(root):
                original = census(root)
                change_instruction(root, "Changed after H198 audit")
                return original
            with patch("h204.training_readiness_gap.census",
                       side_effect=tampered_census):
                with self.assertRaisesRegex(
                    ValueError, "source cohort metadata changed"
                ):
                    training_readiness_gap(owner, [phase])

    def test_image_sha_change_after_h198_census_is_rejected(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            def tampered_census(root):
                original = census(root)
                change_source_sha(root)
                return original
            with patch("h204.training_readiness_gap.census",
                       side_effect=tampered_census):
                with self.assertRaisesRegex(
                    ValueError, "source cohort metadata changed"
                ):
                    training_readiness_gap(owner, [phase])

    def test_change_while_h185_old_sha_scanned_is_rejected(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            def tampered_legacy(roots):
                proof = scan_legacy_manifest_sha_fields(roots)
                change_instruction(owner, "Changed during legacy SHA audit")
                return proof
            with patch("h204.training_readiness_gap.scan_legacy_manifest_sha_fields",
                       side_effect=tampered_legacy):
                with self.assertRaisesRegex(
                    ValueError, "source cohort metadata changed"
                ):
                    training_readiness_gap(owner, [phase])

    def test_privacy_error_excludes_private_identifiers_and_gold(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            def mutated(root):
                original = census(root)
                change_instruction(root, "PRIVATE-SYNTHETIC-CANARY")
                return original
            with patch("h204.training_readiness_gap.census",
                       side_effect=mutated):
                with self.assertRaises(ValueError) as caught:
                    training_readiness_gap(owner, [phase])
            msg = str(caught.exception)
            self.assertNotIn("PRIVATE-SYNTHETIC-CANARY", msg)
            self.assertNotIn(str(owner), msg)
            self.assertNotIn(str(phase), msg)

    def test_verified_snapshot_still_cannot_admit_forged_train_release(self):
        with TemporaryDirectory() as tmp:
            owner, phase = fixtures(Path(tmp))
            proof = training_readiness_gap(owner, [phase])
            self.assertTrue(proof["current_source_metadata_stable_across_audit"])
            forged = {
                "schema": "hala.original-native-train-release.v1",
                "manifest_sha256": "f" * 64,
                "split": "TRAIN",
                "official_admitted": True,
                "complete_source_history_proven": True,
                "complete_old_rgba_proven": True,
                "sealed_dev_gold": True,
                "consumed_ood_reused": False,
            }
            with self.assertRaisesRegex(ValueError, "no independently pinned"):
                verify_real_training_release(forged)


if __name__ == "__main__":
    unittest.main()
