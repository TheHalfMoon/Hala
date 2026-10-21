"""Offline review-only checks; synthetic metadata is NOT licensed training data."""
from __future__ import annotations
import copy
import unittest

from training.external_data_guard import (
    ATTESTATIONS, REGISTRY, admit_metadata, decision, registry,
)


class ExternalDataAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.rev = "a" * 40
        self.checks = dict.fromkeys(ATTESTATIONS, True)
        self.rows = [
            dict(id="trajectory-001", app="fictional-office", episode="episode-01",
                 image_sha256="b" * 64, benchmark_family=None,
                 partition="train", source_revision=self.rev)
        ]

    def admit(self, dataset="opencua-agentnet", partition="train", **kw):
        args = dict(source_id=dataset, revision=self.rev, split=partition,
                    blob_sha256="f" * 64, attestations=self.checks, items=self.rows)
        args.update(kw)
        return admit_metadata(**args)

    def test_registry_identifies_real_candidates_and_quarantines(self):
        self.assertEqual(decision("opencua-agentnet"), "CANDIDATE_ONLY")
        self.assertEqual(decision("hcompany-dragon-train"), "CANDIDATE_ONLY")
        self.assertEqual(decision("guiodyssey-v2"), "CANDIDATE_ONLY")
        for item in ("hcompany-holo4-benchmark-traces", "screenref",
                     "holo4-private-task-factory", "showui-web"):
            with self.subTest(item=item):
                self.assertEqual(decision(item), "DENY")

    def test_no_training_and_rights_not_proven_by_self_attestation(self):
        result = self.admit()
        self.assertTrue(result["candidate_not_yet_imported"])
        self.assertFalse(result["training_data_written"])
        self.assertFalse(result["trained_model"])
        self.assertFalse(result["rights_verified_by_tool"])
        self.assertTrue(result["attestations_operator_claimed_only"])
        self.assertEqual(result["reviewed_metadata_rows"], 1)
        self.assertEqual(result["declared_license"], "mit")

    def test_all_benchmark_only_and_restricted_data_denied(self):
        for source in ("hcompany-holo4-benchmark-traces", "screenref",
                       "holo4-private-task-factory", "showui-web"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.admit(dataset=source)

    def test_dragonly_original_train_and_never_eval(self):
        result = self.admit(dataset="hcompany-dragon-train")
        self.assertEqual(result["split"], "train")
        for split in ("eval", "test", "public_dev", "all"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.admit(dataset="hcompany-dragon-train", partition=split)

    def test_guiodyssey_requires_verified_official_app_train(self):
        for split in ("all", "train", "task_split", "app_split", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.admit(dataset="guiodyssey-v2", partition=split)
        rows = copy.deepcopy(self.rows)
        rows[0]["partition"] = "verified_official_app_split_train"
        result = self.admit(dataset="guiodyssey-v2",
                            partition="verified_official_app_split_train", items=rows)
        self.assertEqual(result["declared_license"], "cc-by-4.0")
        self.assertTrue(result["retain_attribution"])

    def test_missing_external_license_privacy_and_benchmark_checks(self):
        for key in ATTESTATIONS:
            with self.subTest(key=key):
                altered = dict(self.checks)
                altered[key] = False
                with self.assertRaises(ValueError):
                    self.admit(attestations=altered)
        with self.assertRaises(ValueError):
            self.admit(attestations={"source_license_verified": True})

    def test_reject_unpinned_revision_and_unverified_blob_sha(self):
        for revision in ("main", "a" * 39, "a" * 41, "a" * 40 + "x"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.admit(revision=revision)
        with self.assertRaises(ValueError):
            self.admit(blob_sha256="not-a-cryptographic-digest")

    def test_reject_undeclared_benchmark_partition_and_wrong_revision(self):
        mutated = copy.deepcopy(self.rows)
        mutated[0]["benchmark_family"] = "OSWorld"
        with self.assertRaises(ValueError):
            self.admit(items=mutated)
        mutated[0]["benchmark_family"] = None
        mutated[0]["partition"] = "test"
        with self.assertRaises(ValueError):
            self.admit(items=mutated)
        mutated[0]["partition"] = "train"
        mutated[0]["source_revision"] = "c" * 40
        with self.assertRaises(ValueError):
            self.admit(items=mutated)

    def test_reject_public_benchmark_identity_even_with_train_label(self):
        row = copy.deepcopy(self.rows[0])
        row["episode"] = "osworld2-verified-test"
        with self.assertRaises(ValueError):
            self.admit(items=[row])

    def test_reject_raw_unreviewed_images_duplicate_pixels_and_ids(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["raw_image_bytes"] = "MALICIOUS"
        with self.assertRaises(ValueError):
            self.admit(items=rows)
        rows = copy.deepcopy(self.rows) * 2
        with self.assertRaises(ValueError):
            self.admit(items=rows)
        rows[1] = dict(rows[1], id="trajectory-002", episode="episode-02")
        with self.assertRaises(ValueError):
            self.admit(items=rows)
        with self.assertRaises(ValueError):
            self.admit(items=[])

    def test_no_mutation_of_existing_h49_h50_assets(self):
        self.assertTrue(REGISTRY.is_file())
        contents = registry()
        self.assertIn("hcompany-holo4-benchmark-traces", contents)
        self.assertIn("opencua-agentnet", contents)
        self.assertTrue(contents["hcompany-holo4-benchmark-traces"]["benchmark_overlap"])
        self.assertFalse(contents["opencua-agentnet"]["training_partition"] == [])
        self.assertFalse(contents["opencua-agentnet"]["release_use_status"] == "READY")


if __name__ == "__main__":
    unittest.main()
