"""H198 20-case source-only readiness mutations, never native media/gold."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h174.source_candidates import CANDIDATES
from h195.original_vanilla_datepicker_native import PIN
from h198.owner_source_metadata_readiness import (
    COHORTS, EXPECTED_TOTAL, census, main,
)


def prepare(root: Path) -> None:
    for index, (name, count, family, status) in enumerate(COHORTS):
        folder = root / name
        folder.mkdir()
        pin = PIN if family == "vanilla-datepicker" else CANDIDATES[family]
        cases = []
        for i in range(count):
            cases.append({
                "case_id": f"private-native-original-{index}-{i}",
                "source_repo": pin["repo"],
                "source_git_sha": pin["git_sha"],
                "language": ("en" if i % 2 == 0 else "ar"),
                "instruction": "Choose the original source target.",
                "image_sha256": f"{index*100+i+1:064x}",
                "visual_rgba_sha256": f"{index*100+i+1001:064x}",
                "train_eligible": False,
                "dev_eligible": False,
                "ood_eligible": False,
            })
        (folder / "source-prospects-not-admitted.json").write_text(
            json.dumps({
                "status": status,
                "official_train_dev_dataset_admitted": False,
                "cases": cases,
            }), encoding="utf-8",
        )
        # Intentionally unparseable gold: H198 MUST NOT read it.
        (folder / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
            "SEALED-PRIVATE-REFERENCE-NEVER-OPEN", encoding="utf-8"
        )
        # Files are intentionally absent: H198 MUST NOT open images.


def mutate(root: Path, index: int, change):
    folder = root / COHORTS[index][0]
    path = folder / "source-prospects-not-admitted.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


class SourceOnlyTwentyReadinessTests(unittest.TestCase):
    def test_twenty_exact_source_roles_ten_ar_ten_en_no_admission(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root)
            result = census(root)
            self.assertEqual(result["total_quarantined_original_preaction_png_metadata"], 20)
            self.assertEqual(result["source_implementation_families"], 3)
            self.assertEqual(result["source_family_counts"], {
                "pickr":14,"interact-dropzones":4,"vanilla-datepicker":2
            })
            self.assertEqual(result["languages"], {"en":10,"ar":10})
            self.assertEqual(result["unique_png_shas"],20)
            self.assertEqual(result["unique_rgba_shas"],20)
            self.assertFalse(result["training_approved"])
            self.assertFalse(result["dev_approved"])
            self.assertFalse(result["ood_approved"])
            self.assertFalse(result["historic_source_and_rgba_history_complete"])
            self.assertEqual(result["data_gold_or_image_files_opened"],0)
            self.assertEqual(result["gpu_optimizer_steps"],0)

    def test_sealed_unparseable_reference_does_not_get_opened(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root)
            self.assertEqual(census(root)["data_gold_or_image_files_opened"],0)

    def test_missing_cohort_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            (root/COHORTS[0][0]/"source-prospects-not-admitted.json").unlink()
            with self.assertRaisesRegex(ValueError,"source SHA provenance"):
                census(root)

    def test_missing_reference_sidecar_fails_closed_without_reading(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            (root/COHORTS[1][0]/"OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").unlink()
            with self.assertRaisesRegex(ValueError,"sealed private reference"):
                census(root)

    def test_false_admission_and_missing_source_history_flags_rejected(self):
        for key in ("official_train_dev_dataset_admitted",
                    "historical_source_complete",
                    "historical_decoded_rgba_complete"):
            with self.subTest(key=key),TemporaryDirectory() as tmp:
                root=Path(tmp);prepare(root)
                mutate(root,2,lambda d:d.update({key:True}))
                with self.assertRaisesRegex(ValueError,"admission flag changed"):
                    census(root)

    def test_any_case_eligibility_promotion_rejected(self):
        for key in ("train_eligible","dev_eligible","ood_eligible"):
            with self.subTest(key=key),TemporaryDirectory() as tmp:
                root=Path(tmp);prepare(root)
                mutate(root,3,lambda d:d["cases"][0].update({key:True}))
                with self.assertRaisesRegex(ValueError,"unauthorized"):
                    census(root)

    def test_reused_png_hash_across_original_cohorts_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            source=json.loads((root/COHORTS[0][0]/"source-prospects-not-admitted.json").read_text())
            old_hash=source["cases"][0]["image_sha256"]
            mutate(root,4,lambda d:d["cases"][0].update({"image_sha256":old_hash}))
            with self.assertRaisesRegex(ValueError,"source overlap"):
                census(root)

    def test_reused_decoded_rgba_hash_across_original_cohorts_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            source=json.loads((root/COHORTS[0][0]/"source-prospects-not-admitted.json").read_text())
            old_hash=source["cases"][0]["visual_rgba_sha256"]
            mutate(root,4,lambda d:d["cases"][0].update({"visual_rgba_sha256":old_hash}))
            with self.assertRaisesRegex(ValueError,"source overlap"):
                census(root)

    def test_source_git_pin_or_original_repo_tamper_rejected(self):
        for key,value in (("source_git_sha","f"*40),("source_repo","other/donor")):
            with self.subTest(key=key),TemporaryDirectory() as tmp:
                root=Path(tmp);prepare(root)
                mutate(root,3,lambda d:d["cases"][0].update({key:value}))
                with self.assertRaisesRegex(ValueError,"upstream original"):
                    census(root)

    def test_duplicate_case_identity_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            first=json.loads((root/COHORTS[0][0]/"source-prospects-not-admitted.json").read_text())
            case_id=first["cases"][0]["case_id"]
            mutate(root,1,lambda d:d["cases"][0].update({"case_id":case_id}))
            with self.assertRaisesRegex(ValueError,"repeated original case"):
                census(root)

    def test_bad_language_or_missing_instruction_rejected(self):
        for key,value,message in (
            ("language","fr","bilingual"),
            ("instruction","","instruction"),
        ):
            with self.subTest(key=key),TemporaryDirectory() as tmp:
                root=Path(tmp);prepare(root)
                mutate(root,1,lambda d:d["cases"][0].update({key:value}))
                with self.assertRaisesRegex(ValueError,message):
                    census(root)

    def test_known_historic_old_png_reuse_blocks_all_quarantine(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            source=json.loads((root/COHORTS[4][0]/"source-prospects-not-admitted.json").read_text())
            history=root/"historical"
            phase=history/"private-h140"
            phase.mkdir(parents=True)
            (phase/"manifest.json").write_text(
                json.dumps({"image_sha256":source["cases"][1]["image_sha256"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError,"reuses known old PNG"):
                census(root,history)

    def test_hist_empty_cannot_claim_history_screening(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            history=root/"historical";history.mkdir()
            with self.assertRaisesRegex(ValueError,"historical"):
                census(root,history)

    def test_no_private_paths_sha_case_ids_or_gold_in_public_receipt(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);prepare(root)
            record=json.loads((root/COHORTS[1][0]/"source-prospects-not-admitted.json").read_text())
            receipt=census(root)
            output=json.dumps(receipt,sort_keys=True)
            self.assertNotIn(str(root),output)
            self.assertNotIn(record["cases"][0]["case_id"],output)
            self.assertNotIn(record["cases"][0]["image_sha256"],output)
            self.assertNotIn("SEALED-PRIVATE",output)
            self.assertFalse(receipt["training_approved"])


if __name__=="__main__":
    unittest.main()
