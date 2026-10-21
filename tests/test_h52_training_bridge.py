"""H52 source-bound three-step LoRA loader bridge, CPU-only tests.

No model, CUDA, optimizer, official test screens or real service called.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from h52.audit import audit
from h52.training_bridge import (
    ADDITIONAL_FIELDS, DATASET_SHA256, FAMILIES_TRAIN, PUBLIC_DEV_FAMILIES,
    SOURCE_SHA256, emit_manifest, expected_training_manifest, load_export,
)
from training.scripts.train_adapter import load_authoring_records
from training.preflight import TrainingConfig, validate_experiment

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h52/assets"
MANIFEST=ROOT/"h52/remote_training_manifest.json"
BASE=ROOT/"h50/assets"


class H52TrainingBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.rows=load_export(SOURCE/"authoring_sft.jsonl",cls.manifest)

    def test_actual_288_three_step_rows_admitted(self):
        self.assertEqual(len(self.rows),288)
        self.assertEqual(len({r["image_sha256"] for r in self.rows}),288)
        self.assertEqual(sum(r["language"]=="ar" for r in self.rows),144)
        self.assertEqual(sum(r["language"]=="en" for r in self.rows),144)
        self.assertEqual(self.manifest["source_sha256"],SOURCE_SHA256)
        self.assertEqual(self.manifest["dataset_sha256"],DATASET_SHA256)
        self.assertEqual(self.rows,load_authoring_records(
            SOURCE/"authoring_sft.jsonl",self.manifest))

    def test_existing_real_public_dev_disjoint_by_family(self):
        self.assertEqual(self.manifest["families"],FAMILIES_TRAIN)
        self.assertEqual(self.manifest["held_out_families"],PUBLIC_DEV_FAMILIES)
        self.assertFalse(set(FAMILIES_TRAIN)&set(PUBLIC_DEV_FAMILIES))
        self.assertEqual(self.manifest,expected_training_manifest(SOURCE))
        self.assertFalse(self.manifest["model_training_executed"])
        self.assertFalse(self.manifest["independent_blind_test"])
        self.assertEqual(self.manifest["dev_or_benchmark_examples"],0)

    def test_actual_remote_preflight_accepts_only_real_inference_baseline(self):
        # This is a mock *preflight schema test*, not a claim that a model ran.
        cfg=TrainingConfig.from_dict(dict(
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            revision="89644892e4d85e24eaac8bacfd4f463576704203",
            precision="fp16",max_steps=3,batch_size=1,
            gradient_accumulation=1,learning_rate=0.0001,
            lora_rank=4,platform="colab",download_remote_weights=True))
        simulated=dict(model_id=cfg.model_id,revision=cfg.revision,
                       evidence="unit_test_mock_not_real_inference",
                       evaluated_tasks=8,verified_success_rate=0,
                       deficit="No actual model baseline was provided")
        with self.assertRaises(ValueError):
            validate_experiment(cfg,self.manifest,simulated)
        # Never synthesize a valid baseline or mark training complete.

    def test_strict_no_modified_manifest_or_eval_splits(self):
        mutation=[
            lambda v:v.update(dataset_sha256="0"*64),
            lambda v:v.update(source_sha256="0"*64),
            lambda v:v.update(split="test"),
            lambda v:v.update(source_kind="rights_cleared_public"),
            lambda v:v.update(train_rows=287),
            lambda v:v.update(independent_blind_test=True),
            lambda v:v.update(model_training_executed=True),
            lambda v:v.update(training_completed=True),
            lambda v:v.update(held_out_families=["en_fhir_workspace"]),
            lambda v:v.update(families=["en_clinical_ops"]),
            lambda v:v.update(arabic_image_training={"enabled":True}),
        ]
        for index,fn in enumerate(mutation):
            with self.subTest(index=index):
                modified=copy.deepcopy(self.manifest)
                fn(modified)
                with self.assertRaises(ValueError):
                    load_export(SOURCE/"authoring_sft.jsonl",modified)

    def test_empty_or_alias_path_rejected(self):
        with self.assertRaises(ValueError):
            load_export(SOURCE/"capture.json",self.manifest)
        with self.assertRaises(ValueError):
            load_export(SOURCE/"authoring_sft.jsonl",None)

    def test_create_only_byte_identical_derived_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"h52-manifest.json"
            outcome=emit_manifest(SOURCE,dest)
            self.assertFalse(outcome["real_model_training_executed"])
            self.assertEqual(dest.read_bytes(),MANIFEST.read_bytes())
            with self.assertRaises(FileExistsError):
                emit_manifest(SOURCE,dest)

    def test_rehashed_forged_training_rows_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)/"assets"
            shutil.copytree(SOURCE,copy)
            examples=copy/"authoring_sft.jsonl"
            records=examples.read_text(encoding="utf-8").splitlines()
            modified=json.loads(records[0])
            modified["completion"]='{"type":"click","point":[0,0],"coordinate_space":"window","image_scale":1,"element_ref_hint":"visual_target"}'
            records[0]=json.dumps(modified,ensure_ascii=False,sort_keys=True)
            examples.write_text("\n".join(records)+"\n",encoding="utf-8")
            original=json.loads((copy/"sft_manifest.json").read_text("utf-8"))
            original["dataset_sha256"]=hashlib.sha256(examples.read_bytes()).hexdigest()
            (copy/"sft_manifest.json").write_text(json.dumps(original),encoding="utf-8")
            claimed={**original,**ADDITIONAL_FIELDS}
            with self.assertRaises(ValueError):
                load_export(examples,claimed)

    def test_prior_h50_original_training_unchanged(self):
        import h50.counterfactual as h50
        before,_=h50.audit(BASE)
        self.assertEqual(len(before),384)
        self.assertEqual((BASE/"authoring_sft.jsonl").stat().st_size,240656)

if __name__=="__main__":
    unittest.main()
