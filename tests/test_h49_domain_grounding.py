"""CPU replay of original H49 Arabic government-like and English FHIR-like GUI.

Public synthetic DEV is not independent blind TEST or true external app testing.
"""
from __future__ import annotations
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from h49.factory import FAMILIES, RIGHTS, SCHEMA, canonical, render, scenes, validate
from h49.audit import (DATASET, EVALUATOR, INPUTS, INTEGRITY, MANIFEST,
                       inspect, load_export, export)
ROOT=Path("h49/assets")

class H49DomainGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projections,cls.manifest=inspect(ROOT)
        cls.capture=json.loads((ROOT/"capture.json").read_text("utf-8"))

    def test_real_offline_browser_and_exact_pixel_source(self):
        summary=json.loads(self.projections[INTEGRITY])
        self.assertEqual(summary["screenshots"],96)
        self.assertEqual(summary["visible_buttons"],576)
        self.assertEqual(summary["correct_click_replays"],96)
        self.assertEqual(summary["wrong_target_rejections"],96)
        self.assertFalse(summary["real_service_access"])
        self.assertFalse(self.capture["network_used"])
        self.assertFalse(self.capture["native_actions"])
        self.assertFalse(self.capture["model_inference"])
        self.assertEqual(self.capture["puppeteer_core"],"25.12.0")
        self.assertFalse(self.manifest["independent_blind_test"])
        self.assertFalse(self.manifest["adapter_qualified"])
        self.assertFalse(self.manifest["training_executed"])

    def test_language_domain_allocation_and_family_isolation(self):
        rows=scenes();validate(rows)
        self.assertEqual(len(rows),96)
        sft=load_export(ROOT/DATASET,self.manifest)
        self.assertEqual(len(sft),64)
        self.assertEqual(sum(r["language"]=="ar" for r in sft),32)
        self.assertEqual(sum(r["language"]=="en" for r in sft),32)
        self.assertEqual(self.manifest["language_policy"]["arabic"],
                         ["justice_execution","traffic_services"])
        self.assertEqual(self.manifest["language_policy"]["english"],
                         ["fhir_resources","fhir_bundles"])
        self.assertFalse(set(self.manifest["families"]) & set(self.manifest["held_out_families"]))
        self.assertNotIn("municipal_portal",self.manifest["families"])
        self.assertNotIn("research_lab",self.manifest["families"])
        self.assertTrue(all(r["split"]=="train" for r in sft))
        self.assertTrue(all("target_index" not in r["prompt"]
                            and "target_key" not in r["prompt"] for r in sft))

    def test_semantic_goal_paraphrases_and_targets_are_variable(self):
        rows=scenes()
        for family,*_ in FAMILIES:
            cases=[x for x in rows if x["family"]==family]
            self.assertEqual(len(cases),16)
            self.assertEqual(set(x["target_key"] for x in cases),set(range(6)))
            self.assertEqual(len(set(x["goal"] for x in cases)),6)
            self.assertGreater(len(set(x["target_index"] for x in cases)),3)
            for c in cases:
                self.assertNotIn(c["goal"],[x["label"] for x in c["choices"]])
                self.assertEqual(c["choices"][c["target_index"]]["key"],c["target_key"])
                self.assertNotIn("absher.sa",render(c).decode().lower())
                self.assertNotIn("najiz.sa",render(c).decode().lower())
                self.assertEqual(c["rights"],RIGHTS)

    def test_source_bound_and_create_only(self):
        self.assertEqual((ROOT/MANIFEST).read_bytes(),self.projections[MANIFEST])
        self.assertEqual((ROOT/DATASET).read_bytes(),self.projections[DATASET])
        self.assertEqual((ROOT/INPUTS).read_bytes(),self.projections[INPUTS])
        self.assertEqual((ROOT/EVALUATOR).read_bytes(),self.projections[EVALUATOR])
        self.assertEqual((ROOT/INTEGRITY).read_bytes(),self.projections[INTEGRITY])
        self.assertEqual(hashlib.sha256(self.projections[DATASET]).hexdigest(),
                         self.manifest["dataset_sha256"])
        with self.assertRaises(FileExistsError):
            export(ROOT)

    def test_public_dev_is_not_private_or_training(self):
        model=json.loads(self.projections[INPUTS])
        oracle=json.loads(self.projections[EVALUATOR])
        self.assertEqual(len(model["cases"]),96)
        self.assertEqual(len(oracle["cases"]),96)
        self.assertFalse(model["independent_blind_test"])
        self.assertFalse(oracle["independent_blind_test"])
        for row in model["cases"]:
            self.assertNotIn("target_index",row)
            self.assertNotIn("target_key",row)
            self.assertNotIn("expected_control",row)
        self.assertEqual(sum(x["split"]=="dev" for x in oracle["cases"]),32)

    def test_wrong_target_and_forged_browser_geometry_rejected(self):
        mutations=[
            lambda x:x["records"][0]["golden"].update(selected=5),
            lambda x:x["records"][1]["negative"].update(selected=x["records"][1]["golden"]["selected"]),
            lambda x:x["records"][2]["controls"][0]["bounds"].__setitem__(1,620.0),
            lambda x:x["records"][3]["controls"][0].update(label="wrong"),
            lambda x:x["records"][4].update(png_sha256="0"*64),
            lambda x:x.update(real_service_access=True),
            lambda x:x["records"][70].update(split="train"),
        ]
        for i,mutation in enumerate(mutations):
            with self.subTest(tamper=i):
                capture=copy.deepcopy(self.capture)
                mutation(capture)
                with self.assertRaises(ValueError):
                    inspect(ROOT,capture_override=capture)

    def test_coherently_rehashed_forged_training_target_rejected(self):
        with tempfile.TemporaryDirectory() as dir:
            dest=Path(dir)/"h49"
            shutil.copytree(ROOT,dest)
            records=(dest/DATASET).read_text("utf-8").splitlines()
            first=json.loads(records[0])
            first["completion"]='{"type":"click","point":[0,0],"coordinate_space":"window","image_scale":1,"element_ref_hint":"visual_target"}'
            records[0]=json.dumps(first,sort_keys=True,ensure_ascii=False)
            (dest/DATASET).write_text("\n".join(records)+"\n",encoding="utf-8")
            forged=copy.deepcopy(self.manifest)
            forged["dataset_sha256"]=hashlib.sha256((dest/DATASET).read_bytes()).hexdigest()
            (dest/MANIFEST).write_bytes(canonical(forged))
            with self.assertRaises(ValueError):
                load_export(dest/DATASET,forged)

    def test_existing_remote_trainer_explicit_source_dispatch_without_gpu(self):
        from training.scripts.train_adapter import load_authoring_records
        rows=load_authoring_records(ROOT/DATASET,self.manifest)
        self.assertEqual(len(rows),64)
        prior=Path("h48/assets")
        h48=json.loads((prior/"h48-sft-manifest.json").read_text("utf-8"))
        self.assertEqual(len(load_authoring_records(prior/"h48-sft-authoring.jsonl",h48)),64)
        forged=copy.deepcopy(self.manifest)
        forged["source_schema"]="hala.h48.diverse-first-party-visual-grounding.v1"
        with self.assertRaises(ValueError):
            load_authoring_records(ROOT/DATASET,forged)

    def test_h14_frozen_source_unchanged(self):
        manifest=json.loads(Path("evaluation/heldout/manifest.json").read_text("utf-8"))
        frozen=Path("training/artifacts.py").read_bytes()
        self.assertEqual(hashlib.sha256(frozen).hexdigest(),
                         manifest["files"]["training/artifacts.py"])

if __name__=="__main__":
    unittest.main()
