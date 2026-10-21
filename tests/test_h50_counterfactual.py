"""H50 source-bound counterfactual training qualification.

All screenshots are first-party original *fictional* mock browser images; no
real Najiz, Absher, FHIR patient, GPU or real model output in these tests.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from h49.factory import scenes
from h50.counterfactual import SCHEMA,TRAIN,DEV,audit,load_export,export
from training.scripts.train_adapter import load_authoring_records
ROOT=Path("h50/assets")
H49=Path("h49/assets")

class H50CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.manifest=audit(ROOT)
        cls.replay=json.loads((ROOT/"click_replay.json").read_text("utf-8"))

    def test_not_384_original_images_and_never_claim_learned_model(self):
        self.assertEqual(len(self.rows),384)
        self.assertEqual(self.manifest["source_screenshots"],64)
        self.assertEqual(len(self.manifest["media"]),64)
        self.assertEqual(self.manifest["training_rows"],384)
        self.assertFalse(self.manifest["training_executed"])
        self.assertFalse(self.manifest["adapter_qualified"])
        self.assertFalse(self.manifest["independent_blind_test"])
        self.assertEqual(sum(r["language"]=="ar" for r in self.rows),192)
        self.assertEqual(sum(r["language"]=="en" for r in self.rows),192)
        self.assertEqual(self.manifest["licensed_external_images"],0)

    def test_six_different_goal_labels_on_each_exact_same_image(self):
        from collections import defaultdict
        groups=defaultdict(list)
        for row in self.rows:
            groups[row["image_path"]].append(row)
        self.assertEqual(len(groups),64)
        for name,group in groups.items():
            with self.subTest(name=name):
                self.assertEqual(len(group),6)
                self.assertEqual(len({r["prompt"] for r in group}),6)
                self.assertEqual(len({r["image_sha256"] for r in group}),1)
                points={tuple(json.loads(r["completion"])["point"]) for r in group}
                self.assertEqual(len(points),6)
                self.assertEqual([r["id"].rsplit("-k",1)[-1] for r in group],
                                 list("012345"))
                self.assertTrue(all(r["split"]=="train" for r in group))

    def test_exact_h49_media_and_train_only_not_dev(self):
        trainids={r["id"] for r in scenes() if r["split"]=="train"}
        devids={r["id"] for r in scenes() if r["split"]=="dev"}
        source_ids={r["image_path"][4:-4] for r in self.rows}
        self.assertEqual(source_ids,trainids)
        self.assertFalse(source_ids & devids)
        self.assertEqual({r["family"] for r in self.rows},set(TRAIN))
        self.assertEqual(set(self.manifest["held_out_families"]),set(DEV))
        self.assertEqual(self.manifest["dev_used_as_training"],0)
        for item in self.manifest["media"]:
            self.assertEqual((ROOT/item["path"]).read_bytes(),
                             (H49/item["path"]).read_bytes())
            self.assertEqual(hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest(),
                             item["sha256"])

    def test_every_original_mock_button_actually_clicked_in_edge(self):
        self.assertEqual(self.replay["case_count"],64)
        self.assertEqual(self.replay["click_count"],384)
        self.assertFalse(self.replay["real_service_access"])
        self.assertFalse(self.replay["model_inference"])
        self.assertEqual(len(self.replay["cases"]),64)
        for record in self.replay["cases"]:
            self.assertEqual(record["clicked_keys"],list(range(6)))
            self.assertEqual(record["selected_keys"],list(range(6)))
            self.assertEqual(record["click_counts"],list(range(1,7)))

    def test_h50_trainer_ingests_exact_originals_and_preserves_h49(self):
        self.assertEqual(load_export(ROOT/"authoring_sft.jsonl",self.manifest),self.rows)
        self.assertEqual(load_authoring_records(ROOT/"authoring_sft.jsonl",self.manifest),
                         self.rows)
        prior=Path("h49/assets")
        h49=json.loads((prior/"sft_manifest.json").read_text("utf-8"))
        self.assertEqual(len(load_authoring_records(prior/"authoring_sft.jsonl",h49)),64)
        with self.assertRaises(FileExistsError):
            export(ROOT)

    def test_browser_negative_failure_refuses_false_source(self):
        mutate=[
            lambda x:x["cases"][0]["selected_keys"].__setitem__(3,0),
            lambda x:x["cases"][0]["click_counts"].__setitem__(2,0),
            lambda x:x["cases"][1].update(png_sha256="0"*64),
            lambda x:x.update(click_count=383),
            lambda x:x.update(model_inference=True),
            lambda x:x["cases"].pop()
        ]
        for i,change in enumerate(mutate):
            with self.subTest(tamper=i):
                example=copy.deepcopy(self.replay)
                change(example)
                with self.assertRaises(ValueError):
                    audit(ROOT,override_replay=example)

    def test_rehashed_forged_completion_not_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/"h50/assets"
            dest.parent.mkdir(parents=True)
            shutil.copytree(ROOT,dest)
            original=Path(directory)/"h49/assets"
            original.parent.mkdir(parents=True)
            shutil.copytree(H49,original)
            path=dest/"authoring_sft.jsonl"
            rows=path.read_text("utf-8").splitlines()
            first=json.loads(rows[0])
            first["completion"]='{"type":"click","point":[0,0],"coordinate_space":"window","image_scale":1,"element_ref_hint":"visual_target"}'
            rows[0]=json.dumps(first,sort_keys=True,ensure_ascii=False)
            path.write_text("\n".join(rows)+"\n",encoding="utf-8")
            forged=copy.deepcopy(self.manifest)
            forged["dataset_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_export(path,forged)

    def test_h50_is_compatible_with_frozen_h49_public_prompt(self):
        from h49.paired_eval import public_prompt
        options={f:rows for f,split,language,title,layout,rows
                 in __import__("h49.factory",fromlist=["FAMILIES"]).FAMILIES}
        for row in self.rows:
            key=int(row["id"].rsplit("-k",1)[1])
            self.assertEqual(row["prompt"],public_prompt(options[row["family"]][key][1]))
            self.assertNotIn("target_index",row["prompt"])
            self.assertNotIn("target_key",row["prompt"])

if __name__=="__main__":
    unittest.main()
