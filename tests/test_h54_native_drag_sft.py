"""CPU-only H54 native ordered-drag SFT bridge tests with original FIRST-PARTY fake media.

Uses FIVE newly generated fake test JPEGs and 15 fake annotations,
never publishes or loads real H Company media in GitHub CI.
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from h53.dragon_train_shard import SCHEMA as SOURCE_SCHEMA,verify_item
from h54 import native_drag as mod
from training.scripts.train_adapter import load_authoring_records
from training.artifacts import validate_action_completion


def fake_jpeg():
    im=Image.new("RGB",(900,700),color=(40,80,120))
    buffer=BytesIO()
    im.save(buffer,"JPEG")
    return buffer.getvalue()


def make_fixture(root):
    root=Path(root)
    (root/"images").mkdir()
    (root/"tasks").mkdir()
    image=fake_jpeg()
    entries=[]
    records=[]
    for i in range(5):
        key=f"11111111-1111-1111-1111-{i:012d}"
        task=[dict(image_id=key,domain="slider",ordered=True,
              intent=f"Move fictional test slider {j+1} steps",
              start_bbox=[120,340,145,365],
              end_bbox=[200+j*45,340,225+j*45,365],
              metadata=dict(fake_test_only=True)) for j in range(3)]
        raw=(json.dumps(task,indent=2,sort_keys=True)+"\n").encode("utf8")
        (root/"images"/(key+".jpg")).write_bytes(image)
        (root/"tasks"/(key+".json")).write_bytes(raw)
        records.extend(verify_item(key,image,raw))
        entries.append(dict(image_id=key,image_sha256=hashlib.sha256(image).hexdigest(),
          task_sha256=hashlib.sha256(raw).hexdigest(),
          source_image_range=[0,len(image)],source_task_range=[0,len(raw)],
          examples=3))
    jsonl=b"".join(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()+b"\n"
                 for r in records)
    (root/"train_drag.jsonl").write_bytes(jsonl)
    source_hash=hashlib.sha256(jsonl).hexdigest()
    native_manifest=dict(schema=SOURCE_SCHEMA,source_revision="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
         source_file="train/slider.tar",source_train_only=True,source_eval_used=False,
         real_model_training_executed=False,remote_storage_verified=False,
         full_source_tar_sha256_independently_verified=False,
         jsonl_sha256=source_hash,matched_source_entries=entries,
         unique_original_screenshots=5,train_rows=15)
    (root/"manifest.json").write_text(json.dumps(native_manifest),encoding="utf8")
    receipt=root/"fake-source-evidence.json"
    receipt.write_text(json.dumps(dict(source_revision=native_manifest["source_revision"],
       source_file="train/slider.tar",jsonl_sha256=source_hash,
       source_eval_used=False,kaggle_dataset_id=12752110,
       authenticated_private_listing_verified=True,
       matched_source_sha256=entries)),encoding="utf8")
    return receipt,source_hash


class NativeDragSftTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.receipt,self.sourcehash=make_fixture(self.root)
        self.patches=[
          patch.object(mod,"SOURCE_RECEIPT",self.receipt),
          patch.object(mod,"ORIGINAL_SHA256",self.sourcehash),
          patch.object(mod,"ORIGINAL_IMAGES",5),
          patch.object(mod,"ORIGINAL_ACTIONS",15),
        ]
        for handle in self.patches:
            handle.start()
            self.addCleanup(handle.stop)

    def test_true_source_native_drag_completion_and_points(self):
        result=mod.export(self.root)
        self.assertEqual(result["genuine_ordered_drag_labels"],15)
        manifest=json.loads((self.root/mod.META_NAME).read_text("utf8"))
        rows=load_authoring_records(self.root/mod.ROWS_NAME,manifest)
        self.assertEqual(len(rows),15)
        self.assertEqual(manifest["native_action_type"],"drag")
        self.assertEqual(manifest["media"][0]["family"],mod.FAMILY)
        self.assertFalse(manifest["independent_drag_dev_available"])
        completion=json.loads(rows[0]["completion"])
        self.assertEqual(completion["type"],"drag")
        self.assertEqual(completion["start_point"],[132,352])
        self.assertEqual(completion["end_point"],[212,352])
        self.assertIn("original screenshot",rows[0]["prompt"])
        with self.assertRaises(ValueError):validate_action_completion(rows[0]["completion"])

    def test_create_only_no_silent_source_mutation(self):
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
             self.root/"train_drag.jsonl",self.root/"manifest.json",self.receipt)}
        mod.export(self.root)
        with self.assertRaises(FileExistsError):mod.export(self.root)
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
             self.root/"train_drag.jsonl",self.root/"manifest.json",self.receipt)}
        self.assertEqual(before,after)

    def test_manifest_tamper_and_dataset_data_tamper_fail_closed(self):
        mod.export(self.root)
        meta=json.loads((self.root/mod.META_NAME).read_text("utf8"))
        bad=dict(meta,dataset_sha256="0"*64)
        with self.assertRaises(ValueError):mod.load_export(self.root/mod.ROWS_NAME,bad)
        bad=dict(meta,independent_drag_dev_available=True)
        with self.assertRaises(ValueError):mod.load_export(self.root/mod.ROWS_NAME,bad)
        raw=(self.root/mod.ROWS_NAME)
        with raw.open("ab") as f:f.write(b"\n")
        with self.assertRaises(ValueError):mod.load_export(raw,meta)

    def test_original_json_tamper_rejected_even_if_label_unmodified(self):
        mod.export(self.root)
        file=next((self.root/"tasks").glob("*.json"))
        data=json.loads(file.read_text("utf8"))
        data[0]["end_bbox"]=[0,0,10,10]
        file.write_text(json.dumps(data),encoding="utf8")
        meta=json.loads((self.root/mod.META_NAME).read_text("utf8"))
        with self.assertRaises(ValueError):mod.load_export(self.root/mod.ROWS_NAME,meta)

    def test_orig_source_image_tamper_rejected(self):
        mod.export(self.root)
        photo=next((self.root/"images").glob("*.jpg"))
        with photo.open("ab") as f:f.write(b"altered")
        meta=json.loads((self.root/mod.META_NAME).read_text("utf8"))
        with self.assertRaises(ValueError):mod.load_export(self.root/mod.ROWS_NAME,meta)

    def test_model_baseline_requires_real_independent_drag_inference(self):
        fake=dict(evidence="unit_test_mock",task_kind="ordered_drag",
          held_out_source="original_app_disjoint_drag_not_training",
          held_out_dataset_sha256="1"*64,evaluated_tasks=32)
        with self.assertRaises(ValueError):mod.validate_drag_baseline(fake)
        fake["evidence"]="real_inference"
        fake["evaluated_tasks"]=1
        with self.assertRaises(ValueError):mod.validate_drag_baseline(fake)
        fake["evaluated_tasks"]=32
        fake["held_out_source"]="train/slider.tar"
        with self.assertRaises(ValueError):mod.validate_drag_baseline(fake)
        fake["held_out_source"]="original_app_disjoint_drag_not_training"
        self.assertTrue(mod.validate_drag_baseline(fake))  # Schema only, not verified run.
        # Do not pass fake to any remote training job.

    def test_processor_path_accepts_verified_jpeg_only_for_native_drag(self):
        import sys
        from types import SimpleNamespace
        from training.scripts.train_adapter import prepare_inputs
        from unittest.mock import patch
        class Vector:
            def __init__(self, items):
                self.items=items
            def tolist(self):
                return self.items
        class Batch:
            shape=(1,)
            dtype="int64"
            def __init__(self, items):
                self.items=items
            def __getitem__(self,index):
                if index!=0: raise AssertionError
                return Vector(self.items)
        class Processor:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False):
                return "short_prompt" if len(messages)==1 else "short_prompt_completion"
            def __call__(self, text, images, return_tensors):
                if images[0].size!=(900,700):raise AssertionError
                return SimpleNamespace(input_ids=Batch(
                    [10,11] if text==["short_prompt"] else [10,11,12,13]))
        torch=SimpleNamespace(tensor=lambda rows,dtype:rows)
        mod.export(self.root)
        manifest=json.loads((self.root/mod.META_NAME).read_text("utf8"))
        row=load_authoring_records(self.root/mod.ROWS_NAME,manifest)[0]
        with patch.dict(sys.modules,{"torch":torch}):
            inp,label=prepare_inputs(row,Processor(),self.root,4096,
                                     native_drag_jpeg=True)
            self.assertEqual(inp.input_ids.shape,(1,))
            self.assertEqual(len(label[0]),4)
            with self.assertRaises(ValueError):
                prepare_inputs(row,Processor(),self.root,4096,
                               native_drag_jpeg=False)

    def test_click_only_source_format_remains_strict(self):
        from h52.factory import VERSION
        self.assertNotEqual(mod.SCHEMA,VERSION)
        self.assertEqual(mod.RESERVED_DEV_FAMILY,"original_disjoint_drag_holdout_reserved")


if __name__=="__main__":
    unittest.main()
