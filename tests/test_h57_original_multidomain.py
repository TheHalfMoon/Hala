"""H57 real train multi-domain code, tested with only FIRST-PARTY fake JPEG/JSON.

No H Company original third-party image or task text is shipped to PUBLIC GitHub.
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from h57.import_train_multidomain import (REV,SCHEMA as SOURCE_SCHEMA,
                                           verify_pair,audit,sha,canonical)
from h57 import native_multidomain_sft as bridge
from training.scripts.train_adapter import (H57_DRAG_SCHEMA,
                                          load_authoring_records)


def fake_jpeg():
    b=BytesIO()
    Image.new("RGB",(1200,850),(48,86,122)).save(b,"JPEG",quality=85)
    return b.getvalue()

def make_fixture(root):
    root=Path(root)
    image=fake_jpeg()
    contents=[];source_entries=[];source_rows=[]
    by_domain={}
    for domain in ("sheet","slide_resize","text_highlight"):
        by_domain[domain]=dict(accepted_source_images=2,native_drag_examples=4,
                              ordered_true=0,ordered_false=0)
        (root/"images"/domain).mkdir(parents=True)
        (root/"tasks"/domain).mkdir(parents=True)
        for i in range(2):
            key=f"22222222-2222-2222-2222-{len(source_entries):012d}"
            tasks=[]
            for j in range(2):
                ordered=domain=="slide_resize" or (domain=="sheet" and j==0)
                tasks.append(dict(image_id=key,domain=domain,ordered=ordered,
                    intent=f"Move a synthetic {domain} draggable control {i} {j}",
                    start_bbox=[100,300,120,322],
                    end_bbox=[220+j*15,300,240+j*15,322],
                    subtype="original_unit_fixture_not_Hcompany",metadata={"synthetic":True}))
                by_domain[domain]["ordered_true" if ordered else "ordered_false"]+=1
            raw=canonical(tasks)
            (root/"images"/domain/(key+".jpg")).write_bytes(image)
            (root/"tasks"/domain/(key+".json")).write_bytes(raw)
            source_rows.extend(verify_pair(domain,key,image,raw))
            source_entries.append(dict(domain=domain,image_id=key,image_sha256=sha(image),
                task_sha256=sha(raw),original_image_source_offset=1000+len(source_entries),
                original_image_size=len(image),
                original_task_source_offset=2000+len(source_entries),
                original_task_size=len(raw),task_count=2))
    assert len(source_rows)==12
    blob=b"".join(json.dumps(r,sort_keys=True,ensure_ascii=False,
                   separators=(",",":")).encode("utf8")+b"\n" for r in source_rows)
    (root/"train_drag_multidomain.jsonl").write_bytes(blob)
    manifest=dict(schema=SOURCE_SCHEMA,original_revision=REV,only_original_train=True,
                  benchmark_eval_or_test_used=False,model_weights_trained=False,
                  original_paired_image_task_members=source_entries,
                  original_distinct_train_images=6,original_train_drag_actions=12,
                  ordered_true=6,ordered_false=6,domains=by_domain,
                  training_label_sha256=sha(blob))
    (root/"manifest.json").write_bytes(canonical(manifest))
    fake_receipt=root/"fake-public-receipt.json"
    public=dict(pinned_source_revision=REV,
       original_train_jsonl_sha256=sha(blob),
       original_source_manifest_sha256=sha((root/"manifest.json").read_bytes()),
       kaggle_private_dataset=bridge.PRIVATE_KAGGLE_DATASET,
       kaggle_private_provider_confirmed=True,
       private_train_original_labels_remote_verified=True,
       third_party_raw_images_committed_to_public_github=False,
       source_image_task_sha256=[{k:e[k] for k in ("domain","image_id",
          "image_sha256","task_sha256","original_image_source_offset",
          "original_task_source_offset","task_count")} for e in source_entries])
    fake_receipt.write_bytes(canonical(public))
    return fake_receipt,sha(blob)

class H57OriginalMultidomainTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.receipt,self.original_sha=make_fixture(self.root)
        for name,value in (
            ("SOURCE_RECEIPT",self.receipt),("ORIGINAL_TRAIN_SHA",self.original_sha),
            ("ORIGINAL_SCREENS",6),("ORIGINAL_ROWS",12),
            ("ORDERED_TRUE",6),("ORDERED_FALSE",6)):
            patched=patch.object(bridge,name,value)
            patched.start();self.addCleanup(patched.stop)

    def test_native_real_ordered_false_and_true_are_not_coerced(self):
        exported=bridge.export(self.root)
        self.assertEqual(exported["genuine_native_drag_rows"],12)
        manifest=json.loads((self.root/bridge.MANIFEST_FILE).read_text("utf8"))
        records=load_authoring_records(self.root/bridge.ROWS_FILE,manifest)
        self.assertEqual(len(records),12)
        self.assertEqual(len({r["family"] for r in records}),3)
        self.assertEqual(sum(json.loads(r["completion"])["ordered"] for r in records),6)
        self.assertEqual(sum(not json.loads(r["completion"])["ordered"] for r in records),6)
        false_rows=[r for r in records if not json.loads(r["completion"])["ordered"]]
        self.assertTrue(all("ordered=false" in r["prompt"] for r in false_rows))
        self.assertTrue(all("direction-insensitive" in r["prompt"] for r in false_rows))
        self.assertTrue(all(r["image_path"].endswith(".jpg") for r in records))
        self.assertFalse(exported["model_training_executed"])
        self.assertEqual(H57_DRAG_SCHEMA,bridge.SCHEMA)

    def test_create_only_and_original_bytes_preserved(self):
        images={str(p):sha(p.read_bytes()) for p in (self.root/"images").rglob("*.jpg")}
        bridge.export(self.root)
        with self.assertRaises(FileExistsError):bridge.export(self.root)
        self.assertEqual(images,{str(p):sha(p.read_bytes()) for p in
                                 (self.root/"images").rglob("*.jpg")})

    def test_source_task_order_mutation_fails_closed(self):
        bridge.export(self.root)
        task=next((self.root/"tasks").rglob("*.json"))
        data=json.loads(task.read_text("utf8"))
        data[0]["ordered"]=not data[0]["ordered"]
        task.write_bytes(canonical(data))
        meta=json.loads((self.root/bridge.MANIFEST_FILE).read_text("utf8"))
        with self.assertRaises(ValueError):
            bridge.load_export(self.root/bridge.ROWS_FILE,meta)

    def test_derived_completion_tamper_even_with_consistent_hash_fails(self):
        bridge.export(self.root)
        meta=json.loads((self.root/bridge.MANIFEST_FILE).read_text("utf8"))
        path=self.root/bridge.ROWS_FILE
        items=[json.loads(s) for s in path.read_text("utf8").splitlines()]
        c=json.loads(items[0]["completion"])
        c["ordered"]=not c["ordered"]
        items[0]["completion"]=json.dumps(c)
        path.write_bytes(b"".join(json.dumps(r,ensure_ascii=False,
              sort_keys=True,separators=(",",":")).encode()+b"\n" for r in items))
        meta["dataset_sha256"]=sha(path.read_bytes())
        with self.assertRaises(ValueError):bridge.load_export(path,meta)

    def test_public_receipt_or_remote_privacy_tamper_rejected(self):
        data=json.loads(self.receipt.read_text("utf8"))
        data["kaggle_private_provider_confirmed"]=False
        self.receipt.write_bytes(canonical(data))
        with self.assertRaises(ValueError):bridge.verify_source(self.root)

    def test_fake_missing_source_jpeg_rejected(self):
        path=next((self.root/"images").rglob("*.jpg"))
        path.unlink()
        with self.assertRaises(FileNotFoundError):audit(self.root)

    def test_out_of_bounds_and_invalid_original_order_flag_rejected(self):
        image=fake_jpeg();id="33333333-3333-3333-3333-333333333333"
        task=dict(image_id=id,domain="sheet",ordered=False,intent="Move fictional cell",
                  start_bbox=[100,200,110,210],end_bbox=[350,300,360,310])
        rows=verify_pair("sheet",id,image,canonical([task]))
        self.assertFalse(rows[0]["completion"]["ordered"])
        task["ordered"]="false"
        with self.assertRaises(ValueError):verify_pair("sheet",id,image,canonical([task]))
        task["ordered"]=False;task["end_bbox"]=[1199,1,1210,12]
        with self.assertRaises(ValueError):verify_pair("sheet",id,image,canonical([task]))

    def test_only_real_source_disjoint_mixed_baseline_can_pass_schema(self):
        invalid=dict(evidence="mock",task_kind="mixed_ordered_unordered_drag",
            held_out_source="original_source_disjoint_multidomain_drag_not_training",
            held_out_dataset_sha256="1"*64,evaluated_tasks=24)
        with self.assertRaises(ValueError):
            bridge.validate_mixed_drag_baseline(invalid)
        invalid["evidence"]="real_inference"
        invalid["evaluated_tasks"]=23
        with self.assertRaises(ValueError):bridge.validate_mixed_drag_baseline(invalid)
        invalid["evaluated_tasks"]=24
        # This is SCHEMA validation only, NOT provider-attested performance.
        self.assertTrue(bridge.validate_mixed_drag_baseline(invalid))
if __name__=="__main__":
    unittest.main()
