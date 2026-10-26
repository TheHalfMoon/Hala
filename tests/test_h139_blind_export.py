"""H139 blind test transport removes all oracle coordinates by construction."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from h139.blind_export import export_private,SCHEMA
from h139.registry import UPSTREAM,sha,SCHEMA as SOURCE_SCHEMA


class H139BlindRedactionTests(unittest.TestCase):
    def test_private_balanced_export_is_label_free_and_create_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"private-h139"
            folder=root/"blind8"
            folder.mkdir(parents=True)
            rows=[]
            for i in range(8):
                name=f"ood-{i}.png"
                pixels=b"test-image-"+bytes([i])
                (folder/name).write_bytes(pixels)
                rows.append({
                    "id":f"blind-{i}","app":"ood-board",
                    "app_git_sha":UPSTREAM["ood-board"]["git_sha"],
                    "role":"blind_test","lang":"ar" if i%2 else "en",
                    "instruction":f"Move card {i} into In Progress",
                    "image":name,"image_sha256":sha(pixels),
                    "start_bbox":{"x":1,"y":2,"width":3,"height":4},
                    "end_bbox":{"x":5,"y":6,"width":7,"height":8},
                    "seed_index":i,"source_task_id":f"private-id-{i}",
                })
            (folder/"manifest.json").write_text(json.dumps({
                "schema":SOURCE_SCHEMA,"private":True,"split":"blind_test",
                "cases":rows,
            }),encoding="utf8")
            output=root/"redacted.zip"
            with patch("h139.blind_export.audit_sources",return_value={}):
                result=export_private(root,output)
                self.assertFalse(result["truth_labels_exported"])
                self.assertEqual(result["images"],8)
                with ZipFile(output) as z:
                    public=json.loads(z.read("manifest.json"))
                    self.assertEqual(public["schema"],SCHEMA)
                    self.assertEqual(public["case_count"],8)
                    self.assertEqual(len(z.namelist()),9)
                    self.assertNotIn(b"start_bbox",z.read("manifest.json"))
                    self.assertNotIn(b"end_bbox",z.read("manifest.json"))
                    self.assertNotIn(b"source_task_id",z.read("manifest.json"))
                with self.assertRaises(ValueError):
                    export_private(root,output)
            with patch("h139.blind_export.audit_sources",return_value={}):
                with self.assertRaises(ValueError):
                    export_private(root,Path(temp)/"outside.zip")


if __name__=="__main__":
    unittest.main()
