"""Kaggle automatically unpacks dataset ZIPs; frozen admission handles both safely."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile,ZIP_DEFLATED

import h139.blind_gpu as gpu
from h139.blind_export import SCHEMA
from h139.registry import sha,UPSTREAM


class H139BlindMountCompatibilityTests(unittest.TestCase):
    def test_raw_zip_and_auto_unpacked_same_frozen_cases(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            config=root/"frozen-adapter"
            config.mkdir()
            adapter=b"private-model-binary"
            (config/"adapter_model.safetensors").write_bytes(adapter)
            (config/"adapter_config.json").write_text('{"peft_type":"LORA"}',"utf8")
            rows=[]
            contents={}
            for i in range(8):
                name=f"case-{i:02d}.png"
                pixels=b"original-png-bytes-"+bytes([i])
                contents[name]=pixels
                rows.append({"id":f"case-{i}","app":"ood-board",
                    "app_git_sha":UPSTREAM["ood-board"]["git_sha"],
                    "lang":"ar" if i%2 else "en",
                    "instruction":f"Move card {i} to another column",
                    "image":name,"image_sha256":sha(pixels)})
            document={"schema":SCHEMA,"private":True,"source_role":"blind_test",
                "training_eligible":False,"original_manifest_sha256":"a"*64,
                "source_app_sha":UPSTREAM["ood-board"]["git_sha"],
                "case_count":8,"cases":rows}
            raw=(json.dumps(document,sort_keys=True)+"\n").encode()
            contents["manifest.json"]=raw
            zipfile=root/"blind8-redacted-eval-only.zip"
            with ZipFile(zipfile,"w",compression=ZIP_DEFLATED) as z:
                for filename,pixels in contents.items():
                    z.writestr(filename,pixels)
            zip_sha=sha(zipfile.read_bytes())
            with patch.object(gpu,"ADAPTER_SHA",sha(adapter)), \
                 patch.object(gpu,"ARCHIVE_SHA",zip_sha), \
                 patch.object(gpu,"REDACTED_MANIFEST_SHA",sha(raw)), \
                 patch.object(gpu,"OWNER_MANIFEST_SHA","a"*64):
                payload,doc,folder=gpu.admit(root)
                self.assertTrue(payload.is_file())
                self.assertEqual(len(doc["cases"]),8)
                zipfile.unlink()
                unpacked=root/"kaggle-redacted-dataset"
                unpacked.mkdir()
                for filename,pixels in contents.items():
                    (unpacked/filename).write_bytes(pixels)
                payload,doc,folder=gpu.admit(root)
                self.assertTrue(payload.is_dir())
                self.assertEqual(len(doc["cases"]),8)
                (unpacked/"case-00.png").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    gpu.admit(root)
                (unpacked/"case-00.png").write_bytes(contents["case-00.png"])
                (root/"old-train64").mkdir()
                with self.assertRaises(ValueError):
                    gpu.admit(root)


if __name__=="__main__":
    unittest.main()
