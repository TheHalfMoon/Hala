"""Integration test: real decodable source photos freeze into exactly 256/64.

Source synthetic PIL pixels are TEST FIXTURES ONLY; never real H140 data.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from h140.freeze_train_dev import SOURCE_RUNS,SOURCE_APPS,freeze
from h140.qualification import load_split,digest


class H140CompleteTrainDevFreezeTests(unittest.TestCase):
    def test_complete_320_exact_image_bilingual_source_freeze(self):
        with tempfile.TemporaryDirectory() as scratch:
            private=Path(scratch)/"private-h140";private.mkdir()
            index=0
            for split,apps in SOURCE_RUNS.items():
                for app,(name,count) in apps.items():
                    folder=private/name
                    folder.mkdir()
                    rows=[]
                    for i in range(count):
                        filename=f"original-{i:03d}.png"
                        # Fully decodable RGB PNG. Distinct bytes across all
                        # 320 training/development fixtures and applications.
                        color=(index%256,index//256,111)
                        raw=io.BytesIO()
                        Image.new("RGB",(960,720),color).save(raw,"PNG")
                        pixels=raw.getvalue()
                        (folder/filename).write_bytes(pixels)
                        rows.append({
                            "id":f"native-owner-{index}","app":app,
                            "app_git_sha":SOURCE_APPS[app]["sha"],
                            "split":"pilot_only_not_training",
                            "language":"en" if i%2==0 else "ar",
                            "instruction":f"Move native original task {index} to the next column",
                            "image":filename,"image_sha256":digest(pixels),
                            "start_bbox":{"x":40+(i%16)*27,"y":150,"width":22,"height":21},
                            "end_bbox":{"x":80+(i%16)*28,"y":340,"width":21,"height":19},
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_input_contains_oracle_coordinates":False})
                        index+=1
                    doc={"schema":SOURCE_APPS[app]["schema"],"private":True,
                         "training_eligible":False,"blind_eligible":False,
                         "status":"POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY",
                         "cases":rows}
                    (folder/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
            self.assertEqual(index,320)
            def fake_original(root,app):
                spec=SOURCE_APPS[app]
                return {"repository":spec["repo"],"upstream_sha":spec["sha"],
                        "license":"MIT","license_file_verified":True,
                        "tracked_source_modified":False,"application_is_complete":True}
            with patch("h140.freeze_train_dev.image_sha_exclusions",
                       return_value=({"f"*64},15,6)), \
                 patch("h140.freeze_train_dev.source_checkout",
                       side_effect=fake_original):
                receipt=freeze(private)
            self.assertEqual(receipt["status"],
                             "FROZEN_TRAIN_DEV_SOURCE_ONLY_BLIND_AND_NEURAL_PENDING")
            self.assertEqual(receipt["unique_new_train_dev_png_sha"],320)
            self.assertEqual(receipt["training_case_count"],256)
            self.assertEqual(receipt["development_case_count"],64)
            frozen=private/"frozen-source-v1"
            self.assertEqual(len(load_split(frozen/"train","train")["cases"]),256)
            self.assertEqual(len(load_split(frozen/"dev","dev")["cases"]),64)
            with self.assertRaisesRegex(ValueError,"create-only"):
                freeze(private)


if __name__=="__main__":
    unittest.main()
