"""H131 strict private source + native point tests, no live model needed."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from h130.browser_env import TASKS
from h131.contract import (SCHEMA,MODEL_SCHEMA,parse_showui,sha,
                           verify_manifest,verify_model_predictions)
from h131.gpu_infer import locate_private_input
class H131ContractTests(unittest.TestCase):
    def test_showui_native_coords_for_960x720(self):
        self.assertEqual(parse_showui("[0.5, 0.5]"),(480,360))
        self.assertEqual(parse_showui("[1, 1]"),(959,719))
        for raw in ("(0.5, 0.5)","[2,0]","[0, nan]","something",
                    "[True,0.5]","[-1, 0]"):
            self.assertIsNone(parse_showui(raw))

    def test_private_images_and_neural_replay_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            rows=[]
            for task in TASKS:
                file=path/(task+".png")
                Image.new("RGB",(960,720),(25,50,75)).save(file)
                rows.append({"task":task,"image":task+".png",
                    "image_sha256":sha(file.read_bytes()),"instruction":"Drag this card to destination."})
            manifest={"schema":SCHEMA,"phase":"initial","width":960,"height":720,
                      "cases":rows}
            (path/"manifest.json").write_text(json.dumps(manifest),"utf8")
            verified=verify_manifest(path,expect_phase="initial")
            discovered, admitted=locate_private_input("initial",root=path)
            self.assertEqual(discovered,path)
            self.assertEqual(admitted["schema"],SCHEMA)
            predictions={"schema":MODEL_SCHEMA,"phase":"initial",
                "input_manifest_sha256":sha(json.dumps(verified,sort_keys=True,separators=(",",":")).encode()),
                "model_id":"showlab/ShowUI-2B",
                "model_revision":"cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60",
                "cases":[{"task":x["task"],"image_sha256":x["image_sha256"],
                          "raw_start":"[0.5, 0.5]","start":[480,360],
                          "raw_end":"[0.5, 0.5]","end":[480,360]} for x in rows]}
            self.assertEqual(len(verify_model_predictions(predictions,verified)),4)
            predictions["cases"][0]["raw_start"]="[0.9, 0.9]"
            with self.assertRaises(ValueError):
                verify_model_predictions(predictions,verified)
            rows[0]["image_sha256"]="0"*64
            (path/"manifest.json").write_text(json.dumps(manifest),"utf8")
            with self.assertRaises(ValueError):
                verify_manifest(path,expect_phase="initial")
if __name__=="__main__":
    unittest.main()
