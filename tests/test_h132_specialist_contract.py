"""Frozen GUI-grounding specialist source/response tests, synthetic only."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from h130.browser_env import TASKS
from h131.contract import sha
from h132.grounder_contract import (
    MODEL_ID,MODEL_SHA,FORMAT, native_point, instruction_for,verify_predictions)

class H132Tests(unittest.TestCase):
    def test_point_parser_rejects_ambiguous_or_invalid(self):
        self.assertEqual(native_point('{"point_2d": [500, 500]}'),[480,360])
        self.assertEqual(native_point('{"point_2d": [1000, 1000]}'),[959,719])
        self.assertEqual(native_point('{"point_2d": [0, 0]}'),[0,0])
        for example in ('[500,500]','{"point_2d": [-1, 8]}',
                        '{"point_2d": [false, 500]}','{"point_2d": ["500", 500]}',
                        '{"point_2d": [1001, 0]}','{"point_2d":[NaN,500]}',
                        '{"point_2d":[100,200],"hidden_target":1}'):
            self.assertIsNone(native_point(example))

    def test_role_target_is_not_end_reused_for_start(self):
        instruction="Drag the blue slider handle to the green destination."
        a=instruction_for(instruction,"start")
        b=instruction_for(instruction,"end")
        self.assertIn("blue HANDLE",a)
        self.assertIn("green destination",b)
        self.assertNotEqual(a,b)
        self.assertIn('{"point_2d": [x, y]}',a)
        with self.assertRaises(ValueError):
            instruction_for(instruction,"anything")

    def test_private_response_sha_and_native_pair_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cases=[]
            for name in TASKS:
                file=root/(name+".png")
                Image.new("RGB",(960,720)).save(file)
                cases.append({"task":name,"image":file.name,
                              "image_sha256":sha(file.read_bytes()),
                              "instruction":"Drag the blue item to target."})
            manifest={"schema":"hala.h131.private-browser-screenshots.v1",
                      "phase":"initial","width":960,"height":720,"cases":cases}
            (root/"manifest.json").write_text(json.dumps(manifest),"utf8")
            doc={"schema":FORMAT,"model_id":MODEL_ID,"model_sha":MODEL_SHA,
                 "source_manifest_sha256":sha(json.dumps(manifest,sort_keys=True,
                        separators=(",",":")).encode("utf8")),
                 "training_steps":0,"browser_oracle_used":False,
                 "cases":[{"task":c["task"],"image_sha256":c["image_sha256"],
                           "raw_start":'{"point_2d":[500,500]}',"start":[480,360],
                           "raw_end":'{"point_2d":[600,600]}',"end":[576,432]}
                          for c in cases]}
            self.assertEqual(len(verify_predictions(root,doc)),4)
            doc["cases"][0]["start"]=[9,9]
            with self.assertRaises(ValueError):
                verify_predictions(root,doc)
            doc["cases"][0]["start"]=[480,360]
            doc["cases"][0]["raw_start"]="invalid"
            doc["cases"][0]["start"]=None
            self.assertEqual(len(verify_predictions(root,doc)),4)
if __name__=="__main__":
    unittest.main()
