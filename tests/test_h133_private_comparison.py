"""H133 frozen unseen-source and two-model private prediction contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from h133.browser_env import CASES
from h133.source import validate,sha,SOURCE_SCHEMA
from h133.neural import FORMAT
from h133.owner_eval import verify_model
from h125.model_contracts import MODELS
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h130.browser_env import WIDTH,HEIGHT

class H133PrivateContractTests(unittest.TestCase):
    def setup_cases(self,path):
        cases=[]
        for index,task in enumerate(CASES):
            file=path/(task+".png")
            Image.new("RGB",(WIDTH,HEIGHT),(index*20,3,4)).save(file)
            cases.append({"task":task,"image":file.name,
                "image_sha256":sha(file.read_bytes()),"instruction":"Drag blue handle to green destination."})
        manifest={"schema":SOURCE_SCHEMA,"cases_count":8,
                  "width":WIDTH,"height":HEIGHT,"gold_in_model_inputs":False,
                  "cases":cases}
        (path/"manifest.json").write_text(json.dumps(manifest),"utf8")
        return manifest

    def test_refuse_previous_source_sha_and_mismatched_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            data=self.setup_cases(path)
            self.assertEqual(len(validate(path)["cases"]),8)
            with self.assertRaises(ValueError):
                validate(path,forbid_shas={data["cases"][0]["image_sha256"]})
            data["cases"][0]["task"]="old-slider"
            (path/"manifest.json").write_text(json.dumps(data),"utf8")
            with self.assertRaises(ValueError):
                validate(path)

    def test_two_frozen_model_outputs_same_private_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            doc=self.setup_cases(path)
            src_sha=sha(json.dumps(doc,sort_keys=True,separators=(",",":")).encode("utf8"))
            for arm in ("showui","specialist"):
                if arm=="showui":
                    id,rev=MODELS["showui_2b"].model_id,MODELS["showui_2b"].revision_sha
                    start_raw="[0.5, 0.5]"
                else:
                    id,rev=MODEL_ID,MODEL_SHA
                    start_raw='{"point_2d":[500,500]}'
                predicted={"schema":FORMAT,"model_arm":arm,
                    "model_id":id,"model_sha":rev,
                    "source_manifest_sha256":src_sha,
                    "real_neural_calls":16,"training_steps":0,
                    "oracle_coordinates_in_model_input":False,
                    "cases":[{
                      "task":item["task"],"image_sha256":item["image_sha256"],
                      "raw_start":start_raw,"start":[480,360],
                      "raw_end":start_raw,"end":[480,360]}
                        for item in doc["cases"]]}
                self.assertEqual(len(verify_model(self.json_temp(path,arm,predicted),doc,arm)),8)
                predicted["model_sha"]="0"*40
                with self.assertRaises(ValueError):
                    verify_model(self.json_temp(path,arm,predicted),doc,arm)

    def json_temp(self,path,arm,data):
        file=path/("output-"+arm+".json")
        file.write_text(json.dumps(data),"utf8")
        return file

if __name__=="__main__":unittest.main()
