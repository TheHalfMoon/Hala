"""CPU-only H140 original source and model geometry-collapse guards."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from h140.qualification import COUNTS,SCHEMA,admit,collapse_gate,digest


class H140PositionDiversityTests(unittest.TestCase):
    def _source(self,root,split,app_counts):
        folder=root/split
        folder.mkdir()
        sources={}
        cases=[]
        index=0
        for app,amount in app_counts.items():
            sha=f"{len(sources)+17:040x}"
            sources[app]={"license":"MIT","license_file_verified":True,
                "tracked_source_modified":False,"upstream_sha":sha,
                "application_is_complete":True}
            for n in range(amount):
                filename=f"{app}-{n:03d}.png"
                pixels=(b"\x89PNG\r\n\x1a\n"+b"\x00\x00\x00\x0dIHDR"+
                    struct.pack(">II",960,720)+f"H140 {split} {app} {n}".encode())
                (folder/filename).write_bytes(pixels)
                cases.append({"image":filename,"image_sha256":digest(pixels),
                    "app":app,"app_git_sha":sha,"split":split,
                    "language":"en" if n%2==0 else "ar",
                    "instruction":f"Drag task card {app} {n} to the next column",
                    "reference_drag_persisted":True,
                    "model_input_contains_oracle_coordinates":False,
                    "start_bbox":{"x":40+(n%16)*32,"y":240,"width":16,"height":20},
                    "end_bbox":{"x":60+(n%16)*31,"y":370,"width":36,"height":25}})
                index+=1
        assert len(cases)==COUNTS[split]
        (folder/"manifest.json").write_text(json.dumps({
            "schema":SCHEMA,"private":True,"split":split,
            "training_eligible":split!="blind_test",
            "app_sources":sources,"cases":cases}),encoding="utf8")
        return folder

    def test_source_split_app_disjointness_and_authentic_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            dev=self._source(root,"dev",{"training_a":22,"training_b":22,"training_c":20})
            blind=self._source(root,"blind_test",{"unseen_d":16,"unseen_e":16})
            historic=root/"historic.json"
            historic.write_text(json.dumps({"cases":[{"image_sha256":"f"*64}]}))
            report=admit(train,dev,blind,[historic])
            self.assertEqual(report["distinct_images"],352)
            self.assertEqual(report["status"],"QUALIFIED_SOURCE_ONLY_NOT_TRAINED")
            self.assertEqual(report["real_neural_calls"],0)
            with self.assertRaises(ValueError):
                admit(train,dev,blind,[])
            document=json.loads((train/"manifest.json").read_text("utf8"))
            for row in document["cases"]:
                row["end_bbox"]={"x":80,"y":370,"width":20,"height":25}
            (train/"manifest.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,"fixed-coordinate collapse"):
                admit(train,dev,blind,[historic])

    def test_neural_collapse_gate_does_not_accept_fixed_points(self):
        repeated=[[240,340] for _ in range(16)]
        collapse=collapse_gate(repeated)
        self.assertFalse(collapse["passed"])
        self.assertEqual(collapse["unique_points"],1)
        diverse=[[30+i*25,100+i*16] for i in range(16)]
        improved=collapse_gate(diverse)
        self.assertTrue(improved["passed"])
        self.assertEqual(improved["modal_share"],1/16)
        with self.assertRaises(ValueError):
            collapse_gate([[30,40]]*15)
        with self.assertRaises(ValueError):
            collapse_gate([[None,None]]*16)
        with self.assertRaises(ValueError):
            collapse_gate([[960,720]]*16)


if __name__=="__main__":
    unittest.main()
