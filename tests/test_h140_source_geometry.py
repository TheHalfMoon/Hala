"""CPU-only H140 original source and model geometry-collapse guards."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from h140.qualification import (COUNTS,SCHEMA,admit,collapse_gate,digest,
                                 summarize_neural_position_diversity)


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

    def test_quantized_diversity_refuses_subpixel_jitter_and_five_in_one_bin(self):
        # All 16 predictions are numerically unique but sit in ONE 24px cell.
        jitter=[[101+i/100,201+i/100] for i in range(16)]
        report=collapse_gate(jitter)
        self.assertEqual(report["unique_points"],16)
        self.assertEqual(report["unique_position_bins"],1)
        self.assertFalse(report["passed"])
        # Eight spatial bins alone are insufficient if >25% are one location.
        five=[[101+i/100,201+i/100] for i in range(5)]
        dispersed=[[12+25*i,400] for i in range(11)]
        report=collapse_gate(five+dispersed)
        self.assertGreaterEqual(report["unique_position_bins"],8)
        self.assertGreater(report["largest_position_bin_count"],4)
        self.assertFalse(report["passed"])

    def test_eight_position_bins_with_two_per_bin_still_pass(self):
        points=[[24+27*cell+0.1*j,120] for cell in range(8) for j in range(2)]
        report=collapse_gate(points)
        self.assertEqual(report["unique_points"],16)
        self.assertEqual(report["unique_position_bins"],8)
        self.assertEqual(report["largest_position_bin_count"],2)
        self.assertTrue(report["passed"])

    def test_abstention_and_malformed_coordinates_fail_closed(self):
        diverse=[[30+i*25,100+i*16] for i in range(16)]
        with_abstention=[None]+diverse[1:]
        summary=summarize_neural_position_diversity(
            with_abstention,allow_abstain=True)
        self.assertEqual(summary["valid_points"],15)
        self.assertFalse(summary["passed"])
        with self.assertRaises(ValueError):
            summarize_neural_position_diversity(with_abstention)
        for bad in ([float("nan"),100],[float("inf"),100],
                    [True,100],[-1,100],[960,100]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                summarize_neural_position_diversity([bad]+diverse[1:])


if __name__=="__main__":
    unittest.main()
