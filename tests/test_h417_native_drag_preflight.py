"""H417 source-bound 12k original native drag TRAIN preflight; no GPU or raw source in GitHub."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h417.kaggle import h417_native_drag_lora as m


class H417NativeDragPreflight(unittest.TestCase):
    def test_source_frozen_sizes_and_gpu_private_dataset_metadata(self):
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/"h417/kaggle/kernel-metadata.json").read_text())
        self.assertTrue(data["is_private"])
        self.assertTrue(data["enable_gpu"])
        self.assertEqual(data["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(data["kernel_sources"],[])
        self.assertEqual(data["dataset_sources"],[
            "abdulazizshehri/hala-h415-dragon-original-4domain-private",
            "abdulazizshehri/hala-h415-frozen-10k-native-drag-private",
            "abdulazizshehri/hala-h415-original-bilingual-drag-2k-private",
        ])
        self.assertEqual((m.N_SOURCE_DRAGON,m.N_SOURCE_FIRSTPARTY,
                          m.TRAIN_UNIQUE,m.EXPECTED_PASSES,m.STEPS),
                         (10000,2000,11952,14880,3720))
        self.assertEqual(m.STEPS*m.GRAD_ACCUM,m.EXPECTED_PASSES)
        self.assertEqual(m.AR_REPEAT,3)

    def test_original_h60_raw_source_schema_requires_dict_not_derived_json_string(self):
        case={
            "id":"h60-sheet-abcd-t000","split":"train","language":"en",
            "image_path":"images/sheet/abcd.jpg","image_sha256":"a"*64,
            "intent":"Choose rows 1 through 15","ordered":False,
            "completion":{"type":"drag","start_point":[203,203],
                "end_point":[203,659],"ordered":False,
                "coordinate_space":"source_image_pixels"}
        }
        row=m.case_from_dragon(case)
        self.assertEqual(row["start"],[203,203])
        self.assertEqual(row["end"],[203,659])
        self.assertFalse(row["ordered"])
        self.assertEqual(row["body"],case["intent"])
        self.assertEqual(row["domain"],"sheet")
        tampered=dict(case,completion={**case["completion"],"ordered":True})
        with self.assertRaises(ValueError):m.case_from_dragon(tampered)

    def test_normalized_drag_parser_and_directionality(self):
        good={"type":"drag","start_point":[220,345],
              "end_point":[700,610],"ordered":True,
              "coordinate_space":"normalized_1000"}
        raw=json.dumps(good)
        self.assertEqual(m.parse_drag(raw),good)
        self.assertTrue(m.score_predicted(raw,good)["correct"])
        swapped=dict(good,start_point=good["end_point"],end_point=good["start_point"])
        self.assertFalse(m.score_predicted(json.dumps(swapped),good)["correct"])
        unordered=dict(good,ordered=False)
        swapped_unordered=dict(unordered,start_point=unordered["end_point"],
                               end_point=unordered["start_point"])
        self.assertTrue(m.score_predicted(json.dumps(swapped_unordered),unordered)["correct"])
        self.assertFalse(m.score_predicted(json.dumps(swapped),unordered)["correct"])
        for bad in (
            '{"type":"drag","start_point":[NaN,12]}',
            '{"type":"drag","start_point":[1001,3]}',
            '{"type":"drag","start_point":[true,4]}',
            '{"type":"drag","start_point":[1,2],"start_point":[3,4]}',
            '{"point_2d":[10,20]}',
        ):
            self.assertIsNone(m.parse_drag(bad))

    def test_real_firstparty_source_row_coordinates(self):
        row=dict(id="h61-own-001",split="train",language="ar",kind="slider",
                 ordered=True,image_path="png/h61-own-001.png",
                 image_sha256="b"*64,instruction="اسحب العنصر إلى المكان المناسب",
                 start_bbox=[100,200,160,230],end_bbox=[400,200,460,230])
        case=m.case_from_firstparty(row)
        self.assertEqual(case["start"],[130,215])
        self.assertEqual(case["end"],[430,215])
        self.assertEqual(case["language"],"ar")
        self.assertEqual(case["domain"],"slider")
        self.assertIn("normalized_1000",m.prompt(case))
        gold=m.target(case,1024,640)
        self.assertEqual(gold["ordered"],True)
        self.assertEqual(gold["coordinate_space"],"normalized_1000")
        self.assertTrue(m.valid_coords(gold["start_point"]))

    def test_source_digest_pins_original_local_evidence(self):
        root=Path(__file__).resolve().parents[1]
        note=(root/"h415/SOURCE-STATUS.md").read_text()
        self.assertIn(m.DATASET_SHA,note)
        self.assertIn(m.FIRSTPARTY_LABEL_SHA,note)
        self.assertEqual(len(m.SOURCE_SHAS),4)

    def test_optional_owner_windows_real_private_source_partition(self):
        root=os.environ.get("H417_OWNER_PRIVATE_SOURCE_ROOT")
        if not root:
            self.skipTest("Owner Windows original source directory not provided; not in repo/CI")
        root=Path(root)
        names=[name for name,_ in m.SOURCE_SHAS.values()]
        names.extend([m.CURRIC_FILE,m.BIL_FILE])
        private={name:root/name for name in names}
        if not all(p.is_file() for p in private.values()):
            raise AssertionError("Owner real original source file missing")
        with patch.object(m,"only_input_file",side_effect=lambda name: private[name]):
            zips,raw,curriculum,firstparty=m.get_research_sources()
            try:
                train,dev,order=m.partition(raw,curriculum,firstparty)
                self.assertEqual((len(train),len(dev),len(order)),(11952,144,14880))
                self.assertEqual(sum(x["language"]=="ar" for x in train),976)
                self.assertEqual(sum(x["language"]=="ar" for x in dev),24)
                self.assertEqual(len({x["image_sha256"] for x in train}),11952)
                self.assertEqual(len({x["image_sha256"] for x in dev}),144)
                self.assertFalse({x["image_sha256"] for x in train}&
                                 {x["image_sha256"] for x in dev})
                repairs=[row for row in train if row.get("source_repair") is True]
                self.assertEqual(len(repairs),4)
                self.assertTrue(all(r["domain"]=="slider" for r in repairs))
                # This checks that actual JPEG bytes for every repair exist
                # in the attached original source, not just in label JSON.
                self.assertEqual(len({r["image_sha256"] for r in repairs}),4)
                for case in repairs+[next(r for r in train if r["kind"]=="firstparty")]:
                    image,width,height=m.source_image(case,zips)
                    self.assertGreater(width,0)
                    self.assertGreater(height,0)
            finally:
                for z in zips.values():z.close()


if __name__=="__main__":
    unittest.main()
