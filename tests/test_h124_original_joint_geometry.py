"""H124 genuine joint START+END+delta geometry, no oracle in prompt."""
import json,unittest
from h61.prompt_contract import drag_prompt,source_body
from h124.objective import cell,prompt,completion,parse,from_original
from h124.schedule import Schedule,ORIGINAL,TRAIN,HOLDOUT
from h113.dual_point import source_pixel_to_relative


def fixture():
    original={
      "id":"h124-original-source-1","split":"train","action_type":"drag",
      "intent":"Move the slider to the right.","ordered":True,
      "image_width":1000,"image_height":600,
      "image_sha256":"a"*64,"image_path":"slider/images/1.jpg",
      "start_bbox":[95,115,105,125],"end_bbox":[295,475,305,485],
      "completion":{"type":"drag","ordered":True,
                    "start_point":[100,120],"end_point":[300,480],
                    "coordinate_space":"source_image_pixels"}}
    sft={
      "id":original["id"],"split":"train",
      "image_sha256":original["image_sha256"],"image_path":original["image_path"],
      "family":"hcompany_dragon_h61_slider_train","language":"en",
      "prompt":drag_prompt(source_body(original["intent"],True),1000,600,True),
      "completion":json.dumps(original["completion"],sort_keys=True)}
    return sft,original


class H124JointGeometryTests(unittest.TestCase):
    def test_frozen_source_only_joint_START_END_and_delta(self):
        sft,original=fixture()
        output=from_original(sft,original)
        answer=json.loads(output["completion"])
        self.assertEqual(answer["start_point"],[100,200])
        self.assertEqual(answer["end_point"],[300,800])
        self.assertEqual(answer["delta"],[200,600])
        self.assertEqual(answer["start_cell"],[1,2])
        self.assertEqual(answer["end_cell"],[3,8])
        self.assertEqual(parse(output["completion"],1000,600),
                         ([100,120],[300,480]))
        self.assertNotIn("start_bbox",output["prompt"])
        self.assertNotIn("end_bbox",output["prompt"])
        self.assertNotIn("100,120",output["prompt"])
        self.assertNotIn("300,480",output["prompt"])

    def test_strict_two_cell_coordinate_and_delta_consistency(self):
        answer=json.loads(completion([100,200],[300,800]))
        answer["delta"]=[100,600]
        self.assertIsNone(parse(json.dumps(answer),1000,600))
        answer=json.loads(completion([100,200],[300,800]))
        answer["end_cell"]=[9,9]
        self.assertIsNone(parse(json.dumps(answer),1000,600))
        answer=json.loads(completion([100,200],[300,800]))
        answer["coordinate_space"]="source_image_pixels"
        self.assertIsNone(parse(json.dumps(answer),1000,600))
        with self.assertRaises(ValueError):cell([True,100])
        with self.assertRaises(ValueError):cell([-1,2])

    def test_strict_json_duplicate_fields_or_extra_unknown(self):
        raw=completion([250,350],[450,550])
        self.assertIsNone(parse(raw.replace('"delta":','"delta":[1,1],"delta":'),1000,600))
        data=json.loads(raw);data["bbox"]=[0,0,1,1]
        self.assertIsNone(parse(json.dumps(data),1000,600))

    def test_rejects_bad_source_or_eval_split(self):
        sft,original=fixture()
        with self.assertRaises(ValueError):
            from_original({**sft,"split":"eval"},original)
        with self.assertRaises(ValueError):
            from_original(sft,{**original,"split":"eval"})
        with self.assertRaises(ValueError):
            from_original({**sft,"image_sha256":"b"*64},original)

    def test_new_source_cohort_requires_all_original_1024(self):
        self.assertEqual(ORIGINAL,1024)
        self.assertEqual(TRAIN,896)
        self.assertEqual(HOLDOUT,128)
        self.assertEqual(TRAIN+HOLDOUT,ORIGINAL)
        with self.assertRaises(ValueError):prompt("",True)
        self.assertNotIn("original END",prompt("Move element to position",True))
