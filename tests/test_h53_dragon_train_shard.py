"""H53 bounded authentic DragOn import: all fixtures are self-generated fake JPEGs.

No external raw screenshots committed to public Hala GitHub; original external
bytes, if imported, reside only in verified PRIVATE Kaggle.
"""
import copy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from h53.dragon_train_shard import (
    SCHEMA,SOURCE_FILE,SOURCE_REV,URL,verify_item,audit,canonical,sha
)
from training.artifacts import validate_action_completion


def fake_image():
    im=Image.new("RGB",(1280,800),(30,80,110))
    bio=BytesIO()
    im.save(bio,format="JPEG",quality=90)
    return bio.getvalue()


def fake_case(id,goal=0):
    return dict(intent=f"Move a fictional slider to {goal+1}",
        image_id=id,domain="slider",ordered=True,
        start_bbox=[90,300,105,320],
        end_bbox=[200+goal*40,300,215+goal*40,320],
        metadata={"source":"synthetic-unit-test-only"})


class DragOnShardTests(unittest.TestCase):
    def setUp(self):
        self.image=fake_image()
        self.id="00000000-0000-0000-0000-000000000001"
        self.raw=canonical([fake_case(self.id,i) for i in range(3)])

    def test_exact_real_schema_drag_coordinates_are_not_clicks(self):
        rows=verify_item(self.id,self.image,self.raw)
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]["completion"]["type"],"drag")
        self.assertEqual(rows[0]["completion"]["start_point"],[98,310])
        self.assertEqual(rows[0]["completion"]["end_point"],[208,310])
        self.assertTrue(rows[0]["ordered"])
        with self.assertRaises(ValueError):
            validate_action_completion(json.dumps(rows[0]["completion"]))

    def test_actual_image_id_and_domain_not_position(self):
        bad=json.loads(self.raw)
        bad[0]["image_id"]="99999999-9999-9999-9999-999999999999"
        with self.assertRaises(ValueError):verify_item(self.id,self.image,canonical(bad))
        bad=json.loads(self.raw)
        bad[0]["domain"]="eval"
        with self.assertRaises(ValueError):verify_item(self.id,self.image,canonical(bad))
        bad=json.loads(self.raw)
        bad[0]["ordered"]=False
        with self.assertRaises(ValueError):verify_item(self.id,self.image,canonical(bad))

    def test_reject_outside_original_image_bad_box(self):
        bad=json.loads(self.raw)
        bad[0]["end_bbox"]=[1280,100,1300,120]
        with self.assertRaises(ValueError):verify_item(self.id,self.image,canonical(bad))
        bad=json.loads(self.raw)
        bad[0]["start_bbox"]=[20.3,20,30,30]
        with self.assertRaises(ValueError):verify_item(self.id,self.image,canonical(bad))

    def test_unrecognized_or_truncated_screenshot_rejected(self):
        with self.assertRaises(ValueError):verify_item(self.id,self.image[:-10],self.raw)
        with self.assertRaises(ValueError):verify_item(self.id,b"NOT_JPEG",self.raw)

    def test_bounded_pinned_source_never_eval(self):
        self.assertEqual(SOURCE_FILE,"train/slider.tar")
        self.assertEqual(len(SOURCE_REV),40)
        self.assertIn("/train/slider.tar",URL)
        self.assertNotIn("/eval/",URL)

    def test_complete_source_only_fixture_round_trip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"images").mkdir()
            (root/"tasks").mkdir()
            rows=[]
            proofs=[]
            for i in range(5):
                key=f"00000000-0000-0000-0000-{i:012d}"
                raw=canonical([fake_case(key,j) for j in range(3)])
                (root/"images"/(key+".jpg")).write_bytes(self.image)
                (root/"tasks"/(key+".json")).write_bytes(raw)
                rows.extend(verify_item(key,self.image,raw))
                proofs.append(dict(image_id=key,image_sha256=sha(self.image),task_sha256=sha(raw),
                                   examples=3,source_image_range=[0,len(self.image)],
                                   source_task_range=[0,len(raw)]))
            lines=b"".join(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()+b"\n" for x in rows)
            (root/"train_drag.jsonl").write_bytes(lines)
            metadata=dict(schema=SCHEMA,source_revision=SOURCE_REV,source_file=SOURCE_FILE,
               source_train_only=True,source_eval_used=False,
               real_model_training_executed=False,remote_storage_verified=False,
               full_source_tar_sha256_independently_verified=False,
               jsonl_sha256=sha(lines),matched_source_entries=proofs,
               train_rows=15,unique_original_screenshots=5)
            (root/"manifest.json").write_bytes(canonical(metadata))
            self.assertEqual(audit(root)["original_train_drag_annotations"],15)
            modified=copy.deepcopy(rows)
            modified[0]["completion"]["end_point"]=[0,0]
            (root/"train_drag.jsonl").write_bytes(b"".join(
                json.dumps(x,sort_keys=True,ensure_ascii=False).encode()+b"\n"
                for x in modified))
            with self.assertRaises(ValueError):audit(root)

if __name__=="__main__":
    unittest.main()
