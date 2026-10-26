"""Unseen original GUI model inputs must NEVER include owner DOM gold."""
import unittest
from h140.export_blind_redacted import (
    ALLOWED_FIELDS,OWNER_BLIND_SHA,SCHEMA,redacted_cases
)
from h140.blind_source_preflight import SOURCES


class H140BlindRedactionContracts(unittest.TestCase):
    @staticmethod
    def source():
        cases=[]
        for app,policy in SOURCES.items():
            for i in range(16):
                cases.append({
                    "id":f"{app}-{i}","app":app,"app_git_sha":policy["sha"],
                    "language":"ar" if i%2 else "en",
                    "instruction":f"Move unseen original card {app} {i}",
                    "image":f"{app}-{i:02d}.png",
                    "image_sha256":f"{len(cases)+1:064x}",
                    "start_bbox":{"x":20,"y":30,"width":20,"height":20},
                    "end_bbox":{"x":450,"y":400,"width":25,"height":25},
                    "reference_drag_persisted":True,
                    "source_task_id":"GOLD NEVER IN MODEL INPUT",
                })
        return {"split":"blind_test","training_eligible":False,
                "app_sources":{a:policy for a,policy in SOURCES.items()},
                "cases":cases}

    def test_keep_native_images_and_text_but_strip_every_oracle(self):
        doc=self.source()
        r=redacted_cases(doc)
        self.assertEqual(len(r),32)
        self.assertEqual(len(OWNER_BLIND_SHA),64)
        self.assertTrue(SCHEMA.startswith("hala.h140"))
        for entry in r:
            self.assertEqual(set(entry),ALLOWED_FIELDS)
            self.assertNotIn("start_bbox",entry)
            self.assertNotIn("end_bbox",entry)
            self.assertNotIn("source_task_id",entry)
            self.assertNotIn("reference_drag_persisted",entry)

    def test_duplicate_images_or_training_split_rejected(self):
        doc=self.source()
        doc["cases"][1]["image_sha256"]=doc["cases"][0]["image_sha256"]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            redacted_cases(doc)
        doc=self.source()
        doc["training_eligible"]=True
        with self.assertRaisesRegex(ValueError,"only frozen"):
            redacted_cases(doc)


if __name__=="__main__":
    unittest.main()
