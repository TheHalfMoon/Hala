"""H418 original large-data contract and native two-point action tests."""
import unittest
from h418 import raw_original_corpus as h
from h418 import curate as c

class OriginalH60NativeTests(unittest.TestCase):
    def test_source_original_image_and_action_contract(self):
        self.assertEqual(h.SOURCE_IMAGE_UNIQUE,47037)
        self.assertEqual(h.SOURCE_ACTIONS,500554)
        self.assertEqual(h.TARGET,30000)
        self.assertGreater(h.MIN_AVAILABLE_NEW,h.TARGET)
        self.assertEqual(len(c.ALL_SHAS),9)

    def test_source_native_drag_is_not_click_or_eval(self):
        digest="f"*64
        name="h60-dragon-train-sheet-0000.hala.bin"
        row={"id":"h60-sheet-original-t000","split":"train",
             "domain":"sheet","image_path":"images/sheet/original.jpg",
             "image_sha256":digest,"image_width":1024,"image_height":640,
             "language":"en","intent":"Drag the unique source element",
             "ordered":True,"source_task_sha256":digest,
             "completion":{"type":"drag","start_point":[20,30],
                           "end_point":[900,500],"ordered":True,
                           "coordinate_space":"source_image_pixels"}}
        result=h.original_record(row,"sheet",name,digest)
        self.assertEqual(result["native_drag_completion"]["type"],"drag")
        self.assertEqual(result["image_sha256"],digest)
        self.assertTrue(result["ordered"])
        bad=dict(row,split="test")
        with self.assertRaises(ValueError):
            h.original_record(bad,"sheet",name,digest)
        bad=dict(row,completion={**row["completion"],"ordered":False})
        with self.assertRaises(ValueError):
            h.original_record(bad,"sheet",name,digest)
        bad=dict(row,completion={**row["completion"],"end_point":[2000,500]})
        with self.assertRaises(ValueError):
            h.original_record(bad,"sheet",name,digest)

if __name__=="__main__":
    unittest.main()
