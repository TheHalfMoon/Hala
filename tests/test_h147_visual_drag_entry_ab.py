"""H147 image-feedback controller only reads original screen pixels."""
import io
import unittest
from PIL import Image,ImageDraw
from collections import Counter
from h147.visual_drag_entry_ab import (
    changed_samples,case,OFFSET,CASES,OFFSETS,MIN_VISIBLE_CHANGED_SAMPLES
)
class H147ImageObservedNativeDropTests(unittest.TestCase):
    def test_new_task_and_preregistered_visual_scan(self):
        self.assertEqual(OFFSET,114000)
        self.assertEqual(CASES,16)
        self.assertGreater(len(OFFSETS),10)
        self.assertGreater(MIN_VISIBLE_CHANGED_SAMPLES,0)
        cases=[case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((r["language"],r["destination_status"])
                                 for r in cases),
                         {("en","doing"):4,("ar","doing"):4,
                          ("en","done"):4,("ar","done"):4})
        self.assertEqual(len({r["layout"] for r in cases}),4)
        with self.assertRaises(ValueError):case(102000)

    def test_target_pixel_feedback_detects_activated_region_not_other_column(self):
        white=Image.new("RGB",(960,720),"white")
        target=white.copy()
        ImageDraw.Draw(target).rectangle((430,215,545,289),fill="gray")
        other=white.copy()
        ImageDraw.Draw(other).rectangle((35,215,150,289),fill="gray")
        def jpeg_bytes(im):
            out=io.BytesIO()
            im.save(out,format="PNG")
            return out.getvalue()
        base,changed=jpeg_bytes(white),jpeg_bytes(target)
        self.assertEqual(changed_samples(base,base,[480,180]),0)
        self.assertGreater(changed_samples(base,changed,[480,180]),
                           MIN_VISIBLE_CHANGED_SAMPLES)
        self.assertEqual(changed_samples(base,jpeg_bytes(other),[480,180]),0)
if __name__=="__main__":
    unittest.main()
