"""H150 original screenshot-only visual drop-zone inference contracts."""
import io
import unittest
from collections import Counter
from PIL import Image,ImageDraw
from h150.visual_region_controller import (
    case,locate_visible_change,OFFSET,N,DROP_ENTRY_OFFSETS
)
def pixels(image):
    buf=io.BytesIO()
    image.save(buf,format="PNG")
    return buf.getvalue()
class H150ScreenOnlyDropRecognitionTests(unittest.TestCase):
    def test_new_fresh_source_balance_no_h149_reuse(self):
        self.assertEqual(OFFSET,154000)
        self.assertEqual(N,16)
        data=[case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in data),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in data}),4)
        self.assertTrue(all(a<b for a,b in zip(DROP_ENTRY_OFFSETS,DROP_ENTRY_OFFSETS[1:])))
        with self.assertRaises(ValueError):
            case(134000)
    def test_extract_center_of_newly_drawn_drop_border_using_pixels_only(self):
        old=Image.new("RGB",(960,720),"white")
        new=old.copy()
        ImageDraw.Draw(new).rectangle((390,232,580,326),outline="gray",width=4)
        point=locate_visible_change(pixels(old),pixels(new),[480,180])
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0],480)
        self.assertTrue(262<=point[1]<=300)
        self.assertIsNone(locate_visible_change(pixels(old),pixels(old),[480,180]))
        distract=old.copy()
        ImageDraw.Draw(distract).rectangle((20,232,170,326),outline="gray",width=4)
        self.assertIsNone(locate_visible_change(pixels(old),pixels(distract),[480,180]))
if __name__=="__main__":
    unittest.main()
