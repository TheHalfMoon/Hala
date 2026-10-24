"""H109 source coordinate roundtrip and nonleaking image crop tests."""
import hashlib
import unittest
from PIL import Image
from h109.crop_geometry import crop_around_point,CropFrame


class TestH109SourceCrop(unittest.TestCase):
    def test_actual_h108_seven_frame_shaped_examples(self):
        for width,height in ((1024,576),(1024,640),(452,640),(495,640),
                             (828,640),(853,640),(854,640)):
            orig=Image.new("RGB",(width,height),(71,93,113))
            before=hashlib.sha256(orig.tobytes()).digest()
            for x,y in ((0,0),(width//2,height//2),(width-1,height-1)):
                zoom,frame=crop_around_point(orig,[x,y])
                self.assertEqual(zoom.size,(640,640))
                self.assertEqual(frame.source_w,width)
                self.assertEqual(frame.source_h,height)
                self.assertEqual(frame.zoomed_to_source(
                    frame.source_to_zoomed([frame.left,frame.top])),
                    [frame.left,frame.top])
                self.assertEqual(before,hashlib.sha256(orig.tobytes()).digest())

    def test_actual_bounded_source_to_crop_mapping_invertible(self):
        src=Image.new("RGB",(1024,640))
        _,frame=crop_around_point(src,[400,320])
        for y in range(frame.top,frame.top+320,25):
            for x in range(frame.left,frame.left+320,25):
                self.assertEqual(frame.zoomed_to_source(
                    frame.source_to_zoomed([x,y])),[x,y])

    def test_wrong_size_and_unbounded_coordinates_fail(self):
        image=Image.new("RGB",(495,640))
        for bad in ([-1,80],[495,80],[80,640],[True,30]):
            with self.assertRaises(ValueError):
                crop_around_point(image,bad)
        self.assertEqual(crop_around_point(image,[20,20],512)[0].size,(990,1024))
        with self.assertRaises(ValueError):
            crop_around_point(image,[20,20],600)
        with self.assertRaises(ValueError):
            CropFrame(495,640,300,200,300,320)

    def test_bbox_only_inside_roi_at_source_frame(self):
        _,roi=crop_around_point(Image.new("RGB",(1024,640)),[450,330])
        self.assertTrue(roi.source_region_covers(
            [roi.left+12,roi.top+12,roi.left+30,roi.top+30]))
        self.assertFalse(roi.source_region_covers([0,0,10,10]))
        self.assertFalse(roi.source_region_covers([roi.left+2,roi.top+2,
                                                 roi.left+roi.width,roi.top+12]))

    def test_source_and_crop_max_borders_are_not_interchangeable(self):
        roi=CropFrame(495,640,175,160,320,320)
        self.assertEqual(roi.zoomed_to_source([639,639]),[494,479])
        self.assertEqual(roi.source_to_zoomed([494,479]),[638,638])
        with self.assertRaises(ValueError):
            roi.zoomed_to_source([640,0])

if __name__=="__main__":
    unittest.main()
