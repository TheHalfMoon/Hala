"""H104 source↔processor pixel bridge strict original scorer tests (CPU)."""
import unittest
from h104.bridge import FrameBridge,roundtrip_oracle

def scene(srcw,srch,start,end,ordered=True):
    return {"id":"synthetic","domain":"slider","image_width":srcw,
            "image_height":srch,
            "task":{"ordered":ordered,"start_bbox":start,"end_bbox":end}}

class H104MappingTests(unittest.TestCase):
    def test_real_h103_495_to_480_processor_width(self):
        bridge=FrameBridge(495,640,480,640)
        rec=roundtrip_oracle(scene(495,640,[12,14,42,44],[320,470,385,500]),bridge)
        self.assertTrue(rec["valid_source_geometry"])
        self.assertLessEqual(rec["max_pixel_roundtrip_error"],1)
        self.assertFalse(rec["processor_frame_matches_source"])
    def test_real_h103_853_to_864_processor_width(self):
        bridge=FrameBridge(853,640,864,640)
        rec=roundtrip_oracle(scene(853,640,[20,20,50,60],[680,400,720,440]),bridge)
        self.assertTrue(rec["valid_source_geometry"])
        self.assertLessEqual(rec["max_pixel_roundtrip_error"],1)
    def test_identity_original_eval_frame(self):
        bridge=FrameBridge(1024,640,1024,640)
        rec=roundtrip_oracle(scene(1024,640,[2,2,10,14],[900,500,950,550]),bridge)
        self.assertEqual(rec["max_pixel_roundtrip_error"],0)
        self.assertTrue(rec["processor_frame_matches_source"])
    def test_bounded_inverse_forward_pixel_error(self):
        for w,dw in ((495,480),(853,864),(1024,1024)):
            bridge=FrameBridge(w,640,dw,640)
            for x in range(0,w,7):
                source=bridge.to_source(bridge.to_processor([x,500]))
                self.assertLessEqual(abs(source[0]-x),1)
                self.assertEqual(source[1],500)
    def test_strict_source_out_of_range_refused(self):
        bridge=FrameBridge(495,640,480,640)
        for p in ([-1,1],[495,1],[1,640],[True,20]):
            with self.assertRaises(ValueError):bridge.to_processor(p)
        with self.assertRaises(ValueError):FrameBridge(0,640,480,640)
    def test_transformed_boxes_remain_in_grid(self):
        bridge=FrameBridge(853,640,864,640)
        box=bridge.source_bbox_to_processor([0,0,852,639])
        self.assertEqual(box,[0,0,863,639])
    def test_degenerate_reference_refused(self):
        bridge=FrameBridge(495,640,480,640)
        with self.assertRaises(ValueError):
            roundtrip_oracle(scene(495,640,[2,2,4,4],[2,2,4,4]),bridge)
    def test_original_strict_scorer_not_redefined(self):
        bridge=FrameBridge(495,640,480,640)
        from h61.official_eval import score
        source=scene(495,640,[10,10,30,30],[300,400,340,440],False)
        self.assertTrue(roundtrip_oracle(source,bridge)["valid_source_geometry"])
        self.assertEqual(score.__module__,"h61.official_eval")

if __name__=="__main__":unittest.main()
