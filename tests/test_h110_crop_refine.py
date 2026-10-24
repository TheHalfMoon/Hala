"""H110 original H61 512-pixel crop refiner tests."""
import json,unittest,tempfile
from pathlib import Path
from PIL import Image
from h110.refine import make_focus,merge_refinements
from h61.official_eval import parse_action

def action(a,b):
    return json.dumps({"type":"drag","start_point":a,"end_point":b,
                       "coordinate_space":"source_image_pixels","ordered":True})

class H110RefineTests(unittest.TestCase):
    def test_source_roi_mapping(self):
        for dims in ((1024,640),(1024,576),(495,640),(452,640)):
            with tempfile.TemporaryDirectory() as tmp:
                src=Path(tmp)/"src.png"
                Image.new("RGB",dims,(20,30,40)).save(src)
                original={"id":"synthetic","image_path":str(src),
                          "image_width":dims[0],"image_height":dims[1],
                          "task":{"intent":"Drag the slider to target","ordered":True}}
                coarse=json.loads(action([100,120],[dims[0]-70,420]))
                start=make_focus(original,coarse,"start",Path(tmp)/"a.png")
                end=make_focus(original,coarse,"end",Path(tmp)/"b.png")
                self.assertEqual(start.scene["image_width"],2*min(dims[0],512))
                self.assertIn("LOCAL crop-image",start.scene["prompt"])
                a=start.frame.source_to_zoomed(coarse["start_point"])
                b=end.frame.source_to_zoomed(coarse["end_point"])
                out,used=merge_refinements(original,action(coarse["start_point"],coarse["end_point"]),start,action(a,a),end,action(b,b))
                self.assertEqual(parse_action(out,*dims),coarse)
                self.assertEqual(used,{"start":True,"end":True})
    def test_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/"src.png";Image.new("RGB",(1024,640)).save(src)
            original={"image_path":str(src),"image_width":1024,"image_height":640,
                      "task":{"intent":"Drag handle","ordered":True}}
            coarse=json.loads(action([100,120],[600,400]))
            a=make_focus(original,coarse,"start",Path(folder)/"a.png")
            b=make_focus(original,coarse,"end",Path(folder)/"b.png")
            out,used=merge_refinements(original,action(coarse["start_point"],coarse["end_point"]),a,"invalid",b,"invalid")
            self.assertEqual(parse_action(out,1024,640),coarse)
            self.assertFalse(any(used.values()))
            with self.assertRaises(ValueError):
                merge_refinements(original,"invalid",a,"invalid",b,"invalid")
if __name__=="__main__":unittest.main()
