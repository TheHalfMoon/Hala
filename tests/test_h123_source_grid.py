"""H123 independent strict dual-grid source parser and complex selector."""
import hashlib
import json
import unittest

from h123.objective import grids,completion,parse,prompt
from h123.schedule import (_balanced_choose,_holdout_indices,PER_FAMILY,
                           HELDOUT_PER_FAMILY,FAMILIES)


def features(count=320):
    rows=[]
    for i in range(count):
        x=i%8
        rows.append({
            "source_id":f"original-complex-TRAIN-{i:05}",
            "sha":hashlib.sha256(f"original-image-{i}".encode()).hexdigest(),
            "direction":x,"scale":("short","medium","long")[i%3],
            "end_cell":(i%4,(i//4)%4)})
    return rows


class H123SourceTests(unittest.TestCase):
    def test_dual_spatial_cells_and_exact_roundtrip(self):
        for point in ([0,0],[100,250],[249,499],[250,500],
                      [700,300],[999,999],[1000,1000]):
            coarse,fine=grids(point)
            self.assertEqual(coarse,[min(3,point[0]//250),min(3,point[1]//250)])
            self.assertEqual(fine,[min(9,point[0]//100),min(9,point[1]//100)])
            target=parse(completion(point),1000,600)
            self.assertEqual(target,[min(999,round(point[0])),
                                     min(599,round(point[1]*.6))])

    def test_rejects_quadrant_cell_or_native_json_drift(self):
        obj=json.loads(completion([400,600]))
        obj["quadrant"]=[0,0]
        self.assertIsNone(parse(json.dumps(obj),1000,600))
        obj=json.loads(completion([400,600]))
        obj["cell"]=[8,8]
        self.assertIsNone(parse(json.dumps(obj),1000,600))
        obj=json.loads(completion([400,600]))
        obj["coordinate_space"]="source_image_pixels"
        self.assertIsNone(parse(json.dumps(obj),1000,600))
        self.assertIsNone(parse(completion([400,600]).replace('"role":"end"','"role":"end","role":"end"'),1000,600))
        with self.assertRaises(ValueError):
            grids([True,100])

    def test_model_prompt_never_includes_source_answer(self):
        result=prompt("Move the tab to another group",True)
        self.assertIn("10x10 grid",result)
        self.assertIn("4x4 grid",result)
        self.assertNotIn("end_bbox",result)
        self.assertNotIn("start_point",result)
        with self.assertRaises(ValueError):
            prompt("",False)

    def test_exact_256_unique_balanced_family_with_32_heldout(self):
        original=features()
        selected=_balanced_choose(original,PER_FAMILY)
        self.assertEqual(len(selected),256)
        self.assertEqual(len({x["sha"] for x in selected}),256)
        self.assertEqual(len({x["direction"] for x in selected}),8)
        heldout=_holdout_indices(selected)
        self.assertEqual(len(heldout),HELDOUT_PER_FAMILY)
        for direction in range(8):
            self.assertEqual(sum(selected[i]["direction"]==direction for i in heldout),4)
        self.assertEqual(
            [x["source_id"] for x in _balanced_choose(features(),PER_FAMILY)],
            [x["source_id"] for x in selected])

    def test_malformed_original_source_distribution_refused(self):
        with self.assertRaises(ValueError):
            _balanced_choose(features(255),256)
        xs=features()
        for x in xs:
            x["direction"]=0
        with self.assertRaises(ValueError):
            _balanced_choose(xs,256)

    def test_only_four_complex_gui_domains(self):
        self.assertEqual(len(FAMILIES),4)
        self.assertTrue(all("hcompany_dragon_h61_" in x for x in FAMILIES))
        self.assertTrue(all("firstparty" not in x for x in FAMILIES))


if __name__=="__main__":
    unittest.main()
