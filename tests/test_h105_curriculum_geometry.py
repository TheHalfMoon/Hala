"""H105 10k original-schema CPU curriculum diversity/integrity regression tests."""
import hashlib
import json
import unittest

from h105.curriculum import propose,source_geometry,aggregate

DELTA=((50,0),(40,40),(0,50),(-40,40),
       (-50,0),(-40,-40),(0,-50),(40,-40))


def corpus():
    rows,upstream=[],{}
    for n in range(10000):
        direction=0 if n<6200 else (n%7)+1
        dx,dy=DELTA[direction]
        a=[110+(n%4)*220,90+((n//4)%4)*135]
        b=[a[0]+dx,a[1]+dy]
        action={"type":"drag","start_point":a,"end_point":b,
                "coordinate_space":"source_image_pixels","ordered":True}
        ident=f"source-original-train-{n:05d}"
        sha=hashlib.sha256(ident.encode()).hexdigest()
        lang="ar" if n%10==0 else "en"
        family=f"family_{n%5}"
        row={"id":ident,"split":"train","family":family,"language":lang,
             "image_path":f"media/{ident}.jpg","image_sha256":sha,
             "completion":json.dumps(action)}
        src={"id":ident,"split":"train","action_type":"drag","ordered":True,
             "image_width":1024,"image_height":640,
             "image_path":row["image_path"],"image_sha256":sha,
             "start_bbox":[a[0]-2,a[1]-2,a[0]+2,a[1]+2],
             "end_bbox":[b[0]-2,b[1]-2,b[0]+2,b[1]+2],
             "completion":action}
        rows.append(row);upstream[ident]=src
    return rows,upstream


class H105SourceCurriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.src=corpus()

    def test_diversifies_extremely_direction_biased_original_train(self):
        proposal=propose(self.rows,self.src,128)
        self.assertEqual(len(proposal.indices),128)
        self.assertTrue(proposal.summary["all_gates_passed"])
        self.assertLess(proposal.summary["largest_direction_fraction"],.4)
        self.assertGreaterEqual(len(proposal.summary["direction"]),6)
        self.assertGreaterEqual(proposal.summary["language"]["ar"],26)
        self.assertEqual(proposal.summary["unique_images"],128)
        self.assertEqual(set(proposal.summary["family"]),{f"family_{i}" for i in range(5)})
        self.assertEqual(proposal.summary["source_split"],"TRAIN_ONLY")
        self.assertNotIn(self.rows[0]["id"],str(proposal.summary))
        self.assertEqual(len(proposal.sha256),64)

    def test_source_boundary_quarantine_without_mutating_original(self):
        row=self.rows[2].copy()
        orig=self.src[row["id"]].copy()
        orig["end_bbox"]=[0,0,1024,50]
        self.assertIsNone(source_geometry(row,orig))
        self.assertEqual(self.src[row["id"]]["end_bbox"]!=orig["end_bbox"],True)

    def test_source_completion_mismatch_fails_closed(self):
        row=self.rows[1].copy();row["completion"]=json.dumps({"type":"click"})
        with self.assertRaises(ValueError):source_geometry(row,self.src[row["id"]])

    def test_eval_label_forbidden(self):
        row=self.rows[1].copy();row["split"]="eval"
        with self.assertRaises(ValueError):source_geometry(row,self.src[row["id"]])

if __name__=="__main__":unittest.main()
