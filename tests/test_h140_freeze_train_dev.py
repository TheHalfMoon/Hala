"""Source freeze must reject previously evaluated data and invalid reference drags."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from h140.freeze_train_dev import (
    SOURCE_APPS,SOURCE_RUNS,validate_run,image_sha_exclusions,freeze
)
from h140.qualification import digest


class H140TrainDevOwnerFreezeTests(unittest.TestCase):
    def _make(self,folder,sha=None):
        folder.mkdir()
        rows=[]
        for i in range(16):
            f=f"new_{i:02d}.png"
            pixels=io.BytesIO()
            Image.new("RGB",(960,720),(i*11,40,115)).save(pixels,"PNG")
            (folder/f).write_bytes(pixels.getvalue())
            rows.append({
                "id":f"original-run-{i:02d}","app":"vanilla-native-board",
                "app_git_sha":SOURCE_APPS["vanilla-native-board"]["sha"],
                "split":"pilot_only_not_training",
                "image":f,"image_sha256":digest(pixels.getvalue()),
                "language":"en" if i%2==0 else "ar",
                "instruction":f"Move native card original case {i}",
                "start_bbox":{"x":50+i*24,"y":150,"width":20,"height":18},
                "end_bbox":{"x":90+i*26,"y":270,"width":20,"height":18},
                "reference_drag_persisted":True,
                "reference_survived_page_reload":True,
                "model_input_contains_oracle_coordinates":False,
            })
        doc={"schema":SOURCE_APPS["vanilla-native-board"]["schema"],
             "status":"POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY",
             "private":True,"training_eligible":False,"blind_eligible":False,
             "cases":rows}
        (folder/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
        return rows

    def test_new_native_original_screens_and_history_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"private-h140";root.mkdir()
            rows=self._make(root/"train-candidate-vanilla86")
            cases,metrics=validate_run(root/"train-candidate-vanilla86",
                                       "vanilla-native-board",16,set(),set())
            self.assertEqual(len(cases),16)
            self.assertEqual(metrics["language_counts"],{"en":8,"ar":8})
            self.assertEqual(metrics["start_bins"],16)
            self.assertEqual(metrics["end_bins"],16)
            with self.assertRaisesRegex(ValueError,"duplicate, leaked"):
                validate_run(root/"train-candidate-vanilla86","vanilla-native-board",
                             16,{rows[0]["image_sha256"]},set())
            doc=json.loads((root/"train-candidate-vanilla86"/"manifest.json").read_text("utf8"))
            doc["cases"][0]["reference_survived_page_reload"]=False
            (root/"train-candidate-vanilla86"/"manifest.json").write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                validate_run(root/"train-candidate-vanilla86",
                             "vanilla-native-board",16,set(),set())

    def test_exclusions_require_every_old_stage_and_all_diagnostic_runs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);private=root/"private-h140";private.mkdir()
            for k in ("private-h136","private-h137","private-h138","private-h139"):
                x=root/k;x.mkdir()
                (x/"manifest.json").write_text(json.dumps(
                    {"cases":[{"image_sha256":digest(k.encode())}]}))
            for k in ("old-vanilla-pilot","old-react-pilot","old-dndkit-pilot"):
                x=private/k;x.mkdir()
                (x/"manifest.json").write_text(json.dumps(
                    {"training_eligible":False,"blind_eligible":False,
                     "cases":[{"image_sha256":digest(k.encode())}]}))
            previous,n_old,n_pilots=image_sha_exclusions(private,root)
            self.assertEqual(len(previous),7)
            self.assertEqual(n_old,4)
            self.assertEqual(n_pilots,3)
            (private/"old-react-pilot"/"manifest.json").unlink()
            with self.assertRaises(ValueError):
                image_sha_exclusions(private,root)

    def test_source_split_is_explicitly_predeclared_and_create_only(self):
        self.assertEqual(sum(x[1] for x in SOURCE_RUNS["train"].values()),256)
        self.assertEqual(sum(x[1] for x in SOURCE_RUNS["dev"].values()),64)
        self.assertEqual(set(SOURCE_RUNS["train"]),set(SOURCE_RUNS["dev"]))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"private-h140";root.mkdir()
            (root/"frozen-source-v1").mkdir()
            with self.assertRaisesRegex(ValueError,"create-only"):
                freeze(root)


if __name__=="__main__":
    unittest.main()
