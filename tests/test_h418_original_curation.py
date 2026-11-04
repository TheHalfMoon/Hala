"""H418 private original large-data admission and held-out image overlap tests."""
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
import zipfile

from h418 import curate as c

ROOT=Path(__file__).resolve().parents[1]


class H418OriginalSourceTests(unittest.TestCase):
    def test_new_archive_sha_allowlist_matches_owner_original_evidence(self):
        doc=json.loads((ROOT/"h418/ORIGINAL-SHARD-SHA256.json").read_text())
        self.assertFalse(doc["kaggle_remote_byte_hash_verified"])
        self.assertEqual(doc["source_split"],"train")
        self.assertFalse(doc["full_upstream_tar_sha_verified"])
        self.assertEqual(doc["original_64300_rows_with_media_available_after_upload"],30491)
        self.assertEqual(doc["new_original_image_rows_available"],17326)
        self.assertEqual({f["name"]:f["sha256"] for f in doc["files"]},
                         c.EXTRA_SHAS)
        self.assertEqual(len(c.ALL_SHAS),9)
        self.assertEqual(c.FROZEN_ROWS,64300)

    @staticmethod
    def make_row(member,data,index,domain="sheet"):
        return {"id":f"fictional_{index}","split":"train",
                "image_path":f"{domain}/{member}",
                "image_sha256":sha256(data).hexdigest(),
                "family":"hcompany_dragon_h61_"+domain+"_train",
                "prompt":"Drag the visible element",
                "completion":'{"type":"drag"}',"language":"en"}

    def test_selected_originals_are_real_sha_distinct_and_dev_quarantined(self):
        a=b"\\xff\\xd8\\xff".decode("unicode_escape").encode("latin1")+b"a"
        b=b"\\xff\\xd8\\xff".decode("unicode_escape").encode("latin1")+b"b"
        cdata=b"\\xff\\xd8\\xff".decode("unicode_escape").encode("latin1")+b"c"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"real-h60.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("images/sheet/a.jpg",a)
                z.writestr("images/sheet/a2.jpg",a)
                z.writestr("images/sheet/b.jpg",b)
                z.writestr("images/sheet/c.jpg",cdata)
            with zipfile.ZipFile(p) as z:
                source={("sheet",n):(z,"h60-dragon-train-sheet-0000.hala.bin")
                        for n in z.namelist()}
                rows=[
                    self.make_row("images/sheet/a.jpg",a,0),
                    self.make_row("images/sheet/a2.jpg",a,1),
                    self.make_row("images/sheet/b.jpg",b,2),
                    self.make_row("images/sheet/c.jpg",cdata,3),
                ]
                holdout={sha256(b).hexdigest()}
                holdout.update(sha256(f"dev-{i}".encode()).hexdigest()
                               for i in range(143))
                self.assertEqual(len(holdout),144)
                selected,eligible,excluded=c.select(rows,source,holdout,
                            expected_available=4,min_selected=2)
                self.assertEqual(len(selected),2)
                self.assertEqual(len({r["image_sha256"] for r in selected}),2)
                self.assertEqual(eligible,{"sheet":4})
                self.assertEqual(excluded.get("h417_dev_sha_quarantined"),1)
                self.assertEqual(excluded.get("duplicate_source_pixel_sha"),1)
                self.assertEqual(selected[0]["source_zip"],
                                 "h60-dragon-train-sheet-0000.hala.bin")
                self.assertEqual(selected[0]["image_path"],"images/sheet/a.jpg")

    def test_tampered_jpeg_is_rejected_before_adding_to_training(self):
        payload=b"\\xff\\xd8\\xff".decode("unicode_escape").encode("latin1")+b"source"
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"original.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("images/sheet/one.jpg",payload+b"tampered")
            with zipfile.ZipFile(path) as z:
                source={("sheet","images/sheet/one.jpg"):
                        (z,"h60-dragon-train-sheet-0000.hala.bin")}
                row=self.make_row("images/sheet/one.jpg",payload,1)
                heldout={sha256(f"dev-{i}".encode()).hexdigest()
                         for i in range(144)}
                with self.assertRaisesRegex(ValueError,"JPEG bytes differ"):
                    c.select([row],source,heldout,
                             expected_available=1,min_selected=1)

    def test_source_only_export_has_no_weights_and_is_create_only(self):
        payload=b"\\xff\\xd8\\xff".decode("unicode_escape").encode("latin1")+b"media"
        case=self.make_row("images/sheet/a.jpg",payload,1)
        with tempfile.TemporaryDirectory() as tmp:
            doc=c.export([case],{"sheet":1},{"h417_dev_sha_quarantined":0},tmp)
            self.assertEqual(doc["selected_train_source_unique_pixels"],1)
            self.assertFalse(doc["training_performed"])
            self.assertFalse(doc["independent_external_ood"])
            self.assertFalse(doc["weights_public"])
            self.assertEqual(doc["selected_sha256"],
                   sha256((c.canonical(case)+b"\n")).hexdigest())
            with self.assertRaises(FileExistsError):
                c.export([case],{"sheet":1},{},tmp)

    def test_missing_dev_hash_quarantine_refuses_source_admission(self):
        with self.assertRaisesRegex(ValueError,"quarantine"):
            c.select([],{},set(),expected_available=0,min_selected=0)

if __name__=="__main__":
    unittest.main()
