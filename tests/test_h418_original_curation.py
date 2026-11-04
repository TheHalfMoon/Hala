"""H418 original H60 raw -> H61 resized JPEG SHA / H417 DEV isolation."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from h418 import curate as c

ROOT=Path(__file__).resolve().parents[1]
ZIP_NAME="h60-dragon-train-sheet-0000.hala.bin"


def jpeg(seed):
    from PIL import Image,ImageDraw
    im=Image.new("RGB",(1275,1650),(230,235,245))
    dr=ImageDraw.Draw(im)
    dr.rectangle((seed*5,80,seed*5+180,260),fill=(seed*31%250,90,120))
    dr.text((160,500),f"ORIGINAL TRAIN {seed}",fill=(20,80,120))
    out=BytesIO()
    im.save(out,format="JPEG",quality=87)
    return out.getvalue()


def row(member,original,index):
    resized,(w,h),_=c.derived_h61_training_image(original)
    return {"id":f"fictional_{index}","split":"train",
            "image_path":f"sheet/{member}",
            "image_sha256":sha256(resized).hexdigest(),
            "family":"hcompany_dragon_h61_sheet_train",
            "prompt":f"Drag a fictional element. The image is {w}x{h} pixels.",
            "completion":json.dumps({"type":"drag","start_point":[20,20],
                         "end_point":[w-20,h-20],
                         "ordered":False,"coordinate_space":"source_image_pixels"}),
            "language":"en"}


class H418OriginalSourceTests(unittest.TestCase):
    def pillow(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("original source jpeg pixel tests require Pillow")

    def test_new_archive_sha_allowlist_matches_owner_original_evidence(self):
        doc=json.loads((ROOT/"h418/ORIGINAL-SHARD-SHA256.json").read_text())
        self.assertFalse(doc["kaggle_remote_byte_hash_verified"])
        self.assertEqual(doc["source_split"],"train")
        self.assertFalse(doc["full_upstream_tar_sha_verified"])
        self.assertEqual(doc["original_64300_rows_with_media_available_after_upload"],30491)
        self.assertEqual(doc["new_original_image_rows_available"],17326)
        self.assertEqual({f["name"]:f["sha256"] for f in doc["files"]},c.EXTRA_SHAS)
        self.assertEqual(len(c.ALL_SHAS),9)
        self.assertEqual(c.FROZEN_ROWS,64300)

    def test_resized_h61_sha_is_not_original_h60_sha(self):
        self.pillow()
        original=jpeg(2)
        resized,size,rawsize=c.derived_h61_training_image(original)
        self.assertNotEqual(sha256(original).digest(),sha256(resized).digest())
        self.assertNotEqual(size,rawsize)
        self.assertEqual(size[1],640)
        self.assertEqual(row("images/sheet/x.jpg",original,1)["image_sha256"],
                         sha256(resized).hexdigest())

    def test_selected_real_source_and_dev_quarantined_by_raw_sha(self):
        self.pillow()
        a=jpeg(1);b=jpeg(2);third=jpeg(3)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"original.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("images/sheet/a.jpg",a)
                z.writestr("images/sheet/a2.jpg",a)
                z.writestr("images/sheet/b.jpg",b)
                z.writestr("images/sheet/c.jpg",third)
            with zipfile.ZipFile(p) as z:
                data={"images/sheet/a.jpg":a,"images/sheet/a2.jpg":a,
                      "images/sheet/b.jpg":b,"images/sheet/c.jpg":third}
                source={("sheet",name):(z,ZIP_NAME,sha256(original).hexdigest())
                        for name,original in data.items()}
                rows=[row(name,original,i) for i,(name,original) in enumerate(data.items())]
                holdout={sha256(b).hexdigest()}
                holdout.update(sha256(f"dev-{i}".encode()).hexdigest()
                               for i in range(143))
                self.assertEqual(len(holdout),144)
                self.assertNotIn(rows[2]["image_sha256"],holdout)
                selected,eligible,excluded=c.select(rows,source,holdout,
                                        expected_available=4,min_selected=2)
                self.assertEqual(len(selected),2)
                self.assertEqual(len({r["image_sha256"] for r in selected}),2)
                self.assertEqual(eligible,{"sheet":4})
                self.assertEqual(excluded["h417_dev_sha_quarantined"],1)
                self.assertEqual(excluded["duplicate_source_pixel_sha"],1)
                self.assertEqual(selected[0]["source_zip"],ZIP_NAME)
                self.assertEqual(selected[0]["image_path"],"images/sheet/a.jpg")
                self.assertEqual(selected[0]["source_raw_image_sha256"],
                                 sha256(a).hexdigest())
                self.assertNotEqual(selected[0]["image_sha256"],
                                    selected[0]["source_raw_image_sha256"])

    def test_mutated_source_original_raw_bytes_rejected(self):
        self.pillow()
        original=jpeg(4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("images/sheet/one.jpg",jpeg(5))
            with zipfile.ZipFile(path) as z:
                source={("sheet","images/sheet/one.jpg"):
                        (z,ZIP_NAME,sha256(original).hexdigest())}
                heldout={sha256(f"dev-{i}".encode()).hexdigest() for i in range(144)}
                with self.assertRaisesRegex(ValueError,"H60 original JPEG bytes"):
                    c.select([row("images/sheet/one.jpg",original,1)],source,
                             heldout,expected_available=1,min_selected=1)

    def test_mutated_derived_training_jpeg_sha_rejected(self):
        self.pillow()
        original=jpeg(8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("images/sheet/one.jpg",original)
            with zipfile.ZipFile(path) as z:
                source={("sheet","images/sheet/one.jpg"):
                        (z,ZIP_NAME,sha256(original).hexdigest())}
                fake=row("images/sheet/one.jpg",original,1)
                fake["image_sha256"]="a"*64
                heldout={sha256(f"dev-{i}".encode()).hexdigest() for i in range(144)}
                with self.assertRaisesRegex(ValueError,"derived H61 TRAIN JPEG bytes"):
                    c.select([fake],source,heldout,expected_available=1,min_selected=1)

    def test_original_zip_unlabelled_jpegs_are_quarantined(self):
        from unittest.mock import patch
        # Source may retain raw JPEGs lacking a TRAIN annotation. This is not
        # an excuse to train on them, nor to reject qualified labelled images.
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/ZIP_NAME
            media={"images/sheet/qualified.jpg":b"\\xff\\xd8\\xffqualified",
                   "images/sheet/unlabelled.jpg":b"\\xff\\xd8\\xfforphan"}
            labels=(json.dumps({"split":"train","domain":"sheet",
                     "image_path":"images/sheet/qualified.jpg",
                     "image_sha256":sha256(media["images/sheet/qualified.jpg"]).hexdigest()
                    },sort_keys=True)+"\n").encode()
            manifest={"schema":"hala.h60.dragon-train-shard.v1",
                    "source_commit_sha":c.SOURCE_REV,"source_domain":"sheet",
                    "split":"train","official_eval_or_test_used":False,
                    "benchmark_answer_leakage":False,"source_license":"Apache-2.0",
                    "labels_sha256":sha256(labels).hexdigest()}
            with zipfile.ZipFile(file,"w") as z:
                z.writestr("manifest.json",json.dumps(manifest))
                z.writestr("labels.jsonl",labels)
                for path,blob in media.items():
                    z.writestr(path,blob)
            with patch.object(c,"ALL_SHAS",{ZIP_NAME:sha256(file.read_bytes()).hexdigest()}):
                source,opened=c.open_archives(tmp,tmp)
                try:
                    self.assertIn(("sheet","images/sheet/qualified.jpg"),source)
                    self.assertNotIn(("sheet","images/sheet/unlabelled.jpg"),source)
                finally:
                    for z in opened:z.close()

    def test_source_only_export_no_weights_and_create_only(self):
        case={"image_sha256":"b"*64,"source_raw_image_sha256":"a"*64}
        with tempfile.TemporaryDirectory() as tmp:
            proof=c.export([case],{"sheet":1},{"h417_dev_sha_quarantined":0},tmp)
            self.assertFalse(proof["training_performed"])
            self.assertFalse(proof["weights_public"])
            self.assertFalse(proof["independent_external_ood"])
            self.assertEqual(proof["selected_train_source_unique_pixels"],1)
            self.assertEqual(proof["selected_sha256"],
                             sha256(c.canonical(case)+b"\n").hexdigest())
            with self.assertRaises(FileExistsError):
                c.export([case],{"sheet":1},{},tmp)

    def test_missing_dev_source_hash_quarantine_rejected(self):
        with self.assertRaisesRegex(ValueError,"quarantine"):
            c.select([],{},set(),expected_available=0,min_selected=0)


if __name__=="__main__":
    unittest.main()
