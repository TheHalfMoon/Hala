"""CPU-only H140 original source and model geometry-collapse guards."""
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zlib
from functools import lru_cache
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from h140.qualification import (COUNTS,SCHEMA,admit,collapse_gate,digest,
                                 actual_png_bytes,summarize_neural_position_diversity)


class H140PositionDiversityTests(unittest.TestCase):
    @staticmethod
    @lru_cache(maxsize=512)
    def _base_png(label="H393-synthetic-base"):
        """One invented native-size PNG, unique *pixels* per source case."""
        stream=io.BytesIO()
        image=Image.new("RGB",(960,720),(24,42,60))
        fingerprint=hashlib.sha256(label.encode("utf8")).digest()
        for idx in range(8):
            image.putpixel((idx,0),tuple(fingerprint[idx*3:(idx+1)*3]))
        image.save(stream,format="PNG")
        return stream.getvalue()

    def _source(self,root,split,app_counts):
        folder=root/split
        folder.mkdir()
        sources={}
        cases=[]
        index=0
        for app,amount in app_counts.items():
            sha=hashlib.sha1(("synthetic-H140-fixture/"+app).encode()).hexdigest()
            sources[app]={"license":"MIT","license_file_verified":True,
                "tracked_source_modified":False,"upstream_sha":sha,
                "application_is_complete":True}
            for n in range(amount):
                filename=f"{app}-{n:03d}.png"
                base_png=self._base_png(f"{split}/{app}/{n}")
                # Add a unique, correctly checksummed PNG metadata chunk.
                # Reusing this real PNG payload is a synthetic test fixture only.
                label=f"H140 synthetic {split} {app} {n}".encode()
                payload=b"fixture-id"+bytes([0])+label
                chunk=(struct.pack(">I",len(payload))+b"tEXt"+payload+
                       struct.pack(">I",zlib.crc32(b"tEXt"+payload)))
                pixels=base_png[:33]+chunk+base_png[33:]
                (folder/filename).write_bytes(pixels)
                cases.append({"image":filename,"image_sha256":digest(pixels),
                    "app":app,"app_git_sha":sha,"split":split,
                    "language":"en" if n%2==0 else "ar",
                    "instruction":f"Drag task card {app} {n} to the next column",
                    "reference_drag_persisted":True,
                    "model_input_contains_oracle_coordinates":False,
                    "start_bbox":{"x":40+(n%16)*32,"y":240,"width":16,"height":20},
                    "end_bbox":{"x":60+(n%16)*31,"y":370,"width":36,"height":25}})
                index+=1
        assert len(cases)==COUNTS[split]
        (folder/"manifest.json").write_text(json.dumps({
            "schema":SCHEMA,"private":True,"split":split,
            "training_eligible":split!="blind_test",
            "app_sources":sources,"cases":cases}),encoding="utf8")
        return folder

    def test_source_split_app_disjointness_and_authentic_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            dev=self._source(root,"dev",{"training_a":22,"training_b":22,"training_c":20})
            blind=self._source(root,"blind_test",{"unseen_d":16,"unseen_e":16})
            historic=root/"historic.json"
            historic.write_text(json.dumps({"cases":[{"image_sha256":"f"*64}]}))
            report=admit(train,dev,blind,[historic])
            self.assertEqual(report["distinct_images"],352)
            self.assertEqual(report["status"],"QUALIFIED_SOURCE_ONLY_NOT_TRAINED")
            self.assertEqual(report["real_neural_calls"],0)
            with self.assertRaises(ValueError):
                admit(train,dev,blind,[])
            document=json.loads((train/"manifest.json").read_text("utf8"))
            for row in document["cases"]:
                row["end_bbox"]={"x":80,"y":370,"width":20,"height":25}
            (train/"manifest.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,"fixed-coordinate collapse"):
                admit(train,dev,blind,[historic])

    def test_blind_source_sha_alias_refused_even_if_app_name_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            dev=self._source(root,"dev",{"training_a":22,"training_b":22,"training_c":20})
            blind=self._source(root,"blind_test",{"unseen_d":16,"unseen_e":16})
            historic=root/"historic.json"
            historic.write_text(json.dumps({"cases":[{"image_sha256":"f"*64}]}))
            self.assertEqual(admit(train,dev,blind,[historic])["distinct_images"],352)
            train_doc=json.loads((train/"manifest.json").read_text("utf8"))
            blind_doc=json.loads((blind/"manifest.json").read_text("utf8"))
            borrowed_sha=train_doc["app_sources"]["training_a"]["upstream_sha"]
            blind_doc["app_sources"]["unseen_d"]["upstream_sha"]=borrowed_sha
            for row in blind_doc["cases"]:
                if row["app"]=="unseen_d":
                    row["app_git_sha"]=borrowed_sha
            (blind/"manifest.json").write_text(json.dumps(blind_doc))
            with self.assertRaisesRegex(ValueError,"source revision"):
                admit(train,dev,blind,[historic])

    def test_dev_same_name_source_revision_drift_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            dev=self._source(root,"dev",{"training_a":22,"training_b":22,"training_c":20})
            blind=self._source(root,"blind_test",{"unseen_d":16,"unseen_e":16})
            historic=root/"historic.json"
            historic.write_text(json.dumps({"cases":[{"image_sha256":"f"*64}]}))
            self.assertEqual(admit(train,dev,blind,[historic])["distinct_images"],352)
            dev_doc=json.loads((dev/"manifest.json").read_text("utf8"))
            altered_sha="a"*40
            dev_doc["app_sources"]["training_a"]["upstream_sha"]=altered_sha
            for row in dev_doc["cases"]:
                if row["app"]=="training_a":
                    row["app_git_sha"]=altered_sha
            (dev/"manifest.json").write_text(json.dumps(dev_doc))
            with self.assertRaisesRegex(ValueError,"source revision"):
                admit(train,dev,blind,[historic])

    def test_same_split_renamed_source_revision_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            doc=json.loads((train/"manifest.json").read_text("utf8"))
            aliased_sha=doc["app_sources"]["training_a"]["upstream_sha"]
            doc["app_sources"]["training_b"]["upstream_sha"]=aliased_sha
            for row in doc["cases"]:
                if row["app"]=="training_b":
                    row["app_git_sha"]=aliased_sha
            (train/"manifest.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"source revision"):
                from h140.qualification import load_split
                load_split(train,"train")

    def test_header_only_png_with_claimed_sha_must_not_be_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            fake=(bytes.fromhex("89504e470d0a1a0a0000000d")+b"IHDR"+
                  struct.pack(">II",960,720)+b"not-a-real-PNG-image")
            (folder/"claim.png").write_bytes(fake)
            with self.assertRaisesRegex(ValueError,"H140 invalid PNG"):
                actual_png_bytes(folder,{"image":"claim.png","image_sha256":digest(fake)})

    def test_valid_png_digest_cannot_excuse_corrupted_idat_or_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            source=self._base_png()
            damaged=bytearray(source)
            pos=damaged.index(b"IDAT")+5
            damaged[pos]^=1
            for name,binary in (
                ("bad-crc.png",bytes(damaged)),
                ("missing-iend.png",source[:-12]),
            ):
                with self.subTest(name=name):
                    (folder/name).write_bytes(binary)
                    with self.assertRaisesRegex(ValueError,"H140 invalid PNG"):
                        actual_png_bytes(folder,{"image":name,
                                                 "image_sha256":digest(binary)})

    @staticmethod
    def _same_pixels_with_different_metadata(original_png,other_png):
        """Re-encode exactly the same image pixels with distinct PNG metadata."""
        info=PngInfo()
        info.add_text("h393-original-capture",other_png.name)
        with Image.open(original_png) as image:
            with other_png.open("wb") as output:
                image.save(output,format="PNG",pnginfo=info)
        assert digest(original_png.read_bytes())!=digest(other_png.read_bytes())
        with Image.open(original_png) as left, Image.open(other_png) as right:
            assert left.convert("RGBA").tobytes()==right.convert("RGBA").tobytes()

    def test_manifest_distinct_png_sha_must_not_hide_same_decoded_pixels(self):
        from h140.qualification import load_split
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            first=train/"training_a-000.png"
            second=train/"training_a-001.png"
            self._same_pixels_with_different_metadata(first,second)
            doc=json.loads((train/"manifest.json").read_text("utf8"))
            for row in doc["cases"]:
                if row["image"]==second.name:
                    row["image_sha256"]=digest(second.read_bytes())
            (train/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"duplicate decoded pixels"):
                load_split(train,"train")

    def test_cross_split_metadata_alias_refused_by_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=self._source(root,"train",{"training_a":86,"training_b":86,"training_c":84})
            dev=self._source(root,"dev",{"training_a":22,"training_b":22,"training_c":20})
            blind=self._source(root,"blind_test",{"unseen_d":16,"unseen_e":16})
            historic=root/"historic.json"
            historic.write_text(json.dumps({"cases":[{"image_sha256":"f"*64}]}))
            first=train/"training_a-000.png"
            second=dev/"training_a-000.png"
            self._same_pixels_with_different_metadata(first,second)
            doc=json.loads((dev/"manifest.json").read_text("utf8"))
            for row in doc["cases"]:
                if row["image"]==second.name:
                    row["image_sha256"]=digest(second.read_bytes())
            (dev/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"duplicate decoded pixels"):
                admit(train,dev,blind,[historic])

    def test_transparent_rgba_screen_cannot_claim_independent_visible_pixels(self):
        # Browser screenshot provenance requires opaque visible pixels. A hidden
        # RGB value under transparent alpha can vary without changing the scene.
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for alpha in (0,128):
                with self.subTest(alpha=alpha):
                    image=Image.new("RGBA",(960,720),(22,44,66,255))
                    image.putpixel((0,0),(187,13,205,alpha))
                    name=f"nonopaque-{alpha}.png"
                    image.save(folder/name,format="PNG")
                    binary=(folder/name).read_bytes()
                    with self.assertRaisesRegex(ValueError,"H140 non-opaque PNG"):
                        actual_png_bytes(folder,{"image":name,
                                                "image_sha256":digest(binary)})

    def test_opaque_rgb_and_rgba_have_identical_decoded_pixel_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            fingerprints=[]
            for mode in ("RGB","RGBA"):
                image=Image.new(mode,(960,720),(22,44,66))
                name=f"opaque-{mode}.png"
                image.save(folder/name,format="PNG")
                binary=(folder/name).read_bytes()
                out=[]
                self.assertEqual(actual_png_bytes(
                    folder,{"image":name,"image_sha256":digest(binary)},
                    pixel_digest_out=out),binary)
                self.assertEqual(len(out),1)
                fingerprints.append(out[0])
            self.assertEqual(fingerprints[0],fingerprints[1])

    def test_neural_collapse_gate_does_not_accept_fixed_points(self):
        repeated=[[240,340] for _ in range(16)]
        collapse=collapse_gate(repeated)
        self.assertFalse(collapse["passed"])
        self.assertEqual(collapse["unique_points"],1)
        diverse=[[30+i*25,100+i*16] for i in range(16)]
        improved=collapse_gate(diverse)
        self.assertTrue(improved["passed"])
        self.assertEqual(improved["modal_share"],1/16)
        with self.assertRaises(ValueError):
            collapse_gate([[30,40]]*15)
        with self.assertRaises(ValueError):
            collapse_gate([[None,None]]*16)
        with self.assertRaises(ValueError):
            collapse_gate([[960,720]]*16)

    def test_quantized_diversity_refuses_subpixel_jitter_and_five_in_one_bin(self):
        # All 16 predictions are numerically unique but sit in ONE 24px cell.
        jitter=[[101+i/100,201+i/100] for i in range(16)]
        report=collapse_gate(jitter)
        self.assertEqual(report["unique_points"],16)
        self.assertEqual(report["unique_position_bins"],1)
        self.assertFalse(report["passed"])
        # Eight spatial bins alone are insufficient if >25% are one location.
        five=[[101+i/100,201+i/100] for i in range(5)]
        dispersed=[[12+25*i,400] for i in range(11)]
        report=collapse_gate(five+dispersed)
        self.assertGreaterEqual(report["unique_position_bins"],8)
        self.assertGreater(report["largest_position_bin_count"],4)
        self.assertFalse(report["passed"])

    def test_eight_position_bins_with_two_per_bin_still_pass(self):
        points=[[24+27*cell+0.1*j,120] for cell in range(8) for j in range(2)]
        report=collapse_gate(points)
        self.assertEqual(report["unique_points"],16)
        self.assertEqual(report["unique_position_bins"],8)
        self.assertEqual(report["largest_position_bin_count"],2)
        self.assertTrue(report["passed"])

    def test_abstention_and_malformed_coordinates_fail_closed(self):
        diverse=[[30+i*25,100+i*16] for i in range(16)]
        with_abstention=[None]+diverse[1:]
        summary=summarize_neural_position_diversity(
            with_abstention,allow_abstain=True)
        self.assertEqual(summary["valid_points"],15)
        self.assertFalse(summary["passed"])
        with self.assertRaises(ValueError):
            summarize_neural_position_diversity(with_abstention)
        for bad in ([float("nan"),100],[float("inf"),100],
                    [True,100],[-1,100],[960,100]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                summarize_neural_position_diversity([bad]+diverse[1:])


if __name__=="__main__":
    unittest.main()
