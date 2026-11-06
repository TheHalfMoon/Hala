"""Offline first-party synthetic H426 bilingual geometry source proof.

No third-party visual assets, real UIs, training data, Qwen model or GPU.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

g = load("h426_synthetic", ROOT/"h426/synthetic_drag_curriculum.py")
loss = load("h426_synthetic_target_loss", ROOT/"h426/target_aware_loss.py")


class H426InventedBilingualGUI(unittest.TestCase):
    def test_valid_rgb_png_and_original_pixel_sha(self):
        sample = g.generate("train", 17, "slide_resize")
        with Image.open(io.BytesIO(sample["original_png_bytes"])) as opened:
            self.assertEqual(opened.format, "PNG")
            self.assertEqual(opened.size, (960, 640))
            self.assertEqual(opened.mode, "RGB")
        self.assertEqual(hashlib.sha256(sample["original_png_bytes"]).hexdigest(),
                         sample["image_sha256"])
        self.assertEqual(sample["schema"],g.SCHEMA)

    def test_repeatability_and_different_actions_one_original_image(self):
        for domain in g.DOMAINS:
            with self.subTest(domain=domain):
                a = g.generate("train", 27, domain)
                b = g.generate("train", 27, domain)
                self.assertEqual(a["image_sha256"],b["image_sha256"])
                self.assertEqual(a["actions"],b["actions"])
                self.assertEqual([x["language"] for x in a["actions"]],["en","ar"])
                self.assertNotEqual(a["actions"][0]["target"],a["actions"][1]["target"])
                self.assertEqual({x["image_sha256"] for x in a["actions"]},
                                 {a["image_sha256"]})

    def test_all_endpoints_have_visible_rendered_target_marker(self):
        for domain in g.DOMAINS:
            for split in ("train", "dev"):
                for index in (0, 1, 7, 42):
                    sample=g.generate(split,index,domain)
                    for row in sample["actions"]:
                        with self.subTest(domain=domain,split=split,
                                          index=index,language=row["language"]):
                            # The label must come from a visible target,
                            # not an invented unseen random end position.
                            pixel=sample["image"].getpixel(tuple(row["end"]))
                            self.assertEqual(pixel,g.TARGET_COLOR)

    def test_original_to_normalized_coordinate_geometry(self):
        for domain in g.DOMAINS:
            sample=g.generate("train",31,domain)
            for row in sample["actions"]:
                with self.subTest(domain=domain,lang=row["language"]):
                    expected_start=[round(row["start"][0]*1000/g.WIDTH),
                                    round(row["start"][1]*1000/g.HEIGHT)]
                    expected_end=[round(row["end"][0]*1000/g.WIDTH),
                                  round(row["end"][1]*1000/g.HEIGHT)]
                    self.assertEqual(row["target"]["start_point"],expected_start)
                    self.assertEqual(row["target"]["end_point"],expected_end)
                    self.assertNotEqual(expected_start,expected_end)
                    self.assertTrue(all(0<=v<=1000 for v in expected_start+expected_end))

    def test_h426_source_coordinate_token_mask_accepts_every_pair(self):
        for domain in g.DOMAINS:
            sample=g.generate("dev",36,domain)
            for row in sample["actions"]:
                with self.subTest(domain=domain,lang=row["language"]):
                    text=json.dumps(row["target"],separators=(",",":"),ensure_ascii=False)
                    parts=list(text)+["<|im_end|>"]
                    weights=loss.source_coordinate_token_weights(parts,domain)
                    self.assertEqual(len(weights),len(parts))
                    self.assertGreater(sum(x>1 for x in weights),0)
                    self.assertEqual(weights[-1],1.)

    def test_en_ar_visual_target_mapping_is_balanced_not_language_shortcut(self):
        for split in ("train", "dev"):
            for domain in g.DOMAINS:
                first_target_en=0
                for index in range(20):
                    with self.subTest(split=split,domain=domain,index=index):
                        _,_,visual_rows=g._draw_scene(split,index,domain)
                        sample=g.generate(split,index,domain)
                        self.assertEqual(
                            [r["language"] for r in sample["actions"]],
                            ["en","ar"])
                        expected_offset=(index+g.DOMAINS.index(domain))%2
                        first_target_en+=int(expected_offset==0)
                        for n,row in enumerate(sample["actions"]):
                            idx=(n+expected_offset)%2
                            self.assertEqual(tuple(row["start"]),visual_rows[idx][0])
                            self.assertEqual(tuple(row["end"]),visual_rows[idx][1])
                            if domain=="slider":
                                self.assertIn(str(idx+1),row["body"])
                            elif domain=="slide_resize":
                                self.assertIn(str(idx+1),row["body"])
                            else:
                                word=visual_rows[idx][2]
                                self.assertIn(word,row["body"])
                # EN and AR each point at both visual indices equally.
                self.assertEqual(first_target_en,10)

    def test_geometry_and_visible_text_cannot_be_memorized_as_fixed_pixels(self):
        for domain in g.DOMAINS:
            starts, ends, words = set(), set(), set()
            for index in range(40):
                sample = g.generate("train", index, domain)
                for row in sample["actions"]:
                    starts.add(tuple(row["start"]))
                    ends.add(tuple(row["end"]))
                if domain == "text_highlight":
                    _,_,records=g._draw_scene("train",index,domain)
                    words.update(v[2] for v in records)
            with self.subTest(domain=domain):
                self.assertGreater(len(starts), 20)
                self.assertGreater(len(ends), 20)
                if domain == "text_highlight":
                    self.assertGreaterEqual(len(words), 10)
        self.assertEqual(g.SCHEMA,
            "hala.h426.firstparty-invented-bilingual-paired-drag.v3")

    def test_source_disjoint_families_and_unique_large_sample(self):
        report=g.audit_balanced(8)
        self.assertEqual(report["generated_images"],48)
        self.assertEqual(report["generated_action_rows"],96)
        self.assertFalse(report["train_dev_source_family_overlap"])
        self.assertFalse(report["independent_external_ood"])
        self.assertFalse(report["trained_model"])
        self.assertEqual(len(report["counts"]),12)
        self.assertTrue(all(v==8 for v in report["counts"].values()))

    def test_split_image_sha_cannot_overlap_for_same_index(self):
        for domain in g.DOMAINS:
            for index in (0,1,42,79,80):
                a=g.generate("train",index,domain)
                b=g.generate("dev",index,domain)
                self.assertNotEqual(a["image_sha256"],b["image_sha256"])
                self.assertNotEqual(a["family"],b["family"])

    def test_invalid_inputs_rejected(self):
        invalid=(("release",0,"slider"),("train",-1,"slider"),
                 ("train",True,"slider"),("train",10**6,"slider"),
                 ("train",0,"other"),("dev","1","slider"))
        for split,index,domain in invalid:
            with self.subTest(index=index),self.assertRaises(ValueError):
                g.generate(split,index,domain)
        for n in (0,-1,False,513):
            with self.subTest(n=n),self.assertRaises(ValueError):
                g.audit_balanced(n)

    def test_no_external_asset_or_training_admission(self):
        source=(ROOT/"h426/synthetic_drag_curriculum.py").read_text("utf8")
        self.assertNotIn("requests.get(",source)
        self.assertNotIn("http://",source)
        self.assertNotIn("https://",source)
        self.assertNotIn("model.generate(",source)
        self.assertNotIn("optimizer.step(",source)
        self.assertNotIn("kaggle kernels push",source)
        sample=g.generate("train",3,"slider")
        self.assertTrue(all("not_admitted" in a["source_role"]
                            for a in sample["actions"]))


if __name__=="__main__":
    unittest.main()
