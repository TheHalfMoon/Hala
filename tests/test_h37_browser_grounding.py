"""Verify actual local Chromium GUI fixture PNGs and DOM-bound source oracles.

These are authored browser pages. Synthetic browser screenshot checks are not
neural image grounding or independently captured real-world GUI outcomes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h34.protocol import score_attempt
from h37.freeze import checked_png, freeze
from h37.scenes import cases, html_for_case, model_projection, normalized_scenes, scenes_sha256

ROOT = Path("h37/assets")


class BrowserGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = cases()
        cls.meta = json.loads((ROOT / "manifest.json").read_text("utf-8"))
        cls.audit = json.loads((ROOT / "evaluator_only.json").read_text("utf-8"))
        cls.model = json.loads((ROOT / "model_inputs.json").read_text("utf-8"))
        cls.capture = json.loads((ROOT / "capture.json").read_text("utf-8"))

    def test_split_scopes_and_target_distribution(self):
        self.assertEqual(len(self.scenes), 64)
        self.assertEqual(len(self.audit), 64)
        self.assertEqual(len(self.model), 64)
        self.assertEqual(Counter(s["split"] for s in self.scenes), {"dev": 32, "test": 32})
        self.assertEqual(Counter(s["language"] for s in self.scenes), {"en": 32, "ar": 32})
        self.assertEqual(Counter(s["expected_control_id"] for s in self.scenes),
                         {f"control-{i}":16 for i in range(4)})
        assert_domains = {"dev": {"makerspace", "community_theater"},
                          "test": {"trail_maintenance", "library_workshops"}}
        for split, expected in assert_domains.items():
            self.assertEqual({s["family"] for s in self.scenes if s["split"] == split}, expected)
        self.assertEqual(len({x["id"] for x in self.scenes}), 64)

    def test_no_evaluator_oracles_in_model_projection(self):
        self.assertEqual(self.model, model_projection(self.scenes))
        forbidden = {"expected_label", "expected_control_id", "controls",
                     "bounds", "oracle", "target_index", "options"}
        for row in self.model:
            self.assertFalse(forbidden.intersection(row.keys()))
            self.assertTrue(row["image"].startswith(row["id"]))
            self.assertTrue(row["goal"])
        self.assertEqual(self.meta["model_inputs_sha256"],
                         hashlib.sha256((ROOT/"model_inputs.json").read_bytes()).hexdigest())
        self.assertEqual(self.meta["evaluator_sha256"],
                         hashlib.sha256((ROOT/"evaluator_only.json").read_bytes()).hexdigest())
        self.assertFalse(self.meta["independently_blinded_test"])
        self.assertFalse(self.meta["real_external_apps"])
        self.assertFalse(self.meta["model_evaluated"])

    def test_all_64_real_browser_pngs_and_html_are_valid_and_frozen(self):
        self.assertEqual(self.meta["scenes_sha256"], scenes_sha256())
        self.assertEqual(self.meta["capture_sha256"],
                         hashlib.sha256((ROOT/"capture.json").read_bytes()).hexdigest())
        self.assertEqual(self.capture["playwright_core"], "1.56.1")
        self.assertTrue(self.capture["browser_version"])
        hashes = set()
        for scene, receipt in zip(self.scenes, self.meta["images"]):
            self.assertEqual(scene["id"], receipt["id"])
            html_bytes=html_for_case(scene).encode("utf-8")
            self.assertEqual(hashlib.sha256(html_bytes).hexdigest(), receipt["page_sha256"])
            self.assertEqual((ROOT/"pages"/(scene["id"]+".html")).read_bytes(),html_bytes)
            png=(ROOT/"screenshots"/(scene["id"]+".png")).read_bytes()
            self.assertEqual(checked_png(png),(960,540))
            self.assertEqual(hashlib.sha256(png).hexdigest(),receipt["image_sha256"])
            self.assertEqual(len(png),receipt["bytes"])
            hashes.add(receipt["image_sha256"])
        self.assertEqual(len(hashes),64)

    def test_browser_observations_and_model_target_scoring(self):
        for scene, row in zip(self.scenes, self.audit):
            self.assertEqual(row["id"],scene["id"])
            self.assertEqual(row["language"],scene["language"])
            self.assertEqual(row["split"],scene["split"])
            self.assertEqual(len(row["controls"]),4)
            matched=[c for c in row["controls"] if c["element_ref"]==row["expected_control_id"]]
            self.assertEqual(len(matched),1)
            self.assertEqual(matched[0]["label"],row["expected_label"])
            self.assertEqual([x["label"] for x in row["controls"]],scene["options"])
            for ctl in row["controls"]:
                x,y,w,h=ctl["bounds"]
                self.assertTrue(0<=x<x+w<=960 and 0<=y<y+h<=540)
            t=matched[0]["bounds"]
            response=json.dumps({"point":[(t[0]+t[2]/2)/960,(t[1]+t[3]/2)/540]})
            scored=score_attempt(row,response)
            self.assertTrue(scored["target_hit"])
            other=next(x for x in row["controls"] if x["element_ref"]!=row["expected_control_id"])
            x,y,w,h=other["bounds"]
            negative=json.dumps({"point":[(x+w/2)/960,(y+h/2)/540]})
            bad=score_attempt(row,negative)
            self.assertFalse(bad["target_hit"])
            self.assertTrue(bad["wrong_target"])

    def test_tampered_png_and_partial_output_fail_closed(self):
        src=ROOT/"screenshots"/(self.scenes[0]["id"]+".png")
        modified=bytearray(src.read_bytes())
        modified[-14] ^= 1
        with self.assertRaises(ValueError):
            checked_png(bytes(modified))
        with self.assertRaises(ValueError):
            checked_png(src.read_bytes()[:24])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                freeze(Path(d),Path(d)/"not_really_rendered")

    def test_authored_html_has_no_scripts_external_resources_or_answer_markers(self):
        for scene in self.scenes:
            text=html_for_case(scene)
            self.assertNotIn("<script",text)
            self.assertNotIn("https://",text)
            self.assertNotIn("http://",text)
            self.assertNotIn("expected_control_id",text)
            self.assertNotIn("is_expected",text)
            self.assertEqual(text.count('data-control-id='),4)
            self.assertIn(f'lang="{scene["language"]}"',text)


if __name__ == "__main__":
    unittest.main()
