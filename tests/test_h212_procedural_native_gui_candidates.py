"""H212 original synthetic-only GUI candidate provenance and SFT contract tests."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, parse_actions,
    training_messages_for_synthetic_contract, verify_real_training_release,
)
from h212.procedural_native_gui_candidates import (
    COUNT, INSTRUCTIONS, SCHEMA, SPLITS, candidate_manifest,
    enumerate_specs, render_case, _rectangles,
)


class ProceduralCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = candidate_manifest()

    def test_explicit_new_source_identity_and_role_counts(self):
        doc = self.manifest
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(COUNT, 96)
        self.assertEqual(len(doc["cases"]), 96)
        self.assertEqual(len(doc["generator_source_sha256"]),64)
        self.assertEqual(doc["role_counts"],{
            "RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED":64,
            "RESEARCH_DEV_CANDIDATE_NOT_SEALED":16,
            "RESEARCH_EXTRA_CANDIDATE_NOT_OOD":16,
        })
        self.assertFalse(doc["official_admitted"])
        self.assertFalse(doc["dev_gold_sealed"])
        self.assertFalse(doc["official_ood_evaluated"])
        self.assertFalse(doc["historical_source_history_complete"])

    def test_frozen_research_source_manifest_matches_versioned_code(self):
        from pathlib import Path
        from h212 import procedural_native_gui_candidates
        source = Path(procedural_native_gui_candidates.__file__)
        expected = sha256(source.read_text(encoding="utf-8").replace(
            "\r\n", "\n").encode("utf-8")).hexdigest()
        frozen = json.loads((source.parent /
            "RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, frozen["generator_source_sha256"])
        self.assertEqual(len(frozen["cases"]),96)
        self.assertEqual(frozen["registered_candidate_count"],96)
        self.assertEqual(frozen["role_counts"],self.manifest["role_counts"])
        self.assertFalse(frozen["official_admitted"])
        self.assertFalse(frozen["dev_gold_sealed"])
        self.assertFalse(frozen["official_ood_evaluated"])
        for row in frozen["cases"]:
            self.assertEqual(len(row["image_png_sha256"]),64)
            self.assertEqual(len(row["image_decoded_rgba_sha256"]),64)
            self.assertEqual(set(row),{
                "index","family","role","language",
                "image_png_sha256","image_decoded_rgba_sha256",
            })

    def test_entire_roster_unique_visual_hashes_and_role_isolation(self):
        doc = self.manifest
        self.assertEqual(len({c["image_png_sha256"] for c in doc["cases"]}),96)
        self.assertEqual(len({c["image_decoded_rgba_sha256"] for c in doc["cases"]}),96)
        self.assertEqual([c["index"] for c in doc["cases"]], list(range(96)))
        role_by_family = {}
        for c in doc["cases"]:
            if c["family"] in role_by_family:
                self.assertEqual(c["role"], role_by_family[c["family"]])
            role_by_family[c["family"]] = c["role"]
        self.assertEqual(len(role_by_family),4)

    def test_bilingual_intents_across_every_family(self):
        groups={}
        for c in self.manifest["cases"]:
            groups.setdefault(c["family"],Counter()).update([c["language"]])
        for group in groups.values():
            self.assertEqual(set(group), {"en","ar"})
            self.assertEqual(group["en"],group["ar"])
        self.assertIn("اضغط",INSTRUCTIONS["ar"])
        self.assertIn("Click",INSTRUCTIONS["en"])

    def test_sha256_of_actual_png_and_decoded_rgba(self):
        from PIL import Image
        for i in (0,1,30,31,32,63,64,65,79,80,81,95):
            c = render_case(i)
            with Image.open(BytesIO(c.image_png)) as img:
                self.assertEqual(img.format,"PNG")
                self.assertEqual(img.size,(960,720))
                rgba=img.convert("RGBA").tobytes()
            self.assertEqual(sha256(c.image_png).hexdigest(),c.image_sha256)
            self.assertEqual(sha256(rgba).hexdigest(),c.visual_rgba_sha256)

    def test_label_in_target_and_decoy_on_opposite_side(self):
        for index, _role, family, variant in enumerate_specs():
            c=render_case(index)
            raw=parse_actions(c.action_json, training_target=True)
            point=raw["actions"][0]["point"]
            left,top,right,bottom=c.source_bbox
            self.assertTrue(left<=point[0]<=right and top<=point[1]<=bottom)
            target, decoy = _rectangles(family,variant)
            self.assertEqual(target,c.source_bbox)
            a,b,cx,dy=decoy
            self.assertFalse(a<=point[0]<=cx and b<=point[1]<=dy)

    def test_model_visible_prompt_has_no_answer_coordinates_or_split_labels(self):
        for index in (0,1,31,32,64,65,80,95):
            c=render_case(index)
            q,a = training_messages_for_synthetic_contract(
                c.instruction,c.action_json)
            self.assertEqual(q[0]["content"][0],{"type":"image"})
            self.assertNotIn("CANDIDATE",q[0]["content"][-1]["text"])
            self.assertNotIn(c.action_json,q[0]["content"][-1]["text"])
            self.assertEqual(a[-1]["role"],"assistant")
            self.assertEqual(a[-1]["content"][0]["text"],
                             c.action_json)

    def test_actual_pil_multimodal_messages_are_ready_but_unadmitted(self):
        from h212.procedural_native_gui_candidates import research_multimodal_messages
        for index in (0, 1, 32, 33, 64, 65, 80, 81):
            with self.subTest(index=index):
                q, a = research_multimodal_messages(index)
                self.assertEqual(len(q), 1)
                self.assertEqual(len(a), 2)
                self.assertIs(a[0], q[0])
                picture = q[0]["content"][0]["image"]
                self.assertEqual(picture.mode, "RGB")
                self.assertEqual(picture.size, (960,720))
                self.assertEqual(q[0]["content"][0]["type"], "image")
                self.assertEqual(q[0]["content"][1]["type"], "text")
                self.assertTrue(q[0]["content"][1]["text"].startswith("You are Hala"))
                self.assertEqual(a[1]["role"], "assistant")
                parsed = parse_actions(
                    a[1]["content"][0]["text"], training_target=True)
                self.assertEqual(parsed["actions"][0]["kind"],"click")
                self.assertFalse(self.manifest["official_admitted"])

    def test_multiple_renders_bitwise_deterministic(self):
        for n in (0,17,31,32,64,95):
            a=render_case(n)
            b=render_case(n)
            self.assertEqual(a.image_png,b.image_png)
            self.assertEqual(a.visual_rgba_sha256,b.visual_rgba_sha256)
            self.assertEqual(a.action_json,b.action_json)

    def test_bad_case_ids_fail_closed(self):
        for n in (-1,96,"1",1.0,True,None):
            with self.subTest(n=repr(n)):
                with self.assertRaises(ValueError):
                    render_case(n)

    def test_official_training_gate_cannot_be_bypassed_by_generated_source(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        fake={
            "schema":"hala.original-native-train-release.v1",
            "manifest_sha256": self.manifest["generator_source_sha256"],
            "split":"TRAIN", "official_admitted":True,
            "complete_source_history_proven":True,
            "complete_old_rgba_proven":True,
            "sealed_dev_gold":True, "consumed_ood_reused":False,
        }
        with self.assertRaises(ValueError):
            verify_real_training_release(fake)

    def test_metadata_does_not_contain_gold_labels_or_proprietary_media(self):
        doc=self.manifest
        compact=json.dumps(doc,sort_keys=True)
        for c in doc["cases"]:
            self.assertEqual(set(c),{
                "index","family","role","language",
                "image_png_sha256","image_decoded_rgba_sha256",
            })
        self.assertNotIn('"action"',compact)
        self.assertNotIn('"point"',compact)
        self.assertNotIn("owner-root",compact)


if __name__=="__main__":
    unittest.main()
