"""H25 admission hardening: no private oracle state, source rank or test into LLM."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h25 import corpus as base
from h25 import projection_v2 as v2


class H25ProjectionV2Tests(unittest.TestCase):
    def setUp(self):
        self.value = v2.candidate()
        self.refs = base.corpus()

    def test_exact_reviewed_source_regeneration_and_split(self):
        manifest = v2.validate(self.value)
        self.assertEqual(manifest["rows"], 42)
        self.assertEqual(manifest["languages"], {"en": 21, "ar": 21})
        self.assertEqual(manifest["source_cases_sha256"], v2.H25_PIN)
        self.assertEqual({r["source_ref"]["family"] for r in self.value["rows"]},
                         set(base.SPLITS["dev"]))

    def test_model_input_contains_no_bookkeeping_or_class_indicators(self):
        for r in self.value["rows"]:
            x = r["model_input"]
            self.assertEqual(set(x), {"goal", "locale", "observation"})
            y = x["observation"]
            self.assertEqual(set(y), {"observed_frame","current_frame","controls","untrusted_page_text"})
            serialized=json.dumps(x,ensure_ascii=False)
            for prohibited in ('"oracle_private"', '"source_ref"', '"state"',
                               '"expected"', '"unsafe_user_intent"',
                               '"required_parameter_present"',
                               '"conflicting_evidence_present"',
                               '"postcondition_verified"', '"partition"'):
                self.assertNotIn(prohibited,serialized)
            self.assertNotIn(r["source_ref"]["id"],serialized)
            self.assertNotIn(r["source_ref"]["family"],serialized)

    def test_stale_is_detectable_by_inequality_not_embedded_word(self):
        original={x["id"]:x for x in self.refs["cases"]}
        stale=0
        for row in self.value["rows"]:
            case=original[row["source_ref"]["id"]]
            obs=row["model_input"]["observation"]
            different=obs["observed_frame"]!=obs["current_frame"]
            stale+=different
            self.assertEqual(different,case["oracle_private"]["state"]=="stale")
            self.assertFalse(any(tag in obs["observed_frame"] for tag in ("stale","unsafe","dev",case["family"])))
            self.assertEqual(len(obs["observed_frame"]),20)
        self.assertEqual(stale, 6)

    def test_controls_describe_meaningful_distinct_targets_not_verbs(self):
        for row in self.value["rows"]:
            family=row["source_ref"]["family"]
            lang=row["source_ref"]["language"]
            controls=row["model_input"]["observation"]["controls"]
            self.assertEqual([c["label"] for c in controls],list(v2.TEXT[family][lang]))
            self.assertNotEqual(controls[0]["label"],controls[1]["label"])
            self.assertNotIn(controls[0]["label"].casefold(),{"find","locate","اعثر","راجع"})
            self.assertTrue(all(set(c)=={"role","label","bounds"} for c in controls))

    def test_english_arabic_goals_and_untrusted_text_preserved(self):
        case_by_id={c["id"]:c for c in self.refs["cases"]}
        for r in self.value["rows"]:
            x=r["model_input"]
            original=case_by_id[r["source_ref"]["id"]]
            self.assertEqual(x["goal"],original["goal"])
            self.assertEqual(x["observation"]["untrusted_page_text"],
                             original["observed"]["page_untrusted_text"])
            if original["language"]=="ar":
                self.assertGreater(sum(0x600<=ord(c)<=0x6ff for c in x["goal"]),12)

    def test_reject_tampered_references_labels_and_prompts(self):
        changes=[
            lambda v: v["rows"][0]["model_input"]["observation"].update({"protected":True}),
            lambda v: v["rows"][0]["source_ref"].update({"family":"library_lending"}),
            lambda v: v["rows"][0]["model_input"].update({"expected":{"tool":"visual"}}),
            lambda v: v["rows"][1]["model_input"]["observation"].update({"current_frame":"stale"}),
        ]
        for mutate in changes:
            candidate=copy.deepcopy(self.value)
            mutate(candidate)
            with self.assertRaises(ValueError):
                v2.validate(candidate)

    def test_no_training_or_test_qualification_in_manifest(self):
        receipt=v2.validate(self.value)
        self.assertTrue(receipt["no_test_qualification"] and receipt["no_gpu_run"])
        self.assertTrue(receipt["no_native_orcel_execution"])
        self.assertFalse(self.value["model_proposals_authorized"])
        self.assertFalse(self.value["heldout_test"])

    def test_cli_create_only_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"candidate"
            with patch("sys.argv",["projection_v2","--output",str(out)]):
                v2.main()
            canonical=base.canonical(self.value)
            self.assertEqual((out/"projection.json").read_bytes(),canonical)
            self.assertEqual(json.loads((out/"manifest.json").read_text(encoding="utf8")),
                             v2.validate(self.value))
            self.assertEqual((out/"model-inputs.jsonl").read_bytes(),
                             b"".join(base.canonical(x["model_input"]) for x in self.value["rows"]))
            with patch("sys.argv",["projection_v2","--output",str(out)]):
                with self.assertRaises(FileExistsError):
                    v2.main()

    def test_no_changes_to_original_h25_archive(self):
        archive=Path(__file__).resolve().parents[1]/"h25"/"frozen"
        self.assertEqual(hashlib.sha256((archive/"cases.json").read_bytes()).hexdigest(),v2.H25_PIN)
        self.assertEqual((archive/"cases.json").read_bytes(),base.canonical(self.refs))


if __name__=="__main__":
    unittest.main()
