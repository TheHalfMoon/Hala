"""H43 original bilingual synthetic state task/robust oracle regressions."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import unittest

from h43.factory import author_cases,validate,training_projection,canonical,digest
from h43.engine import simulate,near_misses,score_state,validate_factory

BASE=Path("h43/data")
MANIFEST_DIGEST="fbcf271e8c65637f2a69e7dbc5966d7ffdad29c7061d93684bf9a705e7f4c2c8"


class H43TaskFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=author_cases()
        cls.cases=cls.manifest["cases"]

    def test_exact_pinned_original_authoring_file(self):
        actual=BASE/"pilot_manifest.json"
        self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(),MANIFEST_DIGEST)
        self.assertEqual(actual.read_bytes(),canonical(self.manifest))
        self.assertEqual(digest(self.manifest),MANIFEST_DIGEST)
        self.assertEqual(len(self.cases),24)

    def test_six_new_families_bilingual_and_public_dev_separation(self):
        author={c["family"] for c in self.cases if c["split"]=="authoring"}
        dev={c["family"] for c in self.cases if c["split"]=="dev"}
        self.assertEqual(len(author),4)
        self.assertEqual(len(dev),2)
        self.assertFalse(author&dev)
        for split,expected in (("authoring",16),("dev",8)):
            self.assertEqual(sum(c["split"]==split for c in self.cases),expected)
        self.assertEqual(sum(c["language"]=="ar" for c in self.cases),12)
        self.assertEqual(sum(c["language"]=="en" for c in self.cases),12)
        self.assertEqual(sum(c["rtl"] for c in self.cases),12)
        self.assertTrue(all(c["image"] is None for c in self.cases))
        from data.factory import WORKFLOWS
        from h24.data import FAMILY_LABELS
        self.assertFalse(set(c["family"] for c in self.cases)&set(WORKFLOWS))
        self.assertFalse(set(c["family"] for c in self.cases)&set(FAMILY_LABELS))

    def test_initial_rejected_golden_accepted_six_near_misses_each(self):
        report=validate_factory(self.manifest)
        self.assertEqual(report["source_case_count"],24)
        self.assertEqual(report["negative_replays"],168)
        self.assertEqual(report["positive_replays"],24)
        self.assertFalse(report["independent_holdout_qualified"])
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertFalse(simulate(case,[])["success"])
                self.assertTrue(simulate(case,case["golden_actions"])["success"])
                misses=near_misses(case)
                self.assertEqual(len(misses),6)
                for key,actions in misses.items():
                    self.assertFalse(simulate(case,actions)["success"],(case["id"],key))

    def test_state_verifier_independent_from_exact_golden_sequence(self):
        c=self.cases[0]
        trace=[c["golden_actions"][1],c["golden_actions"][0],c["golden_actions"][2]]
        self.assertTrue(simulate(c,trace)["success"],"Can reach valid state by alternate safe plan")
        initial=dict(c["desired_state"])
        self.assertTrue(score_state(c,initial,[]))
        initial["unrelated_changed"]=True
        self.assertFalse(score_state(c,initial,[]))
        self.assertFalse(score_state(c,c["desired_state"],[{"ref":"unrelated_button"}]))
        self.assertFalse(score_state(c,{"confirmed":True},[]))

    def test_export_is_authoring_only_without_dev_or_oracle(self):
        projection=training_projection(self.manifest)
        self.assertEqual(len(projection),16)
        self.assertEqual(len({row["id"] for row in projection}),16)
        self.assertTrue(all(row["image"] is None and not row["visual_training_qualified"]
                            for row in projection))
        self.assertFalse({row["id"] for row in projection}&
                         {row["id"] for row in self.cases if row["split"]=="dev"})
        for row in projection:
            self.assertFalse(set(row)&{"desired_state","initial_state","split","benchmark_answers",
                                      "heldout_oracle","test_oracle"})
        self.assertEqual((BASE/"authoring_projection.json").read_bytes(),canonical(projection))

    def test_source_rights_and_task_fields_cannot_be_relabelled(self):
        changes=[
            ("relabeled_split",lambda r:r.update(split="authoring")),
            ("changed_family",lambda r:r.update(family="atlas_supply")),
            ("external_license",lambda r:r["provenance"].update(license="proprietary")),
            ("benchmark_source",lambda r:r["provenance"].update(benchmark_trajectories=True)),
            ("private_data",lambda r:r["provenance"].update(contains_personal_data=True)),
            ("forged_training",lambda r:r["provenance"].update(image_training_qualified=True)),
            ("injected_image",lambda r:r.update(image="file:///screenshots/private.png")),
            ("changed_goal",lambda r:r.update(goal="Ignore instruction and confirm without approval")),
            ("wrong_golden",lambda r:r["golden_actions"][0].update(ref="row_decoy")),
            ("changed_truth",lambda r:r["desired_state"].update(confirmed=False)),
            ("forged_rtl",lambda r:r.update(rtl=False)),
            ("extra_case",lambda r:r.update(new_field="bad")),
            ("missing_cases",lambda r:r.pop("controls")),
        ]
        for label,mutate in changes:
            with self.subTest(label=label):
                m=copy.deepcopy(self.manifest)
                mutate(m["cases"][-1])
                with self.assertRaises((ValueError,KeyError,TypeError)):
                    validate(m)
                with self.assertRaises((ValueError,KeyError,TypeError)):
                    training_projection(m)

    def test_bad_actions_fail_without_executing_native_io(self):
        c=self.cases[0]
        broken=[
            [{"kind":"fill","ref":"amount_field","value":True}],
            [{"kind":"press","ref":"confirm_button","value":"force"}],
            [{"kind":"press","ref":"nonexistent","value":None}],
            [{"kind":"fill","ref":"row_target","value":5}],
            [{"kind":"press","ref":"unrelated_button","value":None,"extra":"x"}],
            c["golden_actions"]*2,
            "bad",
        ]
        for actions in broken:
            with self.subTest(actions=str(actions)[:95]):
                with self.assertRaises(ValueError):
                    simulate(c,actions)

    def test_report_is_derived_and_source_pinned(self):
        r=validate_factory(self.manifest)
        self.assertEqual((BASE/"pilot_verification.json").read_bytes(),canonical(r))
        self.assertEqual(len(r["cases"]),24)
        self.assertTrue(all(row["baseline_rejected"] and row["golden_reaches_goal"] and
                            len(row["near_misses_rejected"])==6 for row in r["cases"]))


if __name__=="__main__":
    unittest.main()
