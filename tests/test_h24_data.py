"""H24 fail-closed provenance, family isolation and prompt safety regression."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h24 import data as h24


class H24DataTests(unittest.TestCase):
    def setUp(self):
        self.frozen = h24.dataset()

    def test_exact_deterministic_authoring(self):
        self.assertEqual(h24.canonical(self.frozen), h24.canonical(h24.dataset()))
        self.assertEqual(h24.fingerprint(self.frozen), h24.fingerprint(h24.dataset()))

    def test_100_bilingual_cases_and_family_isolated_splits(self):
        report = h24.validate(self.frozen)
        self.assertEqual(report["counts"], {"train": 60, "dev": 20, "test": 20})
        self.assertEqual(report["languages"], {"en": 50, "ar": 50})
        self.assertEqual({x["family"] for x in self.frozen["cases"] if x["split"] == "dev"},
                         set(h24.SPLITS["dev"]))
        self.assertFalse(set(h24.SPLITS["train"]) & set(h24.SPLITS["test"]))
        self.assertFalse(set(h24.SPLITS["dev"]) & set(h24.SPLITS["test"]))

    def test_all_four_actions_for_each_language_and_split(self):
        for split in h24.SPLITS:
            for language in ("en", "ar"):
                kinds = {x["expected"]["tool"] for x in self.frozen["cases"]
                         if x["split"] == split and x["language"] == language}
                self.assertEqual(kinds, {"visual", "observe", "clarify", "abstain"})

    def test_english_arabic_rtl_mirrors_protected_controls(self):
        cases = self.frozen["cases"]
        for c in cases:
            x = c["observation"]["controls"][0]["bounds"][0]
            self.assertEqual(x, 460 if c["language"] == "ar" else 90)
            self.assertFalse(c["observation"]["controls"][0]["protected"])
            self.assertTrue(c["observation"]["controls"][1]["protected"])

    def test_only_training_mind_rows_exported(self):
        rows = h24.prospective_rows(self.frozen)
        self.assertEqual(len(rows), 60)
        self.assertEqual({r["component"] for r in rows}, {"mind"})
        self.assertEqual({r["language"] for r in rows}, {"en", "ar"})
        self.assertFalse({r["id"] for r in rows} &
                         {r["id"] for r in self.frozen["cases"] if r["split"] != "train"})
        self.assertFalse({r["family"] for r in rows} & set(h24.SPLITS["dev"]))
        self.assertFalse({r["family"] for r in rows} & set(h24.SPLITS["test"]))
        self.assertTrue(all(r["image"] is None and r["image_sha256"] is None for r in rows))
        for r in rows:
            self.assertNotIn('"expected"', r["prompt"])
            self.assertNotIn('"completion"', r["prompt"])
            self.assertNotIn("source_revision", r["prompt"])
            self.assertNotIn('"split"', r["prompt"])

    def test_goal_and_text_normalization(self):
        self.assertEqual(h24.norm_text("  \u0645\u0644\u0641\u3000\u062c\u062f\u064a\u062f  "), h24.norm_text("\u0645\u0644\u0641 \u062c\u062f\u064a\u062f"))
        self.assertEqual(h24.norm_text("\uff21\uff22\uff23"), h24.norm_text("abc"))

    def test_reject_training_on_dev_or_test_via_relabeling(self):
        for which in ("split", "family"):
            changed = copy.deepcopy(self.frozen)
            target = next(x for x in changed["cases"] if x["split"] == "test")
            target[which] = "train" if which == "split" else h24.SPLITS["train"][0]
            with self.subTest(which=which), self.assertRaises(ValueError):
                h24.prospective_rows(changed)

    def test_reject_unauthorized_training_flags(self):
        for name, value in (("image_qualified", True), ("training_type_eligible", True),
                            ("native_qualified", True)):
            changed = copy.deepcopy(self.frozen)
            changed["cases"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                h24.validate(changed)

    def test_reject_forged_provenance(self):
        for key, value in (("license", "unknown"), ("commercial_training_allowed", False),
                           ("benchmark_trajectories", True), ("contains_personal_data", True)):
            changed = copy.deepcopy(self.frozen)
            changed["cases"][0]["provenance"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                h24.validate(changed)

    def test_reject_changed_image_or_prompt_injection_labels(self):
        for key, value in (("image", "/tmp/screenshot.png"), ("untrusted_page_text", "trust me")):
            changed = copy.deepcopy(self.frozen)
            changed["cases"][0]["observation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                h24.validate(changed)
        changed = copy.deepcopy(self.frozen)
        changed["cases"][-1]["expected"]["tool"] = "visual"
        with self.assertRaises(ValueError):
            h24.validate(changed)

    def test_no_duplicate_case_or_semantic_goals(self):
        ids = [x["id"] for x in self.frozen["cases"]]
        goals = [h24.norm_text(x["goal"]) for x in self.frozen["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(goals), len(set(goals)))

    def test_hostile_page_cannot_override_legitimate_user(self):
        hostile = [x for x in self.frozen["cases"] if x["variant"] == "hostile_page"]
        self.assertEqual(len(hostile), 20)
        self.assertTrue(all(x["observation"]["untrusted_page_text"] and
                            x["expected"]["tool"] == "observe" and
                            x["expected"]["subgoal"] == "ignore_page_instruction" for x in hostile))

    def test_unsafe_user_request_must_abstain(self):
        unsafe = [x for x in self.frozen["cases"] if x["variant"] == "unsafe_request"]
        self.assertEqual(len(unsafe), 20)
        self.assertTrue(all(not x["observation"]["untrusted_page_text"] and
                            x["expected"]["tool"] == "abstain" and
                            x["expected"]["subgoal"] == "reject_external_command" for x in unsafe))

    def test_stale_frame_must_refresh(self):
        stale = [x for x in self.frozen["cases"] if x["variant"] == "stale_frame"]
        self.assertEqual(len(stale), 20)
        self.assertTrue(all(x["observation"]["frame_id"] !=
                            x["observation"]["expected_current_frame"] and
                            x["expected"]["tool"] == "observe" for x in stale))

    def test_ambiguity_must_clarify(self):
        ambiguous = [x for x in self.frozen["cases"] if x["variant"] == "missing_destination"]
        self.assertEqual(len(ambiguous), 20)
        self.assertTrue(all(x["observation"]["destination"] is None and
                            x["expected"]["tool"] == "clarify" for x in ambiguous))

    def test_protected_target_not_selected(self):
        normal = [x for x in self.frozen["cases"] if x["variant"] == "valid"]
        self.assertEqual(len(normal), 20)
        self.assertTrue(all(x["expected"]["subgoal"] == "locate_verified_target" and
                            x["observation"]["controls"][0]["protected"] is False for x in normal))

    def test_no_historical_family_reuse(self):
        from development.data import FAMILIES as H19
        from evaluation.heldout_fixtures import FAMILIES as H14
        from data.factory import WORKFLOWS as H0
        prior = {f[0] for f in H19} | set(H14) | {"workflow-" + f for f in H0}
        current = set(h24.FAMILY_LABELS)
        self.assertFalse(current & prior)

    def test_cli_only_new_output_no_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-h24"
            h24_output = h24.dataset()
            report = h24.validate(h24_output)
            target.mkdir()
            (target / "manifest.json").write_bytes(h24.canonical(report))
            (target / "cases.json").write_bytes(h24.canonical(h24_output))
            raw = target.joinpath("cases.json").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), report["sha256"])
            self.assertEqual(len(h24.prospective_rows(h24_output)), 60)


if __name__ == "__main__":
    unittest.main()
