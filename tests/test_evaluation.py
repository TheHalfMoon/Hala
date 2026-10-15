"""Synthetic infrastructure checks; no test result qualifies a model."""
import copy
import tempfile
import unittest
from pathlib import Path

from data.factory import generate_manifest, load_frozen, manifest_bytes, validate_manifest, training_records
from evaluation.engine import Environment, OracleProvider, ControlProvider, run_case
from evaluation.metrics import summarize, wilson
from evaluation.coordinates import normalize_box, denormalize_box


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = generate_manifest()

    def gui_case(self, kind="save"):
        return copy.deepcopy(next(c for c in self.manifest["tasks"] if c["suite"] == "gui" and c["workflow"] == kind))

    def test_frozen_reproducibility_counts_strata_and_family_isolation(self):
        self.assertEqual(manifest_bytes(self.manifest), manifest_bytes(generate_manifest()))
        self.assertEqual(self.manifest, load_frozen())
        for suite, total in [("intent", 60), ("gui", 120)]:
            tasks = [c for c in self.manifest["tasks"] if c["suite"] == suite]
            self.assertEqual(len(tasks), total)
            self.assertEqual(sum(c["language"] == "ar" for c in tasks), total // 2)
        splits = {s: {c["family_id"] for c in self.manifest["tasks"] if c["split"] == s} for s in ("train", "dev", "test")}
        self.assertTrue(all(splits[a].isdisjoint(splits[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]))
        self.assertLess(len(set(c["family_id"] for c in self.manifest["tasks"])), 180)

    def test_provenance_and_split_training_deny_unknown_benchmark_and_leaks(self):
        self.assertTrue(training_records(self.manifest))
        self.assertTrue(all(c["split"] == "train" for c in training_records(self.manifest)))
        for mutate in [lambda c: c["provenance"].update(source="unknown"), lambda c: c["provenance"].update(source="benchmark_trajectory"), lambda c: c["provenance"].update(commercial_training_allowed=False)]:
            candidate = copy.deepcopy(self.manifest)
            mutate(candidate["tasks"][0])
            with self.assertRaises(ValueError):
                training_records(candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["tasks"][1]["split"] = "test"
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["tasks"].append(copy.deepcopy(candidate["tasks"][0]))
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate = copy.deepcopy(self.manifest)
        for case in candidate["tasks"]:
            if case["split"] == "test":
                case["split"] = "train"
        with self.assertRaises(ValueError):
            training_records(candidate)

    def test_label_and_coordinate_quality_rejects_corruption(self):
        candidate = copy.deepcopy(self.manifest)
        candidate["tasks"][0]["oracle_steps"][0]["target"] = "invented"
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["tasks"][0]["desired_state"]["activated"] = False
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["tasks"][0]["controls"][0]["normalized_bounds"][0] = 2
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        for scale in (1, 1.5, 2):
            box = [10 * scale, 20 * scale, 90 * scale, 30 * scale]
            self.assertEqual(denormalize_box(normalize_box(box, 800 * scale, 600 * scale), 800 * scale, 600 * scale), box)
        for box in [[-1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 801, 1], [True, 0, 1, 1]]:
            with self.assertRaises(ValueError):
                normalize_box(box, 800, 600)

    def test_state_mutates_and_done_claim_is_independently_checked(self):
        case = self.gui_case()
        result = run_case(case, OracleProvider(case))
        self.assertTrue(result["verified_success"])
        self.assertTrue(result["final_state"]["saved"])
        result = run_case(case, ControlProvider("false_done", case))
        self.assertFalse(result["verified_success"])
        self.assertTrue(result["false_done"])

    def test_wrong_target_stale_schema_controls(self):
        case = self.gui_case()
        for mode, metric in [("wrong_target", "wrong_target"), ("stale", "stale_failures"), ("invalid", "schema_failures")]:
            result = run_case(case, ControlProvider(mode, case))
            self.assertFalse(result["verified_success"])
            self.assertGreater(result[metric], 0)

    def test_executor_current_generation_and_role_not_provider_claim(self):
        case = self.gui_case()
        env = Environment(case)
        req = env.observe()
        proposal = OracleProvider(case).propose(req)
        env.advance_generation()
        with self.assertRaises(ValueError):
            env.execute(proposal, req)

    def test_wrong_text_wrong_role_and_multi_action_do_not_fake_success(self):
        from evaluation.engine import proposal_dict
        case = self.gui_case("form")
        class WrongText:
            name = "wrong_text"
            def propose(self, req):
                raw = OracleProvider(case).propose(req)
                if req.task.step_index == 0:
                    raw["actions"][0]["text"] = "wrong value"
                return raw
        self.assertFalse(run_case(case, WrongText())["verified_success"])
        for mode in ("role", "batch", "dpi"):
            env = Environment(case)
            req = env.observe()
            if mode == "role":
                raw = proposal_dict(req, {"type": "semantic_invoke", "element_ref": "field"})
            else:
                raw = OracleProvider(case).propose(req)
                if mode == "batch":
                    raw["actions"] *= 2
                else:
                    # Wrong DPI reaches the contract validator before execution.
                    control = next(c for c in case["controls"] if c["ref"] == "submit")
                    x, y, w, h = control["bounds"]
                    raw["actions"] = [{"type": "click", "element_ref_hint": "submit", "point": [x + w / 2, y + h / 2], "coordinate_space": "window", "image_scale": 0.5}]
            with self.assertRaises(ValueError):
                env.execute(raw, req)
            self.assertEqual(env.state, case["initial_state"])

    def test_manifest_hash_rejects_tampering_and_bytes_reproduce(self):
        from data.factory import freeze
        with tempfile.TemporaryDirectory() as directory:
            target = freeze(directory)
            self.assertEqual(target.read_bytes(), manifest_bytes(self.manifest))
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                load_frozen(target)

    def test_training_export_has_only_train_families_and_rejects_reports(self):
        from data.export import export_training
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            info = export_training(self.manifest, path)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(info["records"], 116)
            self.assertEqual(info["family_count"], 8)
            self.assertTrue(all(c["split"] == "train" for c in records))
            eval_families = {c["family_id"] for c in self.manifest["tasks"] if c["split"] != "train"}
            self.assertFalse(eval_families.intersection(c["family_id"] for c in records))
            with self.assertRaises(ValueError):
                export_training({"schema": "evaluation-report", "cases": []}, path)

    def test_provider_mutation_cannot_rewrite_verifier_snapshot_or_budget(self):
        from evaluation.engine import proposal_dict
        case = self.gui_case("save")
        original_id = case["task_id"]
        class MutatingProvider:
            name = "mutates_caller"
            def propose(self, request):
                case["desired_state"] = copy.deepcopy(case["initial_state"])
                case.update(max_steps=60, task_id="forged-task", family_id="forged-family")
                return proposal_dict(request, {"type": "done_proposed"}, "verification_recommended")
        result = run_case(case, MutatingProvider())
        self.assertFalse(result["verified_success"])
        self.assertTrue(result["false_done"])
        self.assertFalse(result["final_state"]["saved"])
        self.assertEqual(result["task_id"], original_id)
        self.assertEqual(result["family_id"], "workflow-save")
        case = self.gui_case("save")
        admitted_budget = case["max_steps"]
        class MutatesBudget:
            name = "mutates_budget"
            def propose(self, request):
                case["max_steps"] = 60
                return proposal_dict(request, {"type": "observe"})
        result = run_case(case, MutatesBudget())
        self.assertEqual(result["steps"], admitted_budget)
        self.assertTrue(result["budget_exhausted"])

    def test_training_export_rejects_heldout_identity_relabel_and_source_drift(self):
        from data.export import export_training
        for mode in ("relabel", "fully_relabel", "goal_drift", "unknown_id"):
            candidate = copy.deepcopy(self.manifest)
            if mode in ("relabel", "fully_relabel"):
                case = next(c for c in candidate["tasks"] if c["suite"] == "gui" and c["split"] == "test")
                case.update(workflow="save", family_id="workflow-save", split="train")
                if mode == "fully_relabel":
                    case["task_id"] = "gui-save-00-999998"
                    case["initial_state"]["document"] = case["task_id"]
                    case["desired_state"]["document"] = case["task_id"]
            else:
                case = candidate["tasks"][0]
                if mode == "goal_drift":
                    case["goal"] += " and send the result"
                else:
                    case["task_id"] = "intent-save-00-999998"
                    case["initial_state"]["document"] = case["task_id"]
                    case["desired_state"]["document"] = case["task_id"]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    export_training(candidate, Path(directory) / "train.jsonl")
        alternate_source = generate_manifest(seed=7)
        self.assertEqual(alternate_source, generate_manifest(seed=7))
        self.assertEqual(len(training_records(alternate_source)), 116)

    def test_annotations_include_transient_attempts_within_declared_budget(self):
        for workflow, budget in [("form", 1), ("recovery", 2)]:
            candidate = copy.deepcopy(self.manifest)
            case = next(c for c in candidate["tasks"] if c["suite"] == "gui" and c["workflow"] == workflow)
            case["max_steps"] = budget
            with self.subTest(workflow=workflow):
                with self.assertRaisesRegex(ValueError, "budget"):
                    validate_manifest(candidate)

    def test_incidental_whitespace_unicode_goal_duplicates_rejected(self):
        import unicodedata
        for mutate in (lambda s: " " + s + " ", lambda s: s.replace(" ", "   "), lambda s: s.replace(" ", "\u00a0"), lambda s: "".join(chr(ord(ch) + 0xfee0) if "!" <= ch <= "~" else ch for ch in s)):
            candidate = copy.deepcopy(self.manifest)
            duplicate = copy.deepcopy(candidate["tasks"][0])
            duplicate["task_id"] = "intent-save-01-999998"
            duplicate["goal"] = mutate(duplicate["goal"])
            candidate["tasks"].append(duplicate)
            with self.assertRaisesRegex(ValueError, "Duplicate semantic"):
                validate_manifest(candidate)
        # Arabic canonical composition also shares the same semantic fingerprint.
        candidate = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(next(c for c in candidate["tasks"] if c["language"] == "ar" and c["workflow"] == "archive"))
        duplicate["task_id"] = "intent-archive-05-999998"
        duplicate["goal"] = unicodedata.normalize("NFD", duplicate["goal"])
        candidate["tasks"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate semantic"):
            validate_manifest(candidate)

    def test_multistep_recovery_and_budget(self):
        for workflow in ("drag_drop", "recovery", "form", "scroll_confirm"):
            case = self.gui_case(workflow)
            result = run_case(case, OracleProvider(case))
            self.assertTrue(result["verified_success"], result)
            self.assertGreater(result["steps"], 2)
            case["max_steps"] = 1
            result = run_case(case, OracleProvider(case))
            self.assertFalse(result["verified_success"])
            self.assertLessEqual(result["steps"], 1)

    def test_all_oracle_tasks_are_baseline_only_and_metrics_are_bounded(self):
        results = [run_case(c, OracleProvider(c)) for c in self.manifest["tasks"]]
        summary = summarize(results)
        self.assertEqual(summary["sample_count"], 180)
        self.assertEqual(summary["verified_success_count"], 180)
        self.assertEqual(summary["claim"], "synthetic infrastructure baseline only; no real model quality measurement")
        self.assertEqual(set(summary["language_strata"]), {"en", "ar"})
        self.assertEqual(wilson(0, 0), [None, None])
        self.assertGreater(wilson(180, 180)[0], 0.95)

    def test_optional_png_is_actual_image_and_reports_rtl_capability(self):
        from evaluation.render import render_case
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow optional")
        case = next(c for c in self.manifest["tasks"] if c["language"] == "ar")
        with tempfile.TemporaryDirectory() as directory:
            result = render_case(case, Path(directory) / "frame.png")
            from PIL import Image
            with Image.open(result["path"]) as image:
                self.assertEqual(image.size, tuple(case["frame"][k] for k in ("width", "height")))
                self.assertEqual(image.format, "PNG")
            self.assertIn("rtl_shaping_supported", result)


if __name__ == "__main__":
    unittest.main()
