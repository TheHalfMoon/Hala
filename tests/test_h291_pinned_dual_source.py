"""H291 SOURCE-ONLY security checks; no model, logs, GPU or blind OOD."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from h291.private_dual_source_lora import (
    BRANCH, PIN, EVAL_SOURCE_BRANCH, EVAL_SOURCE_SHA, SHIFT_BLOB,
    QUEUE_BLOB, FLAG, DIAGNOSTIC_REPORT, get_pinned_source,
    failure_receipt, verified_report, QUEUE_SCENES, N_STEPS, MODEL,
    MODEL_REV,
)
from h291.build_private_payload import build, KERNEL_ID, KERNEL_TITLE
from h291.verify_stage_diagnostic import verify_bytes as verify_fail
from h288.verify_private_lora_receipt import verify_bytes as verify_success


def dumps(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


class H291PinnedDualOriginalSources(TestCase):
    def test_distinct_actual_source_git_revisions_and_original_blobs(self):
        self.assertEqual(PIN,
            "76a4ebb204d825b280fd4d332ad75591c5f13007")
        self.assertEqual(BRANCH,
            "ps/h286-relative-train-label-alignment")
        self.assertEqual(EVAL_SOURCE_SHA,
            "c61409e0b0178227180f0d8e05c67f5a210d01f9")
        self.assertEqual(EVAL_SOURCE_BRANCH,
            "ps/h278-queueboard-task-target-contract")
        self.assertNotEqual(PIN, EVAL_SOURCE_SHA)
        self.assertEqual(SHIFT_BLOB,
            "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac")
        self.assertEqual(QUEUE_BLOB,
            "3cf452e84a75b017905bc14b04a8571411929891")
        self.assertNotEqual(SHIFT_BLOB, QUEUE_BLOB)
        self.assertEqual(FLAG, "HALA_H291_PRIVATE_NONCOMMERCIAL_FREE_T4")

    def test_dual_checkout_requires_two_pinned_source_families(self):
        # INVENTED synthetic files only: live clones are separately tested
        # on owner Windows from actual GitHub, not in offline unit CI.
        seen = []
        def mocked_clone(branch, rev, target):
            seen.append((branch, rev, target.name))
            if target.name == "train_h286":
                path = target / "h285" / "shiftdesk.html"
            else:
                path = target / "h275" / "queueboard.html"
                mod = target / "h276" / "queueboard_native_research.py"
                mod.parent.mkdir(parents=True)
                mod.write_text("author_owned_fake_for_unit_test=True")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"non-real byte fixture")
        def mocked_blob(p):
            return SHIFT_BLOB if p.name == "shiftdesk.html" else QUEUE_BLOB
        import h291.private_dual_source_lora as driver
        with TemporaryDirectory() as temp:
            with mock.patch.object(driver, "_verified_clone", side_effect=mocked_clone),                  mock.patch.object(driver, "_git_blob", side_effect=mocked_blob):
                train, eval = get_pinned_source(Path(temp) / "source")
            self.assertEqual(train.parent.name, "h285")
            self.assertEqual(eval.parent.name, "h275")
            self.assertNotEqual(train.parents[1], eval.parents[1])
            self.assertEqual(seen, [
                (BRANCH, PIN, "train_h286"),
                (EVAL_SOURCE_BRANCH, EVAL_SOURCE_SHA, "eval_h278"),
            ])

    def test_missing_independent_source_module_fails_closed(self):
        def simulated_clone(branch, rev, target):
            relative = ("h285/shiftdesk.html" if target.name == "train_h286"
                        else "h275/queueboard.html")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic")
        import h291.private_dual_source_lora as driver
        with TemporaryDirectory() as temp:
            with mock.patch.object(driver, "_verified_clone", side_effect=simulated_clone),                  mock.patch.object(driver, "_git_blob",
                                   side_effect=lambda p: SHIFT_BLOB if p.name
                                   == "shiftdesk.html" else QUEUE_BLOB):
                with self.assertRaisesRegex(ValueError, "independent source-owned oracle"):
                    get_pinned_source(Path(temp) / "two_repo")

    def test_failure_stage_includes_eval_source_but_no_fake_score(self):
        self.assertEqual(DIAGNOSTIC_REPORT,
                         "hala-h291-private-lora-stage-diagnostic.json")
        doc = failure_receipt("EVAL_SOURCE", "ValueError")
        accepted = verify_fail(dumps(doc))
        self.assertEqual(accepted["failure_stage"], "EVAL_SOURCE")
        self.assertFalse(accepted["verified_paired_gui_scores"])
        self.assertEqual(accepted["verified_optimizer_steps"], 0)
        self.assertFalse(accepted["generalizable_gain_proven"])

    def test_success_emitter_still_independent_h288_closed_schema(self):
        def rows(v):
            return [{
                "scene": scene[0], "model_generation": 1, "grammar_call": 1,
                "strict": True, "kind": "click", "native_click": True,
                "task_success": bool(v[i]),
            } for i, scene in enumerate(QUEUE_SCENES)]
        r = verified_report(
            [{"changed": 2, "coordinates_supervised": True,
              "loss_finite": True, "gradient_nonzero": True}
             for _ in range(N_STEPS)],
            rows((False, False)), rows((True, False)),
            ("a" * 64, "b" * 64), ("a" * 64, "b" * 64),
            {"free_t4": True, "devices": 1},
            frozenset(str(i).zfill(64) for i in range(24)),
        )
        self.assertEqual(verify_success(dumps(r))["paired_delta"], 1)

    def test_source_only_two_file_private_kaggle_payload_ast(self):
        src = Path(__file__).resolve().parents[1] / "h291/private_dual_source_lora.py"
        with TemporaryDirectory() as temp:
            out = Path(temp) / "private"
            doc = build(src, out)
            self.assertEqual(doc["uploaded_media_weights_or_gold"], 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()), [
                "h291_private_dual_source_lora.py", "kernel-metadata.json"])
            meta = json.loads((out / "kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["id"], KERNEL_ID)
            self.assertEqual(meta["title"], KERNEL_TITLE)
            self.assertEqual(meta["dataset_sources"], [])
            self.assertEqual(meta["model_sources"], [])
            python = (out / "h291_private_dual_source_lora.py").read_text("utf-8")
            ast.parse(python)
            self.assertIn("EVAL_SOURCE_BRANCH", python)
            self.assertIn("EVAL_SOURCE_SHA", python)
            self.assertIn("core.autocrlf=false", python)
            self.assertIn("HALA_H291_PRIVATE_NONCOMMERCIAL_FREE_T4", python)
            with self.assertRaises(ValueError):
                build(src, out)
