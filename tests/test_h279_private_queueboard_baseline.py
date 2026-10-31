"""Invented-only CPU regression for H279; never invokes Qwen or private media."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h279.private_queueboard_qwen_baseline import (
    SCENES, SOURCE_BLOB, FormatRejected, clone_pinned_source,
    source_task, validate_rows, RelativeDecodingProcessor)
from h279.build_private_payload import build


def invented_rows():
    return [
        {"scene": scene[0], "real_generations": 1,
         "grammar_generate_calls": 1, "strict_normalized_json": True,
         "native_dispatched": True, "firstparty_task_success": False,
         "native_kind": "click"}
        for scene in SCENES
    ]


class H279SourceOnlyTests(unittest.TestCase):
    def test_exact_two_language_tasks(self):
        self.assertEqual(source_task(SCENES[0]),
                         "Complete fictional ticket Q-118")
        self.assertEqual(source_task(SCENES[1]),
                         "أكمل الطلب الخيالي Q-142")
        with self.assertRaises(ValueError):
            source_task(("unregistered", "en", False, "Q-104"))

    def test_closed_aggregate_no_accuracy_claim(self):
        report = validate_rows(invented_rows(), {"free_t4": True, "devices": 2})
        self.assertEqual(report["firstparty_task_successes"], 0)
        self.assertFalse(report["independent_blind_ood"])
        self.assertFalse(report["model_improvement_proven"])
        self.assertEqual(report["optimizer_updates"], 0)

    def test_true_native_success_requires_real_click(self):
        rows = invented_rows()
        rows[0]["firstparty_task_success"] = True
        rows[0]["native_dispatched"] = False
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})
        rows[0]["native_dispatched"] = True
        rows[0]["native_kind"] = "abstain"
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_no_fake_generation_or_extra_fields(self):
        rows = invented_rows()
        rows[1]["real_generations"] = 0
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 2})
        rows = invented_rows()
        rows[0]["raw_prediction"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 2})

    def test_strict_invalid_never_dispatches(self):
        rows = invented_rows()
        rows[0].update(strict_normalized_json=False, native_dispatched=False,
                       native_kind=None)
        validate_rows(rows, {"free_t4": True, "devices": 1})
        rows[0]["native_dispatched"] = True
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_fake_gpu_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(invented_rows(), {"free_t4": False, "devices": 2})

    def test_original_source_blob_is_exact(self):
        from h276.queueboard_native_research import (
            QUEUEBOARD_GIT_BLOB_SHA1, checked_queueboard)
        source = Path(__file__).resolve().parents[1] / "h275" / "queueboard.html"
        checked_queueboard(source)
        self.assertEqual(SOURCE_BLOB, QUEUEBOARD_GIT_BLOB_SHA1)

    def test_normalized_y999_converted_before_h208_pixel_parser(self):
        from h273.relative_native_bridge import normalize_strict_action
        from h199.native_action_sft import parse_actions

        class InventedDecoder:
            def batch_decode(self, *args, **kwargs):
                return ['{"actions":[{"kind":"click","point":[999,999]}]}']
            def apply_chat_template(self, *args, **kwargs):
                return None

        adapter = RelativeDecodingProcessor(
            InventedDecoder(), normalize_strict_action)
        native = adapter.batch_decode([[1]])
        self.assertEqual(len(native), 1)
        action = parse_actions(native[0])["actions"][0]
        self.assertEqual(action["point"], [959, 719])
        self.assertEqual(adapter.converted_calls, 1)
        self.assertFalse(adapter.format_rejected)

    def test_invalid_relative_model_text_rejected_without_native_fallback(self):
        from h273.relative_native_bridge import normalize_strict_action

        class InventedBadDecoder:
            def batch_decode(self, *args, **kwargs):
                return ['not-json']
            def apply_chat_template(self, *args, **kwargs):
                return None

        adapter = RelativeDecodingProcessor(
            InventedBadDecoder(), normalize_strict_action)
        with self.assertRaises(FormatRejected):
            adapter.batch_decode([[1]])
        self.assertTrue(adapter.format_rejected)
        self.assertEqual(adapter.converted_calls, 0)

    def test_private_code_only_packager(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "h279"
            folder.mkdir()
            source = folder / "private_queueboard_qwen_baseline.py"
            source.write_text(
                "from __future__ import annotations\n"
                "def validate_rows(): pass\n"
                "def run_firstparty_queueboard_probe(): pass\n"
                "class RelativeDecodingProcessor: pass\n"
                "def stub():\n    decoder = RelativeDecodingProcessor(None, None)\n"
                'if __name__ == "__main__": pass\n', encoding="utf-8")
            destination = root / "payload"
            result = build(source, destination)
            meta = json.loads(
                (destination / "kernel-metadata.json").read_text("utf-8"))
            self.assertTrue(result["private"])
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["model_sources"], [])
            self.assertEqual(meta["dataset_sources"], [])
            with self.assertRaises(ValueError):
                build(source, destination)


if __name__ == "__main__":
    unittest.main()
