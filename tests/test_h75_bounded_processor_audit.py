"""H75 bounded bilingual audit, end-to-end CLI contract, and fail-closed tests.

All fixtures are synthetic. No provider, weights, source image or GPU is used.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h75.processor_audit import audit_summary, select_processor_audit_rows
from h65.safe_diagnostic import supervise
from h62.run_h61_diagnostic import execute
from h61.colab_run import step_train

SOURCE = "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe"
CODE = "6eed2e800645a83871ae742d2ac2c2a899dc706a"
ZIP = "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a"


class FakeChild:
    pid = 111
    returncode = 0
    def poll(self): return 0
    def wait(self, timeout=None): return 0
    def terminate(self): self.returncode = -15
    def kill(self): self.returncode = -9


class BalancedAudit(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for language, family, n in [
                ("en", "slider", 50), ("en", "sheet", 40),
                ("en", "slide_resize", 40), ("en", "text_highlight", 40),
                ("ar", "firstparty", 20), ("en", "firstparty", 20)]:
            rows.extend({"id": f"{language}-{family}-{i}",
                         "family": family, "language": language}
                        for i in range(n))
        return rows

    def test_full_audit_unchanged_for_default_zero(self):
        rows = self.rows()
        chosen = select_processor_audit_rows(rows, 0)
        self.assertEqual(chosen, rows)
        self.assertEqual(audit_summary(chosen, len(rows))["audit_scope"], "full")

    def test_bounded_audit_deterministically_covers_all_language_families(self):
        rows = self.rows()
        first = select_processor_audit_rows(rows, 32)
        second = select_processor_audit_rows(rows, 32)
        self.assertEqual([x["id"] for x in first], [x["id"] for x in second])
        self.assertEqual(len(first), 32)
        self.assertEqual({(r["language"], r["family"]) for r in first},
                         {(r["language"], r["family"]) for r in rows})
        report = audit_summary(first, len(rows))
        self.assertGreater(report["audit_language_counts"]["ar"], 0)
        self.assertGreater(report["audit_language_counts"]["en"], 0)
        self.assertEqual(report["audit_scope"], "bounded_diagnostic_not_full_processor_audit")

    def test_low_budget_rejects_missing_language_family(self):
        with self.assertRaisesRegex(ValueError, "cannot represent"):
            select_processor_audit_rows(self.rows(), 5)

    def test_no_source_or_invalid_budget_fails(self):
        for invalid in (-1, True, 65, 2.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_processor_audit_rows(self.rows(), invalid)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            select_processor_audit_rows([], 32)

    def test_non_h61_full_legacy_rows_need_no_bilingual_fields(self):
        other = [{"id": "legacy"}]
        self.assertEqual(select_processor_audit_rows(other, 0), other)


class Propagation(unittest.TestCase):
    def test_host_supervisor_explicit_32_reaches_operator(self):
        commands = []
        with tempfile.TemporaryDirectory() as root:
            result = supervise("existing-free-t4", CODE, Path(root),
                               steps="train", stage_sha256=ZIP,
                               stage_source_sha=SOURCE, max_steps=1,
                               processor_audit_records=32,
                               runner=lambda command, **kw: (commands.append(command) or FakeChild()),
                               snapshot=lambda *args: {"status": "VERIFIED_HOST_STAGING", "step": 1},
                               hardware=lambda _: "T4", sleep=lambda _: None,
                               clock=lambda: 0.0)
        self.assertEqual(result["status"], "COMPLETED_WITH_HOST_EGRESS")
        self.assertIn("--processor-audit-records", commands[0])
        self.assertEqual(commands[0][commands[0].index("--processor-audit-records") + 1], "32")
        self.assertIn("--stage-source-sha", commands[0])

    def test_bound_rejected_before_provider_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = mock.Mock()
            for steps, budget, audit in [
                    ("train", 4, 32), ("train", 1, 65),
                    ("evaluate", 1, 32), ("train", 1, -1)]:
                with self.subTest(steps=steps, budget=budget, audit=audit):
                    with self.assertRaises(ValueError):
                        supervise("existing-t4", CODE, Path(temp), steps=steps,
                                  stage_sha256=ZIP, stage_source_sha=SOURCE,
                                  max_steps=budget, processor_audit_records=audit,
                                  hardware=provider)
            provider.assert_not_called()

    def test_h62_operator_sends_diagnostic_envelope_only_when_explicit(self):
        with mock.patch("h62.run_h61_diagnostic.run", return_value=(0, "", "")) as runner:
            execute("existing-t4", CODE, "train", "slider=1", "1", "1",
                    "{}", 60, ZIP, SOURCE, 32)
        command = runner.call_args.args[0]
        self.assertIn("H61_PROCESSOR_AUDIT_RECORDS=32", command)
        self.assertIn("H67_STAGE_SOURCE_SHA=" + SOURCE, command)
        self.assertIn("HALA_PINNED_SOURCE_SHA=" + CODE, command)

    def test_h62_operator_rejects_nonstaged_audit_before_cli(self):
        with mock.patch("h62.run_h61_diagnostic.run") as remote:
            with self.assertRaisesRegex(ValueError, "requires staged"):
                execute("existing-t4", CODE, "train", "", "1", "1",
                        "{}", 30, processor_audit_records=32)
        remote.assert_not_called()

    def test_train_step_writes_explicit_config_identity_without_mutating_source(self):
        original = {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "max_steps": 600, "audit_records": 0,
        }
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "config.json"
            source.write_text(json.dumps(original), encoding="utf8")
            with mock.patch("training.scripts.train_adapter.train", return_value={"step": 1}) as mocked:
                result = step_train(Path(temp) / "work", source, 1, False, 32)
            config = json.loads((Path(temp) / "work" / "training-config.json").read_text("utf8"))
            self.assertEqual(config["max_steps"], 1)
            self.assertEqual(config["audit_records"], 32)
            self.assertEqual(json.loads(source.read_text("utf8")), original)
            self.assertEqual(result, {"step": 1})
            mocked.assert_called_once()

    def test_long_training_cannot_use_shortcut(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "config.json"
            source.write_text('{"audit_records":0}', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "1..3-step"):
                step_train(Path(temp) / "run", source, 10, False, 32)


if __name__ == "__main__":
    unittest.main()
