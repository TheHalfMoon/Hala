"""Pinned actual H37 Kaggle raw replay of H38 optional safety filter.

No new neural inference, blind benchmark, host execution or model qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from h38.replay import analyze, RAW_SHA, AUDIT_SHA

RETAINED = Path("reviews/h37-kaggle-run-20261020")
DERIVED = Path("docs/reviews/h38-h37-explicit-label-guard.json")
DERIVED_SHA = "5c47dc4c531b7efecd0ccc96a50ae0d2d4b518d1d8328811c844ca7907a14f83"


class H38ActualReplayTests(unittest.TestCase):
    def test_genuine_h37_raw_and_deterministic_derived_report(self):
        report = analyze(RETAINED)
        self.assertEqual(hashlib.sha256((RETAINED/"raw_attempts.jsonl").read_bytes()).hexdigest(),RAW_SHA)
        self.assertEqual(hashlib.sha256((RETAINED/"audit.json").read_bytes()).hexdigest(),AUDIT_SHA)
        self.assertEqual(report["counts"],{
            "correct_retained":32,"unbound_rejected":16,"wrong_unique_rejected":16})
        self.assertEqual(len(report["cases"]),64)
        self.assertEqual(report["languages"]["ar"]["correct_retained"],14)
        self.assertEqual(report["languages"]["en"]["correct_retained"],18)
        self.assertEqual(report["layouts"]["grid"]["wrong_unique_rejected"],4)
        self.assertEqual(report["layouts"]["sidebar"]["wrong_unique_rejected"],10)
        self.assertFalse(report["model_improved"])
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["independent_blind_test"])
        self.assertFalse(report["native_actions"])
        canonical=(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
        self.assertEqual(DERIVED.read_bytes(),canonical)
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),DERIVED_SHA)
        self.assertTrue(all("expected_label" not in r for r in report["cases"]))

    def test_corrupted_sha_or_original_audit_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ("raw_attempts.jsonl","audit.json"):
                shutil.copyfile(RETAINED/name,folder/name)
            raw=folder/"raw_attempts.jsonl"
            data=bytearray(raw.read_bytes())
            data[8]^=1
            raw.write_bytes(data)
            with self.assertRaisesRegex(ValueError,"raw SHA"):
                analyze(folder)
            shutil.copyfile(RETAINED/"raw_attempts.jsonl",raw)
            report=folder/"audit.json"
            data=bytearray(report.read_bytes())
            data[9]^=1
            report.write_bytes(data)
            with self.assertRaisesRegex(ValueError,"audit SHA"):
                analyze(folder)


if __name__=="__main__":
    unittest.main()
