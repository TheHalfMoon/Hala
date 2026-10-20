"""H40 genuine Windows CPU-child lifecycle stress evidence regression.

No GPU, paid services or native user-interface control.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.h40_cpu_lifecycle_stress import verify_receipts,run

ROOT=Path("reviews")
FILES=("h40-lifecycle-isolated-20261020","h40-lifecycle-loaded-20261020")


class CpuLifecycleEvidenceTests(unittest.TestCase):
    def test_real_cpu_fault_matrix_complete_and_exact(self):
        for name in FILES:
            with self.subTest(name=name):
                evidence=verify_receipts(ROOT/name)
                self.assertEqual(evidence["attempts"],40)
                self.assertEqual(evidence["passed"],40)
                self.assertEqual(evidence["rounds"],8)
                self.assertFalse(evidence["actual_model"])
                self.assertFalse(evidence["cuda"])
                self.assertFalse(evidence["native_actions"])
                self.assertFalse(evidence["retries"])
                self.assertEqual(len((ROOT/(name+".jsonl")).read_bytes().splitlines()),40)
                self.assertEqual(evidence["source_head"],
                                 "15017fcfd7c8e4155959c5db609982e1f2ec46df")

    def test_modified_or_incomplete_journal_never_qualifies(self):
        original=ROOT/FILES[0]
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/"run"
            journal=output.with_suffix(".jsonl")
            summary=output.with_suffix(".json")
            journal.write_bytes(original.with_suffix(".jsonl").read_bytes())
            summary.write_bytes(original.with_suffix(".json").read_bytes())
            self.assertEqual(verify_receipts(output)["attempts"],40)
            altered=journal.read_bytes().replace(b'"goal": "resource"',b'"goal": "unknown"',1)
            journal.write_bytes(altered)
            with self.assertRaises(ValueError):
                verify_receipts(output)
            journal.write_bytes(original.with_suffix(".jsonl").read_bytes())
            metadata=json.loads(summary.read_text(encoding="utf-8"))
            metadata["passed"]=41
            summary.write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipts(output)
            metadata["passed"]=40
            metadata["native_actions"]=True
            summary.write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipts(output)
            # An attacker who recomputes a plain SHA still cannot change
            # the exact expected lifecycle fault classification.
            metadata["native_actions"]=False
            changed=journal.read_bytes().replace(
                b'"last_fault_code": null',b'"last_fault_code": "busy"',1)
            self.assertNotEqual(changed,journal.read_bytes())
            journal.write_bytes(changed)
            metadata["journal_sha256"]=hashlib.sha256(changed).hexdigest()
            summary.write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipts(output)

    def test_read_only_gates_and_create_only_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/"existing"
            dest.with_suffix(".jsonl").write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                run(dest,rounds=1)
            self.assertEqual(dest.with_suffix(".jsonl").read_bytes(),b"original")
            with self.assertRaises(ValueError):
                run(Path(t)/"bad",rounds=0)
            with self.assertRaises(ValueError):
                run(Path(t)/"bad",rounds=25)
            with self.assertRaises(ValueError):
                run(Path(t)/"bad",rounds=1,call_timeout=8)


if __name__=="__main__":unittest.main()
