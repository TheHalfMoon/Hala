"""No-model, source-only H28 prospective plan bundle tests."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h28 import corpus, protocol
from tools import build_h28_dev_plan


class H28PlanBuilderTests(unittest.TestCase):
    def test_all_896_plan_cells_are_deterministic_and_auditable(self):
        first, refs, manifest = build_h28_dev_plan.prepare()
        second, second_refs, second_manifest = build_h28_dev_plan.prepare()
        self.assertEqual((first, refs, manifest), (second, second_refs, second_manifest))
        self.assertEqual(len(first.splitlines()), 896)
        self.assertEqual(len(refs.splitlines()), 896)
        self.assertEqual(manifest["model_messages_sha256"],
                         hashlib.sha256(first).hexdigest())
        self.assertEqual(manifest["evaluator_sha256"],
                         hashlib.sha256(refs).hexdigest())
        self.assertEqual(manifest["source_sha256"], corpus.PINNED_SOURCE_SHA256)
        self.assertFalse(manifest["independent_holdout"])
        self.assertEqual(manifest["gpu_calls"], 0)
        self.assertEqual(manifest["model_inferences_run"], 0)

    def test_model_prompt_never_contains_oracle_or_family_id(self):
        prompts, refs, _ = build_h28_dev_plan.prepare()
        p = [json.loads(row) for row in prompts.splitlines()]
        r = [json.loads(row) for row in refs.splitlines()]
        self.assertEqual(len(p), len(r))
        for model_row, reference in zip(p, r):
            self.assertEqual(model_row["request_key"], reference["request_key"])
            self.assertEqual(set(model_row), {"request_key", "messages"})
            self.assertEqual(set(reference), {
                "request_key", "source_ref", "rotation",
                "instruction_locale", "expected_wire"
            })
            self.assertEqual([x["role"] for x in model_row["messages"]], ["system", "user"])
            self.assertNotIn(reference["source_ref"]["id"],
                             json.dumps(model_row["messages"], ensure_ascii=False))
            self.assertNotIn("source_ref", json.dumps(model_row["messages"]))
            self.assertNotIn("private_reference", json.dumps(model_row["messages"]))
            state = reference["source_ref"]["state"]
            book = protocol.codebook(reference["rotation"])
            self.assertEqual(book[reference["expected_wire"]], corpus.REFERENCE[state])

    def test_bundle_created_once_never_overwrites(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "new_h28_plan"
            manifest = build_h28_dev_plan.write_plan(target)
            self.assertEqual(json.loads((target / "manifest.json").read_text("utf-8")),
                             manifest)
            self.assertEqual(hashlib.sha256(
                (target / "model_messages.jsonl").read_bytes()).hexdigest(),
                manifest["model_messages_sha256"])
            self.assertEqual(hashlib.sha256(
                (target / "evaluator_only.jsonl").read_bytes()).hexdigest(),
                manifest["evaluator_sha256"])
            with self.assertRaises(FileExistsError):
                build_h28_dev_plan.write_plan(target)


if __name__ == "__main__":
    unittest.main()
