"""H84 evidence-bound evaluation cache regression, CPU-only, no provider/model."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h61.colab_run import step_evaluate, step_official_eval
from h84.eval_cache import sha256
from h60.real_drag_dev_baseline import (
    DEV_SETS, MODEL_ID, MODEL_REVISION, MAX_CASES, PROTOCOL, PROMPT_REVISION)
from h60.real_drag_dev_adapter_eval import ADAPTER_SCHEMA
from h61.official_eval_run import RUN_SCHEMA

SHA1 = "a" * 64
SHA2 = "b" * 64
WEIGHTS = "c" * 64


def identity(sha):
    return {"checkpoint_sha256": sha, "adapter_weights_sha256": WEIGHTS,
            "adapter_dir": "/synthetic-only/adapter"}


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.digest = SHA1
        self.dev_runs = []
        self.official_runs = []

    def mock_identity(self, root):
        return identity(self.digest)

    def dev_runner(self, root, output, checkpoint, *, dev):
        folder = Path(output)
        self.dev_runs.append(folder)
        folder.mkdir(parents=True)
        (folder / "raw-attempts.json").write_text('{"synthetic":"evidence"}', "utf8")
        record = {
            "schema": ADAPTER_SCHEMA, "status": "completed", "dev_set": dev,
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "source_sha256": DEV_SETS[dev]["source_sha256"],
            "prompt_revision": PROMPT_REVISION, "coordinate_protocol": PROTOCOL,
            "split": "public_dev", "evidence": "real_inference",
            "provider": "interactive_approved_free_colab",
            "max_cases": MAX_CASES, "adapters_loaded": True,
            "model_training_executed": True, "adapter_sha256": WEIGHTS,
            "adapter_checkpoint": identity(self.digest),
            "real_generated_cases": [{"id": n} for n in range(MAX_CASES)],
            "raw_attempts_sha256": sha256(folder / "raw-attempts.json"),
        }
        (folder / "run-result.json").write_text(json.dumps(record), "utf8")
        return record

    def official_runner(self, domains, folder, *, limit, adapter, adapter_sha256):
        target = Path(folder)
        self.official_runs.append(target)
        target.mkdir(parents=True)
        raw, scores = {}, {}
        record = {
            "schema": RUN_SCHEMA, "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "evidence": "real_inference", "adapters_loaded": adapter is not None,
            "adapter_sha256": adapter_sha256,
            "requested_domains": list(domains), "requested_limit": limit,
            "requested_max_new_tokens": 160,
            "raw_attempts_sha256": raw, "score_sha256": scores,
            "domains": {d: {"scenes": limit, "task_success": 0} for d in domains}
        }
        for domain in domains:
            dir_ = target / domain
            dir_.mkdir()
            (dir_ / "raw-attempts.json").write_text("{}", "utf8")
            (dir_ / "score.json").write_text('{"task_success":0}', "utf8")
            raw[domain] = sha256(dir_ / "raw-attempts.json")
            scores[domain] = sha256(dir_ / "score.json")
        (target / "official-eval-run.json").write_text(json.dumps(record), "utf8")
        return record

    def dev(self, dev="h55"):
        with (mock.patch("h60.real_drag_dev_adapter_eval.verify_adapter_checkpoint",
                         side_effect=self.mock_identity),
              mock.patch("h60.real_drag_dev_adapter_eval.run",
                         side_effect=self.dev_runner)):
            return step_evaluate(self.work, self.repo, [dev])

    def official(self, label="adapter", domains=None, limit=5):
        with (mock.patch("h60.real_drag_dev_adapter_eval.verify_adapter_checkpoint",
                         side_effect=self.mock_identity),
              mock.patch("h61.official_eval_run.run",
                         side_effect=self.official_runner)):
            return step_official_eval(
                self.work, self.repo, limit,
                domains or ["slider", "sheet"], label)

    def test_dev_results_cached_only_for_full_verified_model_digest(self):
        first = self.dev()
        self.assertEqual(len(self.dev_runs), 1)
        self.assertIn(SHA1, str(self.dev_runs[0]))
        second = self.dev()
        self.assertEqual(first, second)
        self.assertEqual(len(self.dev_runs), 1)
        self.digest = SHA2
        updated = self.dev()
        self.assertEqual(len(self.dev_runs), 2)
        self.assertIn(SHA2, str(self.dev_runs[1]))
        self.assertTrue(self.dev_runs[0].is_dir())
        self.assertEqual(updated["h55"]["adapter_checkpoint"]["checkpoint_sha256"], SHA2)

    def test_dev_rejects_digest_mutation_without_claiming_new_inference(self):
        self.dev()
        cache = self.dev_runs[0]
        record = json.loads((cache / "run-result.json").read_text("utf8"))
        record["adapter_checkpoint"]["checkpoint_sha256"] = SHA2
        (cache / "run-result.json").write_text(json.dumps(record), "utf8")
        with self.assertRaisesRegex(ValueError, "stale or unqualified"):
            self.dev()
        self.assertEqual(len(self.dev_runs), 1)

    def test_dev_rejects_tampered_raw_attempts_and_preserves_record(self):
        self.dev()
        target = self.dev_runs[0]
        (target / "raw-attempts.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "stale or unqualified"):
            self.dev()
        self.assertEqual((target / "raw-attempts.json").read_bytes(), b"tampered")

    def test_dev_ignores_legacy_unbound_cache_folder(self):
        legacy = self.work / "eval" / "adapter-h55"
        legacy.mkdir(parents=True)
        (legacy / "run-result.json").write_text('{"fake":"old"}', "utf8")
        self.dev()
        self.assertNotEqual(self.dev_runs[0], legacy)
        self.assertTrue((legacy / "run-result.json").is_file())

    def test_dev_partial_dir_prevents_expensive_duplicate_gpu_inference(self):
        target = self.work / "eval" / ("adapter-h55-" + SHA1)
        target.mkdir(parents=True)
        (target / "run-partial.json").write_text('{"partial":true}', "utf8")
        with self.assertRaisesRegex(ValueError, "cached evaluation receipt"):
            self.dev()
        self.assertFalse(self.dev_runs)

    def test_official_adapter_cache_bound_to_checkpoint_and_args(self):
        self.official()
        self.assertEqual(len(self.official_runs), 1)
        self.assertEqual(self.official()["slider"]["scenes"], 5)
        self.assertEqual(len(self.official_runs), 1)
        self.digest = SHA2
        self.official()
        self.assertEqual(len(self.official_runs), 2)
        self.assertTrue(self.official_runs[0].is_dir())
        self.official(limit=6)
        self.assertEqual(len(self.official_runs), 3)
        self.official(domains=["sheet", "slider"], limit=6)
        self.assertEqual(len(self.official_runs), 4)

    def test_official_proves_existing_raw_and_score_sha(self):
        self.official()
        target = self.official_runs[0]
        (target / "sheet" / "score.json").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "evidence file SHA mismatch"):
            self.official()
        self.assertEqual(len(self.official_runs), 1)

    def test_official_cannot_reuse_forged_provider_receipt(self):
        self.official()
        target = self.official_runs[0]
        path = target / "official-eval-run.json"
        record = json.loads(path.read_text("utf8"))
        record["adapter_sha256"] = "z" * 64
        path.write_text(json.dumps(record), "utf8")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.official()

    def test_official_base_baseline_cache_is_bound_to_params_separately(self):
        self.official(label="frozen-base", domains=["slider"])
        self.official(label="frozen-base", domains=["slider"])
        self.assertEqual(len(self.official_runs), 1)
        self.official(label="frozen-base", domains=["slider"], limit=7)
        self.assertEqual(len(self.official_runs), 2)
        self.assertNotEqual(self.official_runs[0], self.official_runs[1])

    def test_official_partial_existing_run_never_overwritten(self):
        self.official()
        target = self.official_runs[0]
        (target / "official-eval-run.json").unlink()
        with self.assertRaisesRegex(ValueError, "cached evaluation receipt"):
            self.official()
        self.assertEqual(len(self.official_runs), 1)

    def test_official_wrong_dev_domain_fail_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "arguments invalid"):
            self.official(domains=["slider", "wrong-domain"])

    def test_dev_wrong_dev_name_is_rejected_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "unrecognized frozen"):
            self.dev("wrong")

    def test_cached_official_missing_raw_evidence_with_null_sha_refused(self):
        self.official()
        target = self.official_runs[0]
        (target / "slider" / "raw-attempts.json").unlink()
        record_path = target / "official-eval-run.json"
        record = json.loads(record_path.read_text("utf8"))
        record["raw_attempts_sha256"]["slider"] = None
        record_path.write_text(json.dumps(record), "utf8")
        with self.assertRaisesRegex(ValueError, "evidence file SHA mismatch"):
            self.official()


if __name__ == "__main__":
    unittest.main()
