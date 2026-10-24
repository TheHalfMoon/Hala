"""H85: CPU-only proof that every Qwen load shares one pinned session cache."""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h85.model_cache import HF_HOME, HUB_CACHE, pinned_cache_kwargs
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "training/scripts/baseline.py",
    "training/scripts/train_adapter.py",
    "h60/real_drag_dev_baseline.py",
    "h61/official_eval_run.py",
)


class ModelCacheTests(unittest.TestCase):
    def test_exact_original_revision_and_explicit_cache_dir(self):
        self.assertEqual(HF_HOME, "/tmp/hala-model-cache")
        self.assertEqual(HUB_CACHE, "/tmp/hala-model-cache/hub")
        self.assertEqual(pinned_cache_kwargs(MODEL_ID, MODEL_REVISION),
                         {"revision": MODEL_REVISION, "trust_remote_code": False,
                          "cache_dir": HUB_CACHE})

    def test_different_model_or_revision_fails_before_hub_access(self):
        for model, revision in (
            ("Qwen/Qwen3-VL-4B-Instruct", MODEL_REVISION),
            (MODEL_ID, "f" * 40),
            (MODEL_ID, ""),
            ("", MODEL_REVISION),
        ):
            with self.subTest(model=model, revision=revision):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    pinned_cache_kwargs(model, revision)

    def test_shared_cache_even_if_hf_home_was_previously_different(self):
        with mock.patch.dict(os.environ, {"HF_HOME": "/elsewhere/stale"}, clear=False):
            options = pinned_cache_kwargs(MODEL_ID, MODEL_REVISION)
            self.assertEqual(options["cache_dir"], HUB_CACHE)
            self.assertNotEqual(options["cache_dir"], "/elsewhere/stale/hub")

    def test_every_actual_qwen_and_processor_loader_explicitly_pins_cache(self):
        for relative in ENTRYPOINTS:
            with self.subTest(entrypoint=relative):
                tree = ast.parse((ROOT / relative).read_text("utf8"))
                calls = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in (
                        "AutoProcessor", "Qwen3VLForConditionalGeneration")
                ]
                self.assertEqual(len(calls), 2, "must cover weights AND processor")
                for call in calls:
                    unpacked = [k.value for k in call.keywords if k.arg is None]
                    self.assertEqual(len(unpacked), 1)
                    self.assertIsInstance(unpacked[0], ast.Call)
                    self.assertIsInstance(unpacked[0].func, ast.Name)
                    self.assertEqual(unpacked[0].func.id, "pinned_cache_kwargs")
                    self.assertFalse(any(
                        key.arg in {"revision", "trust_remote_code", "cache_dir"}
                        for key in call.keywords
                    ), "source must have no conflicting unpinned cache arguments")

    def test_explicit_shared_home_before_heavy_library_imports(self):
        # H60/H61 previously set HF_HOME only AFTER importing transformers.
        for relative in (
            "h60/real_drag_dev_baseline.py", "h61/official_eval_run.py",
            "training/scripts/train_adapter.py", "training/scripts/baseline.py",
        ):
            with self.subTest(entrypoint=relative):
                tree = ast.parse((ROOT / relative).read_text("utf8"))
                run = next(n for n in ast.walk(tree)
                           if isinstance(n, ast.FunctionDef)
                           and n.name == ("train" if relative.endswith("train_adapter.py") else "run"))
                assignments = [n for n in ast.walk(run)
                               if isinstance(n, ast.Assign)
                               and any(isinstance(target, ast.Subscript)
                                       and isinstance(target.value, ast.Attribute)
                                       and target.value.attr == "environ"
                                       and isinstance(target.slice, ast.Constant)
                                       and target.slice.value == "HF_HOME"
                                       for target in n.targets)]
                imports = [n for n in ast.walk(run)
                           if isinstance(n, (ast.Import, ast.ImportFrom))
                           and (any(name.name == "transformers" for name in n.names)
                                or (isinstance(n, ast.ImportFrom) and n.module == "transformers"))]
                self.assertTrue(assignments)
                self.assertTrue(imports)
                self.assertLess(min(n.lineno for n in assignments),
                                min(n.lineno for n in imports))


if __name__ == "__main__":
    unittest.main()
