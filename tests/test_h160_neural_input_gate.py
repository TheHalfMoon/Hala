"""H160 strict owner-private cross-app GPU preflight tests."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from h160.pack_private_eval import SCHEMA, instruction
from h160.gpu_infer import validate
from h158.original_todomvc_source import tasks_for


class H160NeuralInputTests(unittest.TestCase):
    def test_all_task_families_have_explicit_bilingual_instructions(self):
        for task in ("complete_item", "filter_active", "clear_completed"):
            for lang in ("en", "ar"):
                with self.subTest(task=task, lang=lang):
                    item = {"task": task, "language": lang, "index": 305}
                    actual = instruction(item)
                    self.assertTrue(actual)
                    if lang == "ar":
                        self.assertTrue(any("\u0600" <= c <= "\u06ff" for c in actual))
                    if task == "complete_item":
                        self.assertIn(tasks_for(lang,305)[1], actual)

    def test_original_adapters_only_and_never_read_reference_points(self):
        from h160 import gpu_infer, pack_private_eval
        source = inspect.getsource(gpu_infer) + inspect.getsource(pack_private_eval)
        self.assertNotIn("model.train(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertIn("h140_blind_calls", source)
        self.assertIn("reference_point", source)  # only explicit negative/no-upload guard
        self.assertNotIn("case['reference_point']", source)

    def test_gpu_rejects_wrong_frozen_zip_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            with ZipFile(path,"w") as z:
                z.writestr("manifest.json",json.dumps({"schema":SCHEMA}))
            with self.assertRaises(ValueError):
                validate(path)


if __name__ == "__main__":
    unittest.main()
