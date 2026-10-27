"""H158 second-app original OOD no-training contract tests."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from h158.original_todomvc_source import COUNTS,TOTAL,PIN,episode
from h158.gpu_infer import validated_model_action,MODEL_SHAS
from h158.kaggle_eval.kaggle_private_eval import one_private_mount
from h158.audit_source import audit


class H158NewApplicationAdmission(unittest.TestCase):
    def test_real_unmodified_original_new_app_action_classes(self):
        self.assertEqual(TOTAL,16)
        self.assertEqual(sum(sum(x.values()) for x in COUNTS.values()),16)
        self.assertEqual(sum(x["ar"] for x in COUNTS.values()),8)
        self.assertEqual(sum(x["en"] for x in COUNTS.values()),8)
        self.assertEqual(len(PIN),40)
        source=inspect.getsource(episode)
        self.assertIn('page.mouse.click(x,y)',source)
        self.assertNotIn('model.generate(',source)

    def test_model_click_only_no_DOM_or_executable_code(self):
        good=json.dumps({"actions":[{"kind":"click","point":[120,230]}]})
        action=validated_model_action(good)
        self.assertTrue(action["valid"])
        self.assertEqual(action["click"],[120.0,230.0])
        for bad in ('{"actions":[{"kind":"eval","text":"x"}]}',
                    '{"actions":[{"kind":"click","point":[-1,20]}]}',
                    '{"actions":[{"kind":"click","point":[10,30],"dom":"#x"}]}',
                    'not json'):
            self.assertFalse(validated_model_action(bad)["valid"])
        self.assertNotEqual(MODEL_SHAS["before"],MODEL_SHAS["after"])

    def test_free_private_one_dataset_contains_three_distinct_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mount=root/"dynamic-private-slug"
            mount.mkdir()
            files=[mount/x for x in ("new-todo.zip","h156-lora.zip","h157-lora.zip")]
            for file in files:
                file.write_bytes(b"test")
            one_private_mount(root,tuple(files))
            (root/"other-dataset").mkdir()
            with self.assertRaises(ValueError):
                one_private_mount(root,tuple(files))

    def test_cannot_invent_model_source_without_frozen_owner_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit(tmp)


if __name__=="__main__":
    unittest.main()
