"""H156 native action protocol must be DOM-free and fail closed."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h156.action_contract_v2 import (
    action_from_dict_v2, parse_v2, native_mouse_keyboard,
)
from h156.source_train_dev import collect


MODEL_SHA="a"*64


class H156NativeActionContractTests(unittest.TestCase):
    def test_select_requires_visible_point_and_bounded_option(self):
        action=action_from_dict_v2({
            "kind":"select","point":[100,120],"text":"United Arab Emirates"})
        self.assertEqual(action.point,(100.0,120.0))
        self.assertEqual(action.text,"United Arab Emirates")
        for value in (
            {"kind":"select","text":"Mali"},
            {"kind":"select","point":[100,120]},
            {"kind":"select","point":[float("nan"),10],"text":"Mali"},
            {"kind":"select","point":[100,720],"text":"Mali"},
            {"kind":"select","point":[100,120],"text":"a\nb"},
            {"kind":"select","point":[100,120],"text":"Mali","dom":"#options"},
        ):
            with self.subTest(value=value),self.assertRaises(ValueError):
                action_from_dict_v2(value)

    def test_frozen_model_role_is_not_implicitly_admitted(self):
        payload=json.dumps({"actions":[
            {"kind":"select","point":[100,150],"text":"Ghana"},
            {"kind":"submit","point":[200,250]}]})
        with self.assertRaises(ValueError):
            parse_v2(payload,task="choose-list",model_checkpoint_sha="untrained")
        self.assertEqual([x.kind for x in parse_v2(
            payload,task="choose-list",model_checkpoint_sha=MODEL_SHA)],
            ["select","submit"])
        with self.assertRaises(ValueError):
            parse_v2(payload,task="click-test",model_checkpoint_sha=MODEL_SHA)

    def test_native_controller_does_not_read_dom(self):
        source=inspect.getsource(native_mouse_keyboard)
        for banned in ("locator(","evaluate(","query_selector(","select_option(","page.goto("):
            self.assertNotIn(banned,source)

    def test_consumed_dataset_requires_no_browser_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"private-h156"/"miniwob-action-train-dev-v2").mkdir(parents=True)
            with patch("h156.source_train_dev.admit_upstream",return_value=root):
                with self.assertRaises(FileExistsError):
                    collect(root,root)


if __name__=="__main__":
    unittest.main()
