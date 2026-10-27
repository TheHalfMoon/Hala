"""H164 preserve original upstream app code while normalizing nondeterministic rendering."""
import inspect
import unittest
from h164.four_app_train_pilot import native_calculator
from h164.new_dev_96 import run as dev_run
from h164.expand_288_train import run as train_run

class H164VisualDeterminismTests(unittest.TestCase):
    def test_shadow_normalization_only_for_new_train_dev(self):
        source=inspect.getsource(native_calculator)
        self.assertIn("rest_mouse=False",source)
        self.assertIn('page.add_style_tag(content=',source)
        self.assertIn(".keypad .equals{box-shadow:none!important}",source)
        self.assertIn("page.mouse.move(0,0)",source)
        self.assertIn("stable_png(page)",source)
        self.assertIn("page.mouse.click(*point)",source)
    def test_original_pilot_stays_unchanged(self):
        source=inspect.getsource(__import__(
            "h164.four_app_train_pilot",fromlist=["run"]
        ).run)
        self.assertIn('execute=mode!="verify"',source)
        self.assertNotIn("rest_mouse=True",source)
    def test_new_train_and_dev_use_same_reproducible_view(self):
        self.assertIn("rest_mouse=True",inspect.getsource(train_run))
        self.assertIn("rest_mouse=True",inspect.getsource(dev_run))
        self.assertIn("H164_288_NEW_TRAIN_EXACT_CHROMIUM_REPLAY",
                      inspect.getsource(train_run))
