"""H140 one-time model evaluation requires SHA-identical original browser replay."""
import tempfile
import unittest
from h140.replay_blind_source import setup_context,source_pairing,CANDIDATES
from h140.blind_original_capture import board_for,param

class FakeContext:
    def __init__(self):
        self.script=None
        self.route_pattern=None
    def add_init_script(self,script):self.script=script
    def route(self,pattern,callback):self.route_pattern=pattern

class FakeBrowser:
    def __init__(self):self.ctx=None
    def new_context(self,**kwargs):
        self.ctx=FakeContext()
        self.viewport=kwargs["viewport"]
        self.service_workers=kwargs["service_workers"]
        return self.ctx

class H140ExactSourceBlindReplay(unittest.TestCase):
    def test_all_frozen_app_source_scripts_seed_once_without_gold(self):
        self.assertEqual(set(CANDIDATES),{"ood-vanilla-complete","ood-second-drag"})
        for app in CANDIDATES:
            attempt=30000 if app=="ood-vanilla-complete" else 36000
            n,source,*_=param(attempt)
            seed=board_for(app,attempt,n,source)
            browser=FakeBrowser()
            ctx=setup_context(browser,seed,app)
            self.assertEqual(browser.viewport,{"width":960,"height":720})
            self.assertEqual(browser.service_workers,"block")
            self.assertIn("===null",ctx.script)
            self.assertIn("localStorage.setItem",ctx.script)
            self.assertNotIn("start_bbox",ctx.script)
            self.assertNotIn("end_bbox",ctx.script)
            self.assertEqual(ctx.route_pattern,"**/*")

    def test_absent_seal_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                source_pairing(folder)

if __name__=="__main__":
    unittest.main()
