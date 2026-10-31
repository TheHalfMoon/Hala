"""H277 actual Chromium cross-family original visual identity audit.

This is source qualification, not source-blind OOD or model inference.
Eight screenshots exist ONLY in volatile memory; verify both PNG bytes
and decoded RGBA fingerprints never coincide across two first-party
offline GUIs. No screenshot files, prompts, gold or weights are saved.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import unittest

try:
    from PIL import Image
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    Image = None
    sync_playwright = None

ROOT=Path(__file__).resolve().parents[1]
STOCKROOM=ROOT/"h262"/"stockroom.html"
QUEUEBOARD=ROOT/"h275"/"queueboard.html"
STOCKROOM_SHA="3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
VIEWPORT={"width":960,"height":720}


@unittest.skipIf(Image is None or sync_playwright is None,
                 "optional browser/Pillow packages; H277 hosted Chromium job required")
class OriginalTwoFamilyPixelAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for source in (STOCKROOM,QUEUEBOARD):
            if not source.is_file() or source.is_symlink():
                raise AssertionError("H277 source family missing or symlinked")
        if sha256(STOCKROOM.read_bytes()).hexdigest()!=STOCKROOM_SHA:
            raise AssertionError("H277 original Stockroom source bytes drifted")
        if b'hala-original-queueboard-h275' not in QUEUEBOARD.read_bytes():
            raise AssertionError("H277 original Queueboard source family drifted")
        cls.playwright=sync_playwright().start()
        try:
            cls.browser=cls.playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox","--disable-dev-shm-usage"])
        except BaseException:
            cls.playwright.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.browser.close()
        finally:
            cls.playwright.stop()
        super().tearDownClass()

    def setUp(self):
        self.context=self.browser.new_context(
            viewport=VIEWPORT,device_scale_factor=1,
            locale="en-US",color_scheme="light",
            reduced_motion="reduce",service_workers="block")
        self.blocked=[]
        self.page_errors=[]
        def firewall(route):
            if route.request.url.startswith(("http:","https:","ws:","wss:")):
                self.blocked.append("unexpected remote request")
                route.abort()
            else:
                route.continue_()
        self.context.route("**/*",firewall)

    def tearDown(self):
        try:
            self.assertEqual(self.blocked,[],"Original app attempted network")
            self.assertEqual(self.page_errors,[],"Original app JS failed")
        finally:
            self.context.close()

    def page(self,source):
        if source not in (STOCKROOM,QUEUEBOARD):
            raise ValueError("H277 only registered two original sources")
        p=self.context.new_page()
        p.set_default_timeout(5000)
        p.on("pageerror",lambda error:self.page_errors.append(str(error)))
        p.goto(source.as_uri(),wait_until="load")
        self.assertEqual(p.url,source.as_uri())
        self.assertEqual(p.viewport_size,VIEWPORT)
        return p

    def fingerprint(self,page):
        png=page.screenshot(type="png",full_page=False,animations="disabled")
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(BytesIO(png)) as im:
            self.assertEqual(im.format,"PNG")
            self.assertEqual(im.size,(960,720))
            rgba=im.convert("RGBA").tobytes()
        self.assertEqual(len(rgba),960*720*4)
        # RGBA equality is checked separately from PNG compression/metadata.
        png_digest=sha256(png).digest()
        rgba_digest=sha256(rgba).digest()
        del png,rgba
        return (png_digest,rgba_digest)

    def test_real_two_original_app_families_have_eight_distinct_scenes(self):
        scenes={}
        stock=self.page(STOCKROOM)
        scenes["stockroom.en.default"]=self.fingerprint(stock)
        stock.locator("#sort").click()
        scenes["stockroom.en.sorted"]=self.fingerprint(stock)
        stock.locator("#lang").click()
        self.assertEqual(stock.locator("html").get_attribute("dir"),"rtl")
        scenes["stockroom.ar.sorted"]=self.fingerprint(stock)
        stock.locator("#search").fill("Gauze")
        stock.locator("#search").evaluate("(el) => el.blur()")
        self.assertGreaterEqual(stock.locator(".item").count(),1)
        scenes["stockroom.ar.search"]=self.fingerprint(stock)
        stock.close()

        queue=self.page(QUEUEBOARD)
        scenes["queueboard.en.default"]=self.fingerprint(queue)
        queue.locator("#priority").click()
        self.assertEqual(queue.locator(".ticket-row").count(),2)
        scenes["queueboard.en.priority"]=self.fingerprint(queue)
        queue.locator("#lang").click()
        self.assertEqual(queue.locator("html").get_attribute("dir"),"rtl")
        scenes["queueboard.ar.priority"]=self.fingerprint(queue)
        queue.locator("#search").fill("Q-137")
        queue.locator("#search").evaluate("(el) => el.blur()")
        self.assertEqual(queue.locator(".ticket-row").count(),1)
        scenes["queueboard.ar.search"]=self.fingerprint(queue)
        queue.close()

        self.assertEqual(len(scenes),8)
        self.assertEqual(len({v[0] for v in scenes.values()}),8,
                         "H277 source-owned PNG bytes duplicate")
        self.assertEqual(len({v[1] for v in scenes.values()}),8,
                         "H277 decoded RGBA source visual duplicate")
        self.assertEqual(len(set(s.split(".")[0] for s in scenes)),2)
        # Never print fingerprint bytes or write PNGs to CI artifacts.

    def test_real_native_clicks_prove_independent_state_oracles(self):
        stock=self.page(STOCKROOM)
        stock.locator("#search").fill("Gauze")
        stock.locator("#search").evaluate("(el) => el.blur()")
        self.assertGreaterEqual(stock.locator(".item").count(),1)
        first=stock.locator(".item .count").first
        before=int(first.inner_text())
        stock.locator("button[data-restock]").first.click()
        after=int(stock.locator(".item .count").first.inner_text())
        self.assertEqual(after,before+1)
        stock.close()

        queue=self.page(QUEUEBOARD)
        self.assertEqual(queue.locator("#completed-count").inner_text(),"0")
        queue.locator('button[data-action="complete"][data-ticket-id="Q-137"]').click()
        self.assertEqual(queue.locator(
            '.ticket-row[data-ticket-id="Q-137"]').get_attribute(
                "data-status"),"done")
        self.assertEqual(queue.locator("#completed-count").inner_text(),"1")
        self.assertEqual(queue.locator(
            '.ticket-row[data-ticket-id="Q-104"]').get_attribute(
                "data-status"),"waiting")
        queue.close()


    def test_original_stockroom_blurred_search_is_stable_across_delay(self):
        # This qualifies only the source-owned Chromium screenshot
        # stability premise for H274. No Qwen inference or OOD scored.
        stock=self.page(STOCKROOM)
        stock.locator("#search").fill("Gauze")
        self.assertGreaterEqual(stock.locator(".item").count(),1)
        stock.locator("#search").evaluate("(el) => el.blur()")
        self.assertFalse(stock.locator("#search").evaluate(
            "(el) => document.activeElement === el"))
        first=self.fingerprint(stock)
        stock.wait_for_timeout(1800)
        second=self.fingerprint(stock)
        self.assertEqual(first,second,
                         "Blurred original Stockroom screen drifted during idle")
        stock.close()


if __name__=="__main__":
    unittest.main()
