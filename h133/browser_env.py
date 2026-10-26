"""Fresh H133 independent browser task family (8 variants, never H130 images).

Inherited mouse executor and observation only; no ground truth/DOM in policy.
"""
import re
from pathlib import Path
from urllib.parse import quote
from h130.browser_env import (BrowserDragEnv, _chromium_binary, WIDTH, HEIGHT)

CASES=(
 "slider-north","slider-south","resize-west","resize-east",
 "column-left","column-right","drop-high","drop-low"
)
SOURCE_SCHEMA="hala.h133.unseen-browser-eight-source.v1"

class UnseenBrowserEnv(BrowserDragEnv):
    def __init__(self, task, chromium_path=None):
        if task not in CASES:
            raise ValueError("H133 only eight predeclared browser scenarios")
        from playwright.sync_api import sync_playwright
        file=Path(__file__).with_name("fixture.html").resolve()
        if not file.is_file():
            raise ValueError("H133 frozen offline fixture missing")
        self._playwright=sync_playwright().start()
        self._browser=None
        self._context=None
        try:
            kwargs={"headless":True}
            binary=chromium_path or _chromium_binary()
            if binary:
                kwargs["executable_path"]=binary
            self._browser=self._playwright.chromium.launch(**kwargs)
            self._context=self._browser.new_context(
                viewport={"width":WIDTH,"height":HEIGHT},
                device_scale_factor=1,service_workers="block")
            self._context.route(re.compile(r"^https?://"),lambda route:route.abort())
            self._page=self._context.new_page()
            self._external_requests=[]
            self._errors=[]
            self._page.on("request",self._track_request)
            self._page.on("pageerror",self._on_page_error)
            self._page.goto(file.as_uri()+"?case="+quote(task),wait_until="load")
            self._page.locator('[data-role="handle"]').wait_for()
            self._task=task
            self._steps=0
            self._terminal=False
        except BaseException:
            self.close()
            raise
