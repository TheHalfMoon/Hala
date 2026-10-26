"""H130 offline Playwright browser actuator + post-action visual feedback.

This module supplies screenshots and task text, executes model-produced pixel
coordinates on a real Chromium page and returns image + success feedback.
The synthetic fixture evaluator's DOM geometry is NOT exposed through the
agent observation or action API. Success of oracle QA is not model success.
"""
from dataclasses import dataclass
import hashlib
import math
import os
import re
from pathlib import Path
from urllib.parse import quote

TASKS = ("slider", "resize", "column", "drop")
WIDTH, HEIGHT = 960, 720
MAX_ACTIONS = 3

@dataclass(frozen=True)
class Observation:
    instruction: str
    screenshot_png: bytes
    width: int
    height: int
    step: int

    @property
    def image_sha256(self):
        return hashlib.sha256(self.screenshot_png).hexdigest()

@dataclass(frozen=True)
class Feedback:
    success: bool
    observation: Observation
    attempts: int
    action_applied: bool
    terminal: bool

def _point(raw):
    if (type(raw) not in (list, tuple) or len(raw) != 2 or
            any(type(n) not in (float, int) or not math.isfinite(n) for n in raw)):
        raise ValueError("H130 only finite model-predicted [x,y] coordinates")
    x, y = raw
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        raise ValueError("H130 action coordinates outside screenshot viewport")
    return (float(x), float(y))

def _chromium_binary():
    """Use an already-installed Chromium; never download browser binaries."""
    home = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = sorted(home.glob("ms-playwright/chromium-*/chrome-win64/chrome.exe"))
    if candidates:
        return str(candidates[-1])
    candidates = sorted(Path.home().glob(
        "Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"))
    return str(candidates[-1]) if candidates else None

class BrowserDragEnv:
    def __init__(self, task, chromium_path=None):
        if task not in TASKS:
            raise ValueError("H130 only named offline GUI scenarios")
        from playwright.sync_api import sync_playwright
        fixture = Path(__file__).with_name("fixture.html").resolve()
        if not fixture.is_file():
            raise ValueError("H130 controlled local offline fixture missing")
        self._playwright = sync_playwright().start()
        self._browser = None
        self._context = None
        try:
            kwargs = {"headless": True}
            executable = chromium_path or _chromium_binary()
            if executable:
                kwargs["executable_path"] = executable
            self._browser = self._playwright.chromium.launch(**kwargs)
            self._context = self._browser.new_context(
                viewport={"width": WIDTH, "height": HEIGHT},
                device_scale_factor=1, service_workers="block")
            self._context.route(re.compile(r"^https?://"),
                                lambda route: route.abort())
            self._page = self._context.new_page()
            self._external_requests = []
            self._page.on("request", self._track_request)
            self._page.on("pageerror", self._on_page_error)
            self._errors = []
            self._page.goto(fixture.as_uri() + "?task=" + quote(task), wait_until="load")
            self._page.locator('[data-role="handle"]').wait_for()
            self._task = task
            self._steps = 0
            self._terminal = False
        except BaseException:
            self.close()
            raise

    def _track_request(self, req):
        if req.url.startswith(("http://", "https://")):
            self._external_requests.append(req.url)

    def _on_page_error(self, error):
        self._errors.append(str(error))

    def observe(self):
        if self._external_requests or self._errors:
            raise RuntimeError("H130 unexpected external network or browser script error")
        return Observation(
            instruction=self._page.locator("#instruction").inner_text(),
            screenshot_png=self._page.screenshot(type="png"),
            width=WIDTH, height=HEIGHT, step=self._steps)

    def execute_drag(self, start, end, *, steps=16):
        if self._terminal or self._steps >= MAX_ACTIONS:
            raise RuntimeError("H130 session ended: create a fresh scene")
        start, end = _point(start), _point(end)
        if type(steps) is not int or not 2 <= steps <= 60:
            raise ValueError("H130 drag mouse steps outside safe limits")
        # Never inspect DOM geometry to adjust the model's points.
        self._page.mouse.move(*start)
        self._page.mouse.down()
        self._page.mouse.move(*end, steps=steps)
        self._page.mouse.up()
        self._steps += 1
        success = self._page.locator("#stage").get_attribute("data-success") == "true"
        self._terminal = success or self._steps >= MAX_ACTIONS
        obs = self.observe()
        return Feedback(success=success, observation=obs,
                        attempts=self._steps, action_applied=True,
                        terminal=self._terminal)

    def close(self):
        if getattr(self, "_context", None) is not None:
            self._context.close()
            self._context = None
        if getattr(self, "_browser", None) is not None:
            self._browser.close()
            self._browser = None
        if getattr(self, "_playwright", None) is not None:
            self._playwright.stop()
            self._playwright = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
