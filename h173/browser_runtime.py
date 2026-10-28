"""Pinned original Chromium renderer shared by source QA and native OOD.

This changes only the browser rasterization backend, not the untouched
original MIT app source, CSS, the screenshot pixels, or any reference gold.
Windows owner-pristine source had reproducible cross-launch pixel drift with
default GPU compositing. Software rasterization is subject to the same strict
decoded RGBA replay checks and may still fail closed.
"""
from h130.browser_env import _chromium_binary

BROWSER_ARGS = ("--disable-gpu",)


def original_browser_options():
    options = {"headless": True, "args": list(BROWSER_ARGS)}
    binary = _chromium_binary()
    if binary:
        options["executable_path"] = binary
    return options
