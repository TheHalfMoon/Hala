"""H322: bounded pre-action recovery from one exact Chromium CDP capture error.

Only source-owned, PUBLIC EXPOSED browser research calls this helper.
This does NOT allow use of an invalid frame or relax PNG, viewport,
source URL, screenshot-lock, task-state, or data-admission validation.
Never logs raw browser exceptions or screenshots.
"""
from __future__ import annotations

from typing import Any, Callable

from h264.screenshot_locked_research import _observed_png

_CAPTURE_ERROR = (
    "Page.screenshot: Protocol error (Page.captureScreenshot): "
    "Unable to capture screenshot"
)


class SourceCaptureRefused(ValueError):
    """Sanitized terminal error: do not expose browser/page metadata."""


def source_capture_png(
    page: Any, *, capture: Callable[[Any], bytes] | None = None,
) -> bytes:
    """Retry once only for the exact known transient Chromium CDP failure.

    Both attempts run BEFORE any native action or external sample handoff;
    successful capture must still pass the original full screenshot checks.
    Any other exception, or a second capture error, is terminal. We do
    not use cached/stale frames or return an empty substitute.
    """
    fn = _observed_png if capture is None else capture
    for attempt in range(2):
        try:
            return fn(page)
        except Exception as error:
            known = (
                type(error).__name__ == "Error"
                and str(error).startswith(_CAPTURE_ERROR)
            )
            if not known or attempt:
                raise SourceCaptureRefused(
                    "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED"
                ) from None
    raise SourceCaptureRefused("H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED")
