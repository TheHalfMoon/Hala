"""H373: safe buckets for public-source Playwright screenshot exceptions.

Only a fixed reason enum may leave this module. The raw Playwright message is
read in memory solely to select a category; it must NEVER be logged, returned,
attached or used as purported proof of a particular browser root cause.
"""
from __future__ import annotations

# These are labels of reported failure *signatures*, not causal findings.
CAPTURE_BUCKETS = frozenset({
    "protocol_capture",
    "target_closed",
    "execution_context",
    "font_wait_unresolved",
    "screenshot_timeout",
    "browser_other",
})
CAPTURE_PHASE_SUFFIXES = (
    "_image_capture",
    "_post_policy_frame",
)


def safe_screenshot_bucket(stage: str, cause: BaseException) -> str | None:
    """Categorize only verified Playwright error types during capture phases.

    This function NEVER returns raw exception details, even when those details
    contain arbitrary untrusted URL/task labels or file paths.
    """
    if not (type(stage) is str and isinstance(cause, BaseException)
            and stage.endswith(CAPTURE_PHASE_SUFFIXES)):
        return None
    typ = type(cause)
    if not (typ.__module__ == "playwright._impl._errors"
            and typ.__name__ in {"Error", "TimeoutError"}):
        return None

    # DO NOT log, store or return this message. Only exact fixed enum strings
    # can leave, and unknown signatures fail closed as browser_other.
    raw = str(cause).lower()
    if ("target page, context or browser has been closed" in raw
            or "target closed" in raw):
        return "target_closed"
    if ("protocol error (page.capturescreenshot)" in raw
            or "unable to capture screenshot" in raw):
        return "protocol_capture"
    if ("execution context was destroyed" in raw
            or "cannot find context with specified id" in raw):
        return "execution_context"
    if typ.__name__ == "TimeoutError":
        if ("waiting for fonts to load" in raw
                and "fonts loaded" not in raw):
            return "font_wait_unresolved"
        return "screenshot_timeout"
    return "browser_other"
