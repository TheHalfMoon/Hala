"""H370: privacy-safe, fail-closed original-browser error stage categories.

Never expose exception messages/tracebacks, source task text, browser URLs,
image bytes, points, labels, model outputs or independently sealed results.
This diagnostic is for publicly exposed source-browser research ONLY.
"""
from __future__ import annotations

STAGES = {
    "h319": (
        "initial_original_scene",
        "first_render_and_policy",
        "first_image_capture",
        "first_image_digest",
        "first_policy_callback",
        "first_post_policy_frame",
        "native_click",
        "post_click_state",
        "second_render_and_policy",
        "second_image_capture",
        "second_image_digest",
        "second_policy_callback",
        "second_post_policy_frame",
        "native_type",
        "final_source_state",
    ),
    "h337": (
        "initial_original_scene",
        "first_render_and_policy",
        "first_image_capture",
        "first_image_digest",
        "first_policy_callback",
        "first_post_policy_frame",
        "native_click",
        "post_click_state",
        "miss_distance_geometry",
        "final_url",
    ),
}
PREFIX = {
    "h319": "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED",
    "h337": "H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED",
}
# Explicit mapping is intentionally narrower than exception type hierarchy;
# a user-supplied class name must never be interpolated into a CI log.
CATEGORIES = {
    ValueError: "value",
    AssertionError: "assertion",
    TypeError: "type",
    KeyError: "key",
    IndexError: "index",
    RuntimeError: "runtime",
    TimeoutError: "timeout",
    OSError: "os",
}


def browser_failure_code(flow: str, stage: str, cause: BaseException) -> str:
    """Provide ONLY fixed stage/type values, preserving original refusal ID."""
    if not (type(flow) is str and flow in STAGES and type(stage) is str
            and stage in STAGES[flow] and isinstance(cause, BaseException)):
        raise ValueError("H370 INVALID_DIAGNOSTIC_INPUT")
    kind = CATEGORIES.get(type(cause), "other")
    # Playwright's two public exception types can be recognized without
    # including arbitrary exception text or custom subclass names.
    typ = type(cause)
    if typ.__module__ == "playwright._impl._errors":
        if typ.__name__ == "TimeoutError":
            kind = "browser_timeout"
        elif typ.__name__ == "Error":
            kind = "browser_error"
    return PREFIX[flow] + " stage=" + stage + " kind=" + kind
