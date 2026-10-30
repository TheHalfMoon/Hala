"""H243: fail-closed Playwright browser-context offline guard.

A route("**/*") callback alone does not intercept every WebSocket created
inside a Dedicated Worker. The context's browser-level offline switch must
be enabled BEFORE page creation/navigation, in addition to existing routes.

This is defense in depth, NOT an OS-level network sandbox or proof that
WebRTC, browser extensions, native processes, or all future APIs are isolated.
All resulting GUI scenes remain research-only and not admitted TRAIN/DEV/OOD.
"""
from __future__ import annotations


def require_browser_offline(context: object) -> None:
    """Enable Playwright network emulation or abort before creating a page.

    No fallback to request routing alone is permitted.
    """
    setter = getattr(context, "set_offline", None)
    if not callable(setter):
        raise RuntimeError("H243 fail-closed: browser context has no offline control")
    try:
        setter(True)
    except Exception as exc:
        raise RuntimeError("H243 fail-closed: browser context offline setup failed") from exc
