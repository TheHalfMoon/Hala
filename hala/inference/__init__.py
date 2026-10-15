"""CPU mock inference and an explicitly unavailable real-backend seam."""
from dataclasses import dataclass, field
import threading
import time


class ServiceError(Exception):
    """Only fixed public messages are exposed by the transport."""
    MESSAGES = {
        "invalid_request": (400, "Invalid request."),
        "unauthorized": (401, "Bearer authentication required."),
        "tenant_mismatch": (403, "Request tenant does not match credential."),
        "not_found": (404, "Unknown endpoint."),
        "method_not_allowed": (405, "Method not allowed."),
        "deadline_exceeded": (408, "Request deadline exceeded."),
        "cancelled": (409, "Request cancelled."),
        "length_required": (411, "Content-Length required."),
        "body_too_large": (413, "Request body exceeds limit."),
        "unsupported_media_type": (415, "application/json required."),
        "headers_too_large": (431, "Request headers exceed limit."),
        "internal_error": (500, "Internal service error."),
        "invalid_backend_output": (502, "Backend output failed validation."),
        "output_too_large": (502, "Backend output exceeds limit."),
        "busy": (503, "Service capacity exhausted."),
        "backend_unqualified": (503, "Real backend is not qualified."),
        "unsupported_responsibility": (503, "Candidate responsibility is unsupported."),
    }

    def __init__(self, code):
        self.code = code
        self.status, self.message = self.MESSAGES[code]
        super().__init__(self.message)


@dataclass
class RequestContext:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)

    def check(self):
        if time.monotonic() >= self.deadline:
            self.cancelled.set()
            raise ServiceError("deadline_exceeded")
        if self.cancelled.is_set():
            raise ServiceError("cancelled")


class MockBackend:
    def metadata(self):
        return {"name": "deterministic_mock", "mock": True, "qualified": False,
                "checkpoint": None, "ready": True,
                "capabilities": ["plan", "action", "decision"]}

    def infer(self, responsibility, request, context):
        from hala import HalaMind, HalaType, HalaReliance
        context.check()
        components = {"plan": (HalaMind, "plan"), "action": (HalaType, "action"),
                      "decision": (HalaReliance, "decide")}
        if responsibility not in components:
            raise ServiceError("not_found")
        component, method = components[responsibility]
        result = getattr(component(), method)(request)
        context.check()
        return result


class RealBackend:
    """No weights or runtime accepted until a separate implementation is qualified.

    A caller-provided `qualified=True` flag is deliberately not supported.
    """
    def metadata(self):
        return {"name": "real_unavailable", "mock": False, "qualified": False,
                "checkpoint": None, "ready": False, "capabilities": []}

    def infer(self, responsibility, request, context):
        context.check()
        raise ServiceError("backend_unqualified")
