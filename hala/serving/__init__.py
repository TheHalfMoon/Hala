"""Authenticated loopback HTTP transport with bounded admission and framing."""
from dataclasses import dataclass
import hmac
import ipaddress
import json
import math
import re
import select
import socket
import socketserver
import threading
import time

from hala.inference import MockBackend, RequestContext, ServiceError

API_VERSION = "hala.v1"
ENDPOINTS = {"/v1/hala/plan": "plan", "/v1/hala/action": "action",
             "/v1/hala/decision": "decision", "/v2/hala/action": "v2_action",
             "/v2/hala/plan": "v2_plan", "/v2/hala/decision": "v2_decision"}


@dataclass(frozen=True, repr=False)
class Config:
    token: str
    tenant_id: str
    request_timeout_s: float = 2.0
    max_concurrency: int = 4
    max_body_bytes: int = 65536
    max_header_bytes: int = 8192
    max_output_bytes: int = 65536

    def __post_init__(self):
        if not isinstance(self.token, str) or not 32 <= len(self.token) <= 256 or not self.token.isascii() or not all(33 <= ord(c) <= 126 for c in self.token):
            raise ValueError("Token must contain 32..256 printable ASCII non-space characters.")
        if not isinstance(self.tenant_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.tenant_id):
            raise ValueError("A bounded tenant identifier is required.")
        timeout = self.request_timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("Request timeout must be finite and in (0,60].")
        limits = {"max_concurrency": (1, 64), "max_body_bytes": (1, 1048576),
                  "max_header_bytes": (64, 65536), "max_output_bytes": (160, 1048576)}
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Invalid " + name + " limit.")


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("Nonfinite number")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite number")
        return result

    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float)
        if not isinstance(value, dict):
            raise ValueError("Object required")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ServiceError("invalid_request") from None


class Service:
    def __init__(self, config, backend=None, candidate_backend=None):
        self.config = config
        self.backend = backend or MockBackend()
        from hala.inference.vision_candidate import CandidateBackend
        self.candidate_backend = candidate_backend or CandidateBackend()

    def authenticate(self, value):
        try:
            provided = value.removeprefix("Bearer ").encode("ascii") if value.startswith("Bearer ") else b""
        except UnicodeError:
            provided = b""
        if not hmac.compare_digest(provided, self.config.token.encode("ascii")):
            raise ServiceError("unauthorized")

    def infer(self, responsibility, raw, context):
        context.check()
        if responsibility.startswith('v2_'):
            return self.infer_candidate(responsibility, raw, context)
        from hala.contracts import ContractError, Request, Proposal, Plan, Decision
        try:
            request = Request.from_dict(raw)
        except (ContractError, RecursionError):
            raise ServiceError("invalid_request") from None
        if request.tenant_id != self.config.tenant_id:
            raise ServiceError("tenant_mismatch")
        if request.cancelled:
            context.cancelled.set()
            raise ServiceError("cancelled")
        if not self.backend.metadata()["ready"]:
            raise ServiceError("backend_unqualified")
        record = self.backend.infer(responsibility, request, context)
        context.check()
        try:
            raw_output = record.to_dict()
            parsers = {"plan": Plan, "action": Proposal, "decision": Decision}
            parsed = parsers[responsibility].from_dict(raw_output, request)
            result = {"api_version": API_VERSION, "result": parsed.to_dict()}
            if len(encode(result)) > self.config.max_output_bytes:
                raise ServiceError("output_too_large")
            return result
        except (ContractError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
            raise ServiceError("invalid_backend_output") from None

    def infer_candidate(self, responsibility, raw, context):
        from hala.inference.vision_candidate import parse_request
        from hala.contracts import ContractError
        if responsibility != 'v2_action':
            raise ServiceError('unsupported_responsibility')
        try:
            request, image, eligible, digest = parse_request(raw, self.config.tenant_id)
        except (ContractError, RecursionError, OverflowError):
            raise ServiceError('invalid_request') from None
        context.check()
        proposal = self.candidate_backend.infer(request, image, eligible, digest, context)
        context.check()
        result = {'api_version': 'hala.v2', 'result': proposal}
        if len(encode(result)) > self.config.max_output_bytes:
            raise ServiceError('output_too_large')
        return result

    def metadata(self, path):
        result = {"api_version": API_VERSION, "backend": self.backend.metadata()}
        if path == "/health":
            result["healthy"] = True
        elif path == "/ready":
            result["ready"] = result["backend"]["ready"]
        return result


class _Lease:
    """Admission persists until both socket handler and inference worker finish."""
    def __init__(self, server):
        self.server = server
        self.owners = 1
        self.lock = threading.Lock()

    def retain(self):
        with self.lock:
            self.owners += 1

    def release(self):
        with self.lock:
            self.owners -= 1
            if self.owners == 0:
                self.server._release_slot()


class _Reader:
    def __init__(self, sock, context):
        self.sock, self.context = sock, context
        self.buffer = b""

    def receive(self, maximum):
        self.context.check()
        self.sock.settimeout(max(.001, self.context.deadline - time.monotonic()))
        try:
            piece = self.sock.recv(maximum)
        except socket.timeout:
            raise ServiceError("deadline_exceeded") from None
        if not piece:
            self.context.cancelled.set()
            raise ServiceError("cancelled")
        self.context.check()
        self.buffer += piece

    def line(self, limit):
        while b"\r\n" not in self.buffer:
            if len(self.buffer) >= limit:
                raise ServiceError("headers_too_large")
            self.receive(min(4096, limit - len(self.buffer)))
        line, self.buffer = self.buffer.split(b"\r\n", 1)
        if len(line) + 2 > limit:
            raise ServiceError("headers_too_large")
        return line

    def body(self, size):
        while len(self.buffer) < size:
            self.receive(min(4096, size - len(self.buffer)))
        if len(self.buffer) > size:
            raise ServiceError("invalid_request")
        result, self.buffer = self.buffer, b""
        return result


def _write(sock, status, body, timeout):
    reasons = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
               404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout",
               409: "Conflict", 411: "Length Required", 413: "Content Too Large",
               415: "Unsupported Media Type", 431: "Request Header Fields Too Large",
               500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable"}
    header = ("HTTP/1.1 %d %s\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: %d\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n" % (status, reasons[status], len(body))).encode("ascii")
    sock.settimeout(max(.001, timeout))
    sock.sendall(header + body)


def _error(sock, code):
    error = ServiceError(code)
    try:
        _write(sock, error.status, encode({"api_version": API_VERSION, "error": {"code": code, "message": error.message}}), .1)
        # Half-close and drain a bounded amount so unread request bytes do not
        # reset the socket and erase the error response on Windows.
        sock.shutdown(socket.SHUT_WR)
        drain_deadline = time.monotonic() + .05
        remaining = 65536
        while remaining and time.monotonic() < drain_deadline:
            sock.settimeout(max(.001, drain_deadline - time.monotonic()))
            piece = sock.recv(min(4096, remaining))
            if not piece:
                break
            remaining -= len(piece)
    except OSError:
        pass


class _Handler:
    def __init__(self, server, sock, lease, accepted_at):
        self.server, self.sock, self.lease = server, sock, lease
        self.service = server.service
        self.context = RequestContext(accepted_at + self.service.config.request_timeout_s)

    def read(self):
        reader = _Reader(self.sock, self.context)
        remaining = self.service.config.max_header_bytes
        line = reader.line(remaining)
        remaining -= len(line) + 2
        try:
            method, path, version = line.decode("ascii").split(" ")
        except (ValueError, UnicodeError):
            raise ServiceError("invalid_request") from None
        if version not in ("HTTP/1.0", "HTTP/1.1") or not re.fullmatch(r"[A-Z]{1,16}", method) or not 1 <= len(path) <= 1024:
            raise ServiceError("invalid_request")
        headers = {}
        while True:
            line = reader.line(remaining)
            remaining -= len(line) + 2
            if not line:
                break
            try:
                name, value = line.decode("ascii").split(":", 1)
            except (ValueError, UnicodeError):
                raise ServiceError("invalid_request") from None
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) or any(ord(c) < 32 and c != "\t" for c in value) or "\x7f" in value:
                raise ServiceError("invalid_request")
            name = name.lower()
            if name in headers:
                raise ServiceError("invalid_request")
            headers[name] = value.strip(" \t")
        self.service.authenticate(headers.get("authorization", ""))
        if "transfer-encoding" in headers or "expect" in headers:
            raise ServiceError("invalid_request")
        if method not in ("GET", "POST"):
            raise ServiceError("method_not_allowed")
        if path not in ENDPOINTS and path not in ("/health", "/ready", "/models"):
            raise ServiceError("not_found")
        if (path in ENDPOINTS) != (method == "POST"):
            raise ServiceError("method_not_allowed")
        length = headers.get("content-length", "0" if method == "GET" else None)
        if length is None:
            raise ServiceError("length_required")
        if not re.fullmatch(r"[0-9]{1,10}", length):
            raise ServiceError("invalid_request")
        length = int(length)
        if length > self.service.config.max_body_bytes:
            raise ServiceError("body_too_large")
        if method == "GET" and length:
            raise ServiceError("invalid_request")
        if method == "POST" and headers.get("content-type", "").lower() not in ("application/json", "application/json; charset=utf-8"):
            raise ServiceError("unsupported_media_type")
        body = reader.body(length)
        return path, strict_json(body) if method == "POST" else None

    def inference(self, responsibility, raw):
        done = threading.Event()
        result = []
        self.lease.retain()

        def work():
            try:
                result.append((True, self.service.infer(responsibility, raw, self.context)))
            except ServiceError as exc:
                result.append((False, exc))
            except Exception:
                result.append((False, ServiceError("internal_error")))
            finally:
                done.set()
                self.lease.release()

        worker = threading.Thread(target=work, name="hala-inference", daemon=True)
        try:
            worker.start()
        except Exception:
            self.lease.release()
            raise
        while not done.wait(.005):
            self.context.check()
            readable, _, _ = select.select([self.sock], [], [], 0)
            if readable:
                # This transport serves one request per connection. Disconnect or
                # unsolicited additional bytes cancel the current inference.
                self.context.cancelled.set()
                raise ServiceError("cancelled")
        self.context.check()
        successful, value = result[0]
        if not successful:
            raise value
        return value

    def run(self):
        try:
            path, raw = self.read()
            if raw is None:
                result = self.service.metadata(path)
                status = 503 if path == "/ready" and not result["ready"] else 200
            else:
                result = self.inference(ENDPOINTS[path], raw)
                status = 200
            self.context.check()
            body = encode(result)
            if len(body) > self.service.config.max_output_bytes:
                raise ServiceError("output_too_large")
            _write(self.sock, status, body, self.context.deadline - time.monotonic())
        except ServiceError as exc:
            self.context.cancelled.set()
            _error(self.sock, exc.code)
        except OSError:
            self.context.cancelled.set()
        except Exception:
            self.context.cancelled.set()
            _error(self.sock, "internal_error")
        finally:
            self.server.shutdown_request(self.sock)
            self.lease.release()


class Server(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address, service):
        self.service = service
        self._slots = threading.BoundedSemaphore(service.config.max_concurrency)
        self._idle = threading.Condition()
        self.active_requests = 0
        super().__init__(address, None)

    def process_request(self, request, client_address):
        accepted_at = time.monotonic()
        if not self._slots.acquire(blocking=False):
            _error(request, "busy")
            self.shutdown_request(request)
            return
        with self._idle:
            self.active_requests += 1
        lease = _Lease(self)
        handler = _Handler(self, request, lease, accepted_at)
        thread = threading.Thread(target=handler.run, name="hala-http", daemon=True)
        try:
            thread.start()
        except Exception:
            self.shutdown_request(request)
            lease.release()

    def _release_slot(self):
        with self._idle:
            self.active_requests -= 1
            self._slots.release()
            self._idle.notify_all()

    def wait_for_idle(self, timeout):
        with self._idle:
            return self._idle.wait_for(lambda: self.active_requests == 0, timeout)

    def handle_error(self, request, client_address):
        # Socket errors must never emit exception details or request data.
        _error(request, "internal_error")


def create_server(service, host="127.0.0.1", port=8765):
    if not ipaddress.ip_address(host).is_loopback or ":" in host:
        raise ValueError("Only IPv4 loopback binding is supported.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid port.")
    return Server((host, port), service)
