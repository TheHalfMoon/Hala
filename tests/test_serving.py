"""Real loopback qualification of bounded mock transport; no model weights."""
import http.client
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import threading
import time
import unittest

from hala.inference import MockBackend, RealBackend, RequestContext
from hala.serving import Config, Service, create_server


TOKEN = "test-token-only-01234567890123456789"


def payload():
    return {
        "schema": "hala.request.v1", "request_id": "r-1", "tenant_id": "tenant-1",
        "task": {"task_id": "task-1", "goal": "Click Save", "step_index": 0, "max_steps": 10},
        "device": {"device_id": "device-1", "session_id": "session-1", "os": "windows-11", "locale": "en-US"},
        "observation": {
            "generation": "obs-1", "window": {"window_ref": "window-1", "title": "Editor", "process": "synthetic.exe"},
            "accessibility": [{"element_ref": "save-1", "role": "Button", "name": "Save", "bounds": [10, 20, 100, 30]}],
            "image": {"format": "png", "width": 800, "height": 600, "coordinate_space": "window", "scale": 1.0, "redactions": []},
            "truncation": {"elements_omitted": 0},
        },
        "allowed_actions": ["semantic_invoke", "ask_user", "abstain", "observe"],
        "method_ceiling": "WINDOW_TARGETED_INPUT", "cancelled": False,
    }


class BlockingBackend(MockBackend):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.context = None

    def infer(self, responsibility, request, context):
        self.context = context
        self.started.set()
        self.release.wait(3)
        return super().infer(responsibility, request, context)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self.start()

    def start(self, backend=None, **changes):
        config = Config(token=TOKEN, tenant_id="tenant-1", **changes)
        service = Service(config, backend or MockBackend())
        server = create_server(service, port=0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        self.server, self.service = server, service

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(1)

    def call(self, method="GET", path="/health", body=None, token=TOKEN, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        merged = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        merged.update(headers or {})
        connection.request(method, path, body, merged)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, json.loads(data) if data else {}

    def raw(self, data):
        sock = socket.create_connection(self.server.server_address, timeout=3)
        sock.sendall(data)
        result = b""
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            result += piece
        sock.close()
        status = int(result.split(b" ", 2)[1])
        return status, json.loads(result.split(b"\r\n\r\n", 1)[1])

    def test_metadata_and_all_mock_paths(self):
        for path in ["/health", "/ready", "/models"]:
            status, result = self.call(path=path)
            self.assertEqual(status, 200)
            self.assertEqual(result["api_version"], "hala.v1")
            self.assertFalse(result["backend"]["qualified"])
            self.assertTrue(result["backend"]["mock"])
        self.assertTrue(self.call(path="/ready")[1]["ready"])
        for path in ["plan", "action", "decision"]:
            status, result = self.call("POST", "/v1/hala/" + path, json.dumps(payload()))
            self.assertEqual(status, 200, result)
            self.assertEqual(result["result"]["tenant_id"], "tenant-1")
            self.assertFalse(result["result"]["metadata"]["qualified"])
            self.assertNotIn("receipt", result["result"])

    def test_auth_and_cross_tenant(self):
        for token in ["wrong", "", TOKEN + "x", "☃"]:
            if token == "☃":
                continue  # HTTP client forbids non-Latin-1 headers.
            status, result = self.call(token=token)
            self.assertEqual(status, 401)
            self.assertNotIn(TOKEN, json.dumps(result))
        raw = payload()
        raw["tenant_id"] = "tenant-2"
        self.assertEqual(self.call("POST", "/v1/hala/action", json.dumps(raw))[0], 403)
        self.assertEqual(self.call("POST", "/v1/hala/action", json.dumps(payload()))[0], 200)

    def test_isolated_binding_state(self):
        raw = payload()
        raw["request_id"] = "r-2"
        raw["device"]["session_id"] = "session-2"
        for item in [payload(), raw, payload()]:
            _, result = self.call("POST", "/v1/hala/action", json.dumps(item))
            self.assertEqual(result["result"]["request_id"], item["request_id"])
            self.assertEqual(result["result"]["session_id"], item["device"]["session_id"])

    def test_strict_json_and_contract_errors(self):
        for body in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{} trailing', '[]', '{', '{"x":' + '[' * 2000 + '1' + ']' * 2000 + '}']:
            status, result = self.call("POST", "/v1/hala/plan", body)
            self.assertEqual(status, 400, body[:40])
            self.assertEqual(result["error"]["code"], "invalid_request")
        self.assertEqual(self.call("POST", "/v1/hala/plan", b"\xff")[0], 400)

    def test_concurrent_requests_preserve_independent_bindings(self):
        class Concurrent(MockBackend):
            def __init__(self):
                self.barrier = threading.Barrier(4)

            def infer(self, responsibility, request, context):
                self.barrier.wait(timeout=1)
                return super().infer(responsibility, request, context)

        self.start(Concurrent(), max_concurrency=4)
        requests = []
        for index in range(4):
            item = payload()
            item["request_id"] = "r-" + str(index)
            item["device"]["session_id"] = "s-" + str(index)
            requests.append(item)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda item: self.call("POST", "/v1/hala/action", json.dumps(item)), requests))
        for item, (status, result) in zip(requests, results):
            self.assertEqual(status, 200, result)
            self.assertEqual(result["result"]["request_id"], item["request_id"])
            self.assertEqual(result["result"]["session_id"], item["device"]["session_id"])

    def test_body_output_and_header_bounds(self):
        self.start(max_body_bytes=32)
        self.assertEqual(self.call("POST", "/v1/hala/plan", "x" * 33)[0], 413)
        self.start(max_output_bytes=256)
        self.assertEqual(self.call("POST", "/v1/hala/plan", json.dumps(payload()))[0], 502)
        self.start(max_header_bytes=256)
        request = b"GET /health HTTP/1.1\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\nX-Large: " + b"x" * 300 + b"\r\n\r\n"
        self.assertEqual(self.raw(request)[0], 431)

    def test_framing_content_type_methods_and_routes(self):
        for method in ["PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "CONNECT", "TRACE"]:
            self.assertEqual(self.call(method)[0], 405)
        self.assertEqual(self.call(path="/missing")[0], 404)
        self.assertEqual(self.call("GET", "/v1/hala/action")[0], 405)
        self.assertEqual(self.call("POST", "/health", "{}")[0], 405)
        self.assertEqual(self.call("POST", "/v1/hala/plan", "{}", headers={"Content-Type": "text/plain"})[0], 415)
        prefix = b"POST /v1/hala/plan HTTP/1.1\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\nContent-Type: application/json\r\n"
        for header in [b"Content-Length: 1\r\nContent-Length: 1\r\n", b"Transfer-Encoding: chunked\r\n", b"Content-Length: -1\r\n", b"Content-Length: 0\r\nContent-Length: 2\r\n"]:
            self.assertEqual(self.raw(prefix + header + b"\r\n{}")[0], 400)

    def test_absolute_read_deadline(self):
        self.start(request_timeout_s=.12)
        sock = socket.create_connection(self.server.server_address, timeout=3)
        prefix = b"POST /v1/hala/plan HTTP/1.1\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{"
        sock.sendall(prefix)
        time.sleep(.18)
        data = sock.recv(4096)
        sock.close()
        self.assertIn(b" 408 ", data)
        self.assertIn(b"deadline_exceeded", data)

    def test_slow_header_deadline(self):
        self.start(request_timeout_s=.12)
        sock = socket.create_connection(self.server.server_address, timeout=3)
        sock.sendall(b"GET /health HTTP/1.1\r\n")
        time.sleep(.18)
        self.assertIn(b" 408 ", sock.recv(4096))
        sock.close()

    def test_pre_cancelled_request(self):
        raw = payload()
        raw["cancelled"] = True
        status, result = self.call("POST", "/v1/hala/action", json.dumps(raw))
        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "cancelled")

    def test_trickle_does_not_extend_absolute_deadline(self):
        self.start(request_timeout_s=.15)
        sock = socket.create_connection(self.server.server_address, timeout=3)
        started = time.monotonic()
        sock.sendall(b"GET /health HTTP/1.1\r\nX-Slow: ")
        for _ in range(3):
            time.sleep(.04)
            try:
                sock.sendall(b"x")
            except OSError:
                break
        self.assertIn(b" 408 ", sock.recv(4096))
        self.assertLess(time.monotonic() - started, .6)
        sock.close()

    def test_deadline_does_not_release_running_worker_capacity(self):
        backend = BlockingBackend()
        self.start(backend, request_timeout_s=.15, max_concurrency=1)
        status, result = self.call("POST", "/v1/hala/action", json.dumps(payload()))
        self.assertEqual(status, 408)
        self.assertTrue(backend.context.cancelled.is_set())
        self.assertEqual(self.call()[0], 503)
        backend.release.set()
        self.assertTrue(self.server.wait_for_idle(1))
        self.assertEqual(self.call()[0], 200)

    def test_disconnect_cancels_and_holds_capacity(self):
        backend = BlockingBackend()
        self.start(backend, max_concurrency=1)
        sock = socket.create_connection(self.server.server_address, timeout=3)
        data = json.dumps(payload()).encode()
        sock.sendall(b"POST /v1/hala/action HTTP/1.1\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
        self.assertTrue(backend.started.wait(1))
        sock.close()
        self.assertTrue(backend.context.cancelled.wait(1))
        self.assertEqual(self.call()[0], 503)
        backend.release.set()
        self.assertTrue(self.server.wait_for_idle(1))

    def test_admission_before_handler_spawn(self):
        self.start(max_concurrency=1, request_timeout_s=.3)
        sock = socket.create_connection(self.server.server_address, timeout=3)
        sock.sendall(b"GET ")
        for _ in range(100):
            if self.server.active_requests:
                break
            time.sleep(.005)
        self.assertEqual(self.server.active_requests, 1)
        self.assertEqual(self.call()[0], 503)
        self.assertEqual(self.server.active_requests, 1)
        sock.close()

    def test_backend_errors_are_safe(self):
        class Broken(MockBackend):
            def infer(self, *args):
                raise RuntimeError(TOKEN + " secret path")
        self.start(Broken())
        status, result = self.call("POST", "/v1/hala/plan", json.dumps(payload()))
        self.assertEqual(status, 500)
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertNotIn("secret path", json.dumps(result))

    def test_real_backend_is_unqualified(self):
        self.start(RealBackend())
        status, result = self.call(path="/ready")
        self.assertEqual(status, 503)
        self.assertFalse(result["ready"])
        self.assertFalse(result["backend"]["mock"])
        self.assertEqual(self.call("POST", "/v1/hala/plan", json.dumps(payload()))[0], 503)

    def test_invalid_backend_bindings_and_authority_fail_closed(self):
        class Forged(MockBackend):
            def __init__(self, field):
                self.field = field

            def infer(self, responsibility, request, context):
                result = super().infer(responsibility, request, context).to_dict()
                result[self.field] = "forged"
                class Record:
                    def to_dict(self):
                        return result
                return Record()

        for field in ["tenant_id", "request_id", "session_id", "observation_generation", "receipt"]:
            self.start(Forged(field))
            status, result = self.call("POST", "/v1/hala/action", json.dumps(payload()))
            self.assertEqual(status, 502)
            self.assertEqual(result["error"]["code"], "invalid_backend_output")

    def test_in_process_deadline_and_cancel(self):
        context = RequestContext(time.monotonic() + 1)
        context.cancelled.set()
        with self.assertRaises(Exception) as cancelled:
            self.service.infer("action", payload(), context)
        self.assertEqual(cancelled.exception.code, "cancelled")
        context = RequestContext(time.monotonic() - 1)
        with self.assertRaises(Exception) as expired:
            self.service.infer("action", payload(), context)
        self.assertEqual(expired.exception.code, "deadline_exceeded")


class ConfigTests(unittest.TestCase):
    def test_invalid_config(self):
        for key in ["request_timeout_s", "max_concurrency", "max_body_bytes", "max_header_bytes", "max_output_bytes"]:
            for value in [0, -1, True, float("nan"), float("inf")]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    Config(token=TOKEN, tenant_id="tenant-1", **{key: value})
        for token in ["", "short", "x" * 33 + "\n", "☃" * 32]:
            with self.assertRaises(ValueError):
                Config(token=token, tenant_id="tenant-1")
        for key in ["max_concurrency", "max_body_bytes", "max_header_bytes", "max_output_bytes"]:
            with self.assertRaises(ValueError):
                Config(token=TOKEN, tenant_id="tenant-1", **{key: 1.5})
        with self.assertRaises(ValueError):
            Config(token=TOKEN, tenant_id="")

    def test_loopback_only(self):
        service = Service(Config(token=TOKEN, tenant_id="tenant-1"))
        with self.assertRaises(ValueError):
            create_server(service, host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
