"""Loopback image/element binding checks with a deterministic pixel test double.

These fixtures exercise Hala's parser and candidate mapping; they are not model
qualification or rendered UI screenshots.
"""
import base64
import hashlib
import io
import json
import unittest

from PIL import Image, ImageDraw

from hala.inference.vision_candidate import CandidateBackend
from hala.serving import Config, Service
from tests import test_serving as transport


TARGET = (220, 40, 40)
BACKGROUND = (24, 28, 32)
REDACTION = (700, 500, 12, 12)


def fixture(locale="en-US", rtl=False, image_id=0):
    """Make synthetic geometry and image pixels; Arabic locale is data only."""
    left = (40 + image_id * 7, 80, 100, 50)
    right = (640 - image_id * 7, 80, 100, 50)
    selected = left if rtl else right
    other = right if rtl else left
    raw_request = transport.payload()
    raw_request["allowed_actions"] = ["click", "abstain"]
    raw_request["device"]["locale"] = locale
    raw_request["observation"]["accessibility"] = [
        {"element_ref": "target-" + str(image_id), "role": "Button",
         "name": "\u062d\u0641\u0638" if rtl else "Save", "bounds": list(selected)},
    ]
    raw_request["observation"]["image"]["width"] = 800
    raw_request["observation"]["generation"] = "frame-" + str(image_id)
    raw_request["request_id"] = "request-" + str(image_id)
    raw_request["task"]["task_id"] = "task-" + str(image_id)
    raw_request["device"]["device_id"] = "device-" + str(image_id)
    raw_request["device"]["session_id"] = "session-" + str(image_id)
    raw_request["observation"]["accessibility"][0]["bounds"] = list(selected)

    canvas = Image.new("RGB", (800, 600), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((selected[0], selected[1], selected[0] + selected[2] - 1,
                    selected[1] + selected[3] - 1), fill=TARGET)
    draw.rectangle((REDACTION[0], REDACTION[1], REDACTION[0] + REDACTION[2] - 1,
                    REDACTION[1] + REDACTION[3] - 1), fill=(0, 0, 0))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    data = stream.getvalue()
    media = {"png_base64": base64.b64encode(data).decode("ascii"),
             "sha256": hashlib.sha256(data).hexdigest(),
             "frame_id": raw_request["observation"]["generation"],
             "request_id": raw_request["request_id"], "tenant_id": "tenant-1",
             "task_id": raw_request["task"]["task_id"],
             "device_id": raw_request["device"]["device_id"],
             "session_id": raw_request["device"]["session_id"],
             "origin": [0, 0], "dpi": 96}
    raw = {"schema": "hala.image_request.v2", "policy_request": raw_request, "media": media}
    return raw, data, selected, other


class PixelTargetEngine:
    """Deterministic test double that finds the unique target-colored pixels."""
    provenance = {"engine": "pixel-test-double", "model_id": "synthetic-fixture",
                  "model_revision": "fixture-v1", "processor_revision": "pillow-rgb-v1",
                  "precision": "integer-pixels", "prompt_revision": "pixel-region-v1"}

    def __init__(self):
        self.calls = []

    def __call__(self, image, request, context):
        pixels = image.convert("RGB")
        points = [(x, y) for y in range(pixels.height) for x in range(pixels.width)
                  if pixels.getpixel((x, y)) == TARGET]
        if not points:
            raise AssertionError("submitted image contains no target pixels")
        # Record a digest of every decoded pixel and return the region centroid.
        self.calls.append((hashlib.sha256(pixels.tobytes()).hexdigest(), pixels.size,
                           dict(request), context))
        x = sum(point[0] for point in points) // len(points)
        y = sum(point[1] for point in points) // len(points)
        return json.dumps({"point": [x * 1000 // pixels.width,
                                      y * 1000 // pixels.height]})


class ImageBindingHTTPTests(unittest.TestCase):
    tearDown = transport.ServerTests.tearDown
    call = transport.ServerTests.call

    def setUp(self):
        self.servers = []
        self.engine = PixelTargetEngine()
        self.service = Service(Config(transport.TOKEN, "tenant-1"),
                               candidate_backend=CandidateBackend(self.engine))
        self.start()

    def start(self):
        # Reuse the established loopback server lifecycle and request helper.
        from hala.serving import create_server
        from threading import Thread
        server = create_server(self.service, port=0)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        self.server = server

    def post(self, raw):
        body = json.dumps(raw, ensure_ascii=False).encode("utf-8")
        return self.call("POST", "/v2/hala/action", body)

    def assert_advisory_flags(self, result):
        proposal = result["result"]
        self.assertFalse(proposal["execution_authorized"])
        self.assertIsNone(proposal["confidence"])
        self.assertTrue(proposal["verification_required"])

    def test_authenticated_pixels_and_mirrored_locale_layout_bindings(self):
        for index, (locale, rtl) in enumerate((("en-US", False), ("ar-SA", True))):
            raw, data, bounds, _ = fixture(locale, rtl, index)
            status, response = self.post(raw)
            self.assertEqual(status, 200, response)
            proposal = response["result"]
            self.assertEqual(proposal["media_sha256"], hashlib.sha256(data).hexdigest())
            for field, value in (("request_id", raw["policy_request"]["request_id"]),
                                 ("tenant_id", "tenant-1"),
                                 ("task_id", raw["policy_request"]["task"]["task_id"]),
                                 ("device_id", raw["policy_request"]["device"]["device_id"]),
                                 ("session_id", raw["policy_request"]["device"]["session_id"]),
                                 ("frame_id", raw["policy_request"]["observation"]["generation"])):
                self.assertEqual(proposal[field], value)
            self.assertEqual(proposal["actions"][0]["element_ref_hint"],
                             raw["policy_request"]["observation"]["accessibility"][0]["element_ref"])
            x, y, width, height = bounds
            px, py = proposal["actions"][0]["point"]
            self.assertTrue(x <= px < x + width and y <= py < y + height)
            self.assertEqual(self.engine.calls[-1][0], hashlib.sha256(
                Image.open(io.BytesIO(data)).convert("RGB").tobytes()).hexdigest())
            self.assertEqual(self.engine.calls[-1][1], (800, 600))
            self.assert_advisory_flags(response)
        # The fixture names a rectangle only; it contains no shaped Arabic text.
        points = [record[0] for record in self.engine.calls]
        self.assertEqual(len(points), 2)
        self.assertNotEqual(points[0], points[1])

    def test_tampered_media_and_mixed_bindings_reject_before_engine(self):
        valid, _, _, _ = fixture()
        before = len(self.engine.calls)
        status, response = self.call("POST", "/v2/hala/action",
                                     json.dumps(valid).encode("utf-8"), token="wrong-token")
        self.assertEqual(status, 401, response)
        self.assertEqual(len(self.engine.calls), before)

        mutations = []
        swapped, _, _, _ = fixture()
        _, other_data, _, _ = fixture(image_id=9)
        # Keep this request's digest while changing only its image bytes.
        swapped["media"]["png_base64"] = base64.b64encode(other_data).decode("ascii")
        mutations.append(swapped)
        for media_field in ("request_id", "frame_id", "session_id", "device_id"):
            mixed, _, _, _ = fixture()
            mixed["media"][media_field] = "cross-request-identity"
            mutations.append(mixed)
        fresh, _, _, _ = fixture(image_id=6)
        other_request, _, _, _ = fixture(image_id=7)
        for field in ("request_id", "tenant_id", "task_id", "device_id", "session_id", "frame_id"):
            fresh["media"][field] = other_request["media"][field]
        mutations.append(fresh)
        for malformed in mutations:
            before = len(self.engine.calls)
            status, response = self.post(malformed)
            self.assertEqual(status, 400, response)
            self.assertEqual(response["error"]["code"], "invalid_request")
            self.assertEqual(len(self.engine.calls), before)

    def test_redaction_pixels_and_unrelated_redacted_target(self):
        raw, data, bounds, other = fixture()
        raw["policy_request"]["observation"]["image"]["redactions"] = [list(REDACTION)]
        raw["media"]["png_base64"] = base64.b64encode(data).decode("ascii")
        status, response = self.post(raw)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["result"]["actions"][0]["element_ref_hint"], "target-0")
        self.assert_advisory_flags(response)

        # Replace the required black patch with visible pixels and recompute digest.
        bad_image = Image.open(io.BytesIO(data)).convert("RGB")
        ImageDraw.Draw(bad_image).rectangle((700, 500, 711, 511), fill=(30, 30, 30))
        encoded = io.BytesIO()
        bad_image.save(encoded, format="PNG")
        raw["media"]["png_base64"] = base64.b64encode(encoded.getvalue()).decode("ascii")
        raw["media"]["sha256"] = hashlib.sha256(encoded.getvalue()).hexdigest()
        before = len(self.engine.calls)
        status, response = self.post(raw)
        self.assertEqual(status, 400, response)
        self.assertEqual(len(self.engine.calls), before)

    def test_h29_excluded_overlap_rejects_but_disjoint_redaction_succeeds(self):
        raw, _, bounds, other = fixture()
        raw["policy_request"]["observation"]["accessibility"].append(
            {"element_ref": "excluded-overlap", "role": "Button", "name": "Other",
             "bounds": [bounds[0] + 20, bounds[1] + 10, 100, 40]})
        redaction_x, redaction_y = bounds[0] + 105, bounds[1] + 20
        overlap_image = Image.open(io.BytesIO(base64.b64decode(
            raw["media"]["png_base64"]))).convert("RGB")
        ImageDraw.Draw(overlap_image).rectangle((redaction_x, redaction_y,
                                                redaction_x + 1, redaction_y + 1), fill=(0, 0, 0))
        stream = io.BytesIO()
        overlap_image.save(stream, format="PNG")
        raw["media"]["png_base64"] = base64.b64encode(stream.getvalue()).decode("ascii")
        raw["media"]["sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
        raw["policy_request"]["observation"]["image"]["redactions"] = [
            [redaction_x, redaction_y, 2, 2]]
        before = len(self.engine.calls)
        status, response = self.post(raw)
        self.assertEqual(status, 502, response)
        self.assertEqual(response["error"]["code"], "invalid_backend_output")
        self.assertEqual(len(self.engine.calls), before + 1)

        raw, _, bounds, other = fixture(image_id=3)
        raw["policy_request"]["observation"]["accessibility"].append(
            {"element_ref": "excluded-disjoint", "role": "Button", "name": "Other",
             "bounds": list(other)})
        disjoint_image = Image.open(io.BytesIO(base64.b64decode(
            raw["media"]["png_base64"]))).convert("RGB")
        redaction_x, redaction_y = other[0] + 5, other[1] + 5
        ImageDraw.Draw(disjoint_image).rectangle((redaction_x, redaction_y,
                                                 redaction_x + 1, redaction_y + 1), fill=(0, 0, 0))
        stream = io.BytesIO()
        disjoint_image.save(stream, format="PNG")
        raw["media"]["png_base64"] = base64.b64encode(stream.getvalue()).decode("ascii")
        raw["media"]["sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
        raw["policy_request"]["observation"]["image"]["redactions"] = [
            [redaction_x, redaction_y, 2, 2]]
        status, response = self.post(raw)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["result"]["actions"][0]["element_ref_hint"], "target-3")
        self.assert_advisory_flags(response)

    def test_consecutive_frames_use_distinct_pixels_and_identity(self):
        observed_points = []
        for call_index, (image_id, rtl) in enumerate(((4, False), (5, True))):
            raw, data, bounds, _ = fixture(locale="ar-SA" if rtl else "en-US",
                                           rtl=rtl, image_id=image_id)
            status, response = self.post(raw)
            self.assertEqual(status, 200, response)
            proposal = response["result"]
            self.assertEqual(proposal["media_sha256"], hashlib.sha256(data).hexdigest())
            expected_bindings = {
                "request_id": raw["policy_request"]["request_id"],
                "tenant_id": "tenant-1",
                "task_id": raw["policy_request"]["task"]["task_id"],
                "device_id": raw["policy_request"]["device"]["device_id"],
                "session_id": raw["policy_request"]["device"]["session_id"],
                "frame_id": raw["policy_request"]["observation"]["generation"],
            }
            for field, expected in expected_bindings.items():
                self.assertEqual(proposal[field], expected)
            target_ref = raw["policy_request"]["observation"]["accessibility"][0]["element_ref"]
            self.assertEqual(proposal["actions"][0]["element_ref_hint"], target_ref)

            # Compute the expected decoder output from the fixture rectangle,
            # independently of the test double's output.
            x, y, width, height = bounds
            center_x, center_y = x + (width - 1) // 2, y + (height - 1) // 2
            expected_point = ((center_x * 1000 // 800) * 800 // 1000,
                              (center_y * 1000 // 600) * 600 // 1000)
            point = proposal["actions"][0]["point"]
            self.assertEqual(point, list(expected_point))
            observed_points.append(point)

            self.assertEqual(len(self.engine.calls), call_index + 1)
            expected_pixel_hash = hashlib.sha256(
                Image.open(io.BytesIO(data)).convert("RGB").tobytes()).hexdigest()
            self.assertEqual(self.engine.calls[call_index][0], expected_pixel_hash)
            self.assertEqual(self.engine.calls[call_index][1], (800, 600))
            self.assert_advisory_flags(response)
        self.assertNotEqual(observed_points[0], observed_points[1])


if __name__ == "__main__":
    unittest.main()
