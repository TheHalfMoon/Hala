"""CPU candidate contracts with fake inference; no model qualification."""
import base64
import copy
import hashlib
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from PIL import Image
from hala.inference import RequestContext, ServiceError
from hala.inference.vision_candidate import CandidateBackend
from hala.serving import Config, Service
from tests import test_serving as transport
payload, TOKEN = transport.payload, transport.TOKEN


def candidate_payload():
    request = payload()
    request['allowed_actions'] = ['click', 'abstain']
    stream = io.BytesIO()
    Image.new('RGB', (800, 600)).save(stream, format='PNG')
    image = stream.getvalue()
    return {'schema': 'hala.image_request.v2', 'policy_request': request,
            'media': {'png_base64': base64.b64encode(image).decode(),
                      'sha256': hashlib.sha256(image).hexdigest(),
                      'frame_id': 'obs-1', 'request_id': 'r-1', 'tenant_id': 'tenant-1',
                      'task_id': 'task-1', 'device_id': 'device-1', 'session_id': 'session-1',
                      'origin': [0, 0], 'dpi': 96}}


class FakeEngine:
    provenance = {'engine': 'cpu_fake', 'model_id': 'cpu-contract-fixture',
                  'model_revision': 'fixture-v1', 'processor_revision': 'fixture-v1',
                  'precision': 'none', 'prompt_revision': 'h16-v2-normalized-point-v1'}
    def __init__(self):
        self.calls = 0
        self.output = '{"point":[63,59]}'
    def __call__(self, image, request, context):
        self.calls += 1
        self.request = request
        return copy.deepcopy(self.output)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.service = Service(Config(TOKEN, 'tenant-1'), candidate_backend=CandidateBackend(self.engine))
        self.raw = candidate_payload()
    def infer(self):
        return self.service.infer('v2_action', self.raw, RequestContext(time.monotonic()+2))
    def rejected(self, code='invalid_request'):
        with self.assertRaises(ServiceError) as caught:
            self.infer()
        self.assertEqual(caught.exception.code, code)
    def test_candidate_is_explicit_unqualified_advice(self):
        result = self.infer()
        self.assertEqual(result['api_version'], 'hala.v2')
        self.assertFalse(result['result']['provenance']['qualified'])
        self.assertEqual(result['result']['provenance']['engine'], 'cpu_fake')
        self.assertEqual(result['result']['actions'][0]['point'], [50, 35])
        self.assertFalse(result['result']['execution_authorized'])
        self.assertIsNone(result['result']['confidence'])
        self.assertEqual(self.engine.request, {'goal':'Click Save'})
    def test_media_binding_and_tamper_precede_engine(self):
        for field in ['sha256', 'frame_id', 'request_id', 'tenant_id', 'task_id', 'device_id', 'session_id']:
            with self.subTest(field=field):
                self.raw = candidate_payload()
                self.raw['media'][field] = 'wrong'
                self.rejected()
        self.assertEqual(self.engine.calls, 0)
    def test_frame_scale_origin_and_png_dimensions(self):
        for mutation in [('dpi', 120), ('origin', [1, 0]), ('png_base64', 'invalid')]:
            self.raw = candidate_payload()
            self.raw['media'][mutation[0]] = mutation[1]
            self.rejected()
        for field, value in [('scale', 1.5), ('width', 801), ('format', 'jpeg')]:
            self.raw = candidate_payload()
            self.raw['policy_request']['observation']['image'][field] = value
            self.rejected()
        self.assertEqual(self.engine.calls, 0)
    def test_policy_ceiling_budget_protected_and_redacted(self):
        for kind in ['ceiling', 'budget', 'protected', 'window', 'redacted', 'allowed']:
            self.raw = candidate_payload()
            request = self.raw['policy_request']
            if kind == 'ceiling': request['method_ceiling'] = 'STRUCTURED_ONLY'
            if kind == 'budget': request['task']['step_index'] = 10
            if kind == 'protected': request['observation']['accessibility'][0]['protected'] = True
            if kind == 'window': request['observation']['window']['protected'] = True
            if kind == 'redacted': request['observation']['image']['redactions'] = [[10,20,100,30]]
            if kind == 'allowed': request['allowed_actions'] = ['observe']
            self.rejected()
        self.assertEqual(self.engine.calls, 0)
    def test_output_nonfinite_unknown_unobserved_and_outside(self):
        for output in ['{"point":[NaN,59]}', '{"point":[63,59],"element_ref":"unknown"}',
                       '{"point":[700,59]}', '{"point":[Infinity,59]}',
                       '{"point":[63,59],"success":true}', '{"point":[63,59],"confidence":0.5}',
                       '{"point":[63,59],"point":[63,59]}', '{"point":[true,59]}',
                       '{"point":[63.0,59]}', '{"point":[1001,59]}',
                       'prefix {"point":[63,59]}', '```\n{"point":[63,59]}\n```']:
            self.engine.output = output
            self.rejected('invalid_backend_output')

    def test_explicit_json_fence_and_server_target_mapping(self):
        self.engine.output = '```json\n{"point":[63,59]}\n```'
        result = self.infer()['result']
        self.assertEqual(result['actions'][0]['element_ref_hint'], 'save-1')
        self.assertEqual(result['actions'][0]['point'], [50,35])

    def test_ambiguous_or_overlapping_protected_target(self):
        for protected in [False, True]:
            self.raw = candidate_payload()
            self.raw['policy_request']['observation']['accessibility'].append(
                {'element_ref':'other', 'role':'Button', 'name':'Other',
                 'bounds':[40,30,20,15], 'protected':protected})
            self.rejected('invalid_backend_output')

    def test_decoder_generalized_dimensions_and_axis_endpoints(self):
        from hala.inference.vision_candidate import decode_point
        self.assertEqual(decode_point('{"point":[500,250]}', 640, 480), (320,120))
        self.assertEqual(decode_point('{"point":[1000,1000]}', 640, 480), (639,479))
        self.assertEqual(decode_point('{"point":[0,0]}', 640, 480), (0,0))
    def test_tenant_cancel_deadline_unsupported_default(self):
        self.raw['policy_request']['tenant_id'] = 'other'
        self.rejected('tenant_mismatch')
        self.raw = candidate_payload()
        self.raw['policy_request']['cancelled'] = True
        self.rejected('cancelled')
        with self.assertRaises(ServiceError):
            self.service.infer('v2_action', candidate_payload(), RequestContext(time.monotonic()-1))
        for responsibility in ['v2_plan', 'v2_decision']:
            with self.assertRaises(ServiceError) as caught:
                self.service.infer(responsibility, candidate_payload(), RequestContext(time.monotonic()+2))
            self.assertEqual(caught.exception.code, 'unsupported_responsibility')
        default = Service(Config(TOKEN, 'tenant-1'))
        with self.assertRaises(ServiceError) as caught:
            default.infer('v2_action', candidate_payload(), RequestContext(time.monotonic()+2))
        self.assertEqual(caught.exception.code, 'backend_unqualified')

    def test_malformed_media_pixels_and_caller_qualification(self):
        for change in ['unknown', 'pixels', 'oversize', 'invalid_png', 'qualified']:
            self.raw = candidate_payload()
            if change == 'unknown': self.raw['media']['secret'] = 'extra'
            if change == 'pixels': self.raw['policy_request']['observation']['image'].update(width=8192, height=8192)
            if change == 'oversize': self.raw['media']['png_base64'] = 'A' * 700000
            if change == 'invalid_png':
                data = b'\x89PNG\r\n\x1a\ninvalid'
                self.raw['media']['png_base64'] = base64.b64encode(data).decode()
                self.raw['media']['sha256'] = hashlib.sha256(data).hexdigest()
            if change == 'qualified': self.raw['qualified'] = True
            self.rejected()
        self.assertEqual(self.engine.calls, 0)

    def test_context_after_engine_and_output_bound(self):
        class CancelEngine(FakeEngine):
            def __call__(self, image, request, context):
                context.cancelled.set()
                return self.output
        self.service.candidate_backend = CandidateBackend(CancelEngine())
        self.rejected('cancelled')
        self.service = Service(Config(TOKEN, 'tenant-1', max_output_bytes=160),
                               candidate_backend=CandidateBackend(FakeEngine()))
        self.rejected('output_too_large')

    def test_local_loader_rejected_before_dependencies(self):
        from hala.inference.remote_qwen import RemoteQwenEngine
        with self.assertRaises(ValueError):
            RemoteQwenEngine.load('.')

    def test_remote_loader_provider_and_optin_guards_before_ml_imports(self):
        from hala.inference.remote_qwen import RemoteQwenEngine
        cases = [({}, '/content/cache'),
                 ({'HALA_REMOTE_SERVING_ALLOWED':'1'}, '/content/cache'),
                 ({'HALA_REMOTE_SERVING_ALLOWED':'1', 'COLAB_RELEASE_TAG':'release',
                   'KAGGLE_KERNEL_RUN_TYPE':'Interactive'}, '/content/cache'),
                 ({'HALA_REMOTE_SERVING_ALLOWED':'1', 'COLAB_RELEASE_TAG':'release'}, '/kaggle/working/cache'),
                 ({'HALA_REMOTE_SERVING_ALLOWED':'1', 'COLAB_RELEASE_TAG':'release'}, '/tmp/other-cache')]
        for environment, cache in cases:
            with self.subTest(environment=environment, cache=cache), patch.dict('os.environ', environment, clear=True), patch('hala.inference.remote_qwen.platform.system', return_value='Linux'):
                with self.assertRaises(ValueError):
                    RemoteQwenEngine.load(cache)
        with patch.dict('os.environ', {'HALA_REMOTE_SERVING_ALLOWED':'1'}, clear=True), patch('hala.inference.remote_qwen.platform.system', return_value='Linux'), patch('hala.inference.remote_qwen.Path.exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Kaggle commercial-use'):
                RemoteQwenEngine.load('/content/cache')


class V2TransportTests(unittest.TestCase):
    setUp = transport.ServerTests.setUp
    tearDown = transport.ServerTests.tearDown
    start = transport.ServerTests.start
    call = transport.ServerTests.call
    def test_v2_auth_and_default_gate(self):
        body = __import__('json').dumps(candidate_payload())
        self.assertEqual(self.call('POST', '/v2/hala/action', body, token='wrong')[0], 401)
        self.assertEqual(self.call('POST', '/v2/hala/action', body)[0], 503)
    def test_v2_transport_candidate(self):
        self.service.candidate_backend = CandidateBackend(FakeEngine())
        status, result = self.call('POST', '/v2/hala/action', __import__('json').dumps(candidate_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(result['api_version'], 'hala.v2')

    def test_v2_deadline_retains_capacity_until_worker_exit(self):
        self.start(max_concurrency=1, request_timeout_s=.08)
        started, release = threading.Event(), threading.Event()
        class Blocking(FakeEngine):
            def __call__(self, image, request, context):
                started.set()
                release.wait(2)
                return self.output
        self.service.candidate_backend = CandidateBackend(Blocking())
        results = []
        worker = threading.Thread(target=lambda: results.append(self.call('POST', '/v2/hala/action', json.dumps(candidate_payload()))))
        worker.start()
        try:
            self.assertTrue(started.wait(1))
            worker.join(1)
            self.assertEqual(results[0][0], 408)
            self.assertEqual(self.call(path='/health')[0], 503)
        finally:
            release.set()
            worker.join(2)
        self.assertTrue(self.server.wait_for_idle(1))

    def test_v2_disconnect_cancels_and_retains_capacity(self):
        self.start(max_concurrency=1)
        started, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        class Blocking(FakeEngine):
            def __call__(self, image, request, context):
                started.set()
                if context.cancelled.wait(1): cancelled.set()
                release.wait(1)
                return self.output
        self.service.candidate_backend = CandidateBackend(Blocking())
        body = json.dumps(candidate_payload()).encode()
        client = socket.create_connection(self.server.server_address, timeout=2)
        wire = (f'POST /v2/hala/action HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode()+body
        try:
            client.sendall(wire)
            self.assertTrue(started.wait(1))
            client.close()
            self.assertTrue(cancelled.wait(1))
            self.assertEqual(self.call(path='/health')[0], 503)
        finally:
            client.close()
            release.set()
        self.assertTrue(self.server.wait_for_idle(1))
