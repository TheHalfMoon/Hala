"""CPU process/HTTP contracts only; no actual model, CUDA or OOM evidence."""
import subprocess
import json
import socket
import hashlib
import queue
from types import SimpleNamespace
import sys
import threading
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from hala.inference import RequestContext, ServiceError
from hala.serving.lifecycle import WorkerEngine, PROVENANCE
from PIL import Image


class WorkflowTests(unittest.TestCase):
    def test_operator_preflight_rejects_bad_font_before_start(self):
        from evaluation import serving_lifecycle as operator
        arguments = ['operator','--provider','kaggle','--freeze-sha256','a'*64,
                     '--source-commit','b'*40,'--font','missing.ttf','--output','unused']
        with patch.object(sys,'argv',arguments), patch.object(operator,'verify_sources'), \
                patch.object(operator,'verify_git'), patch('hala.serving.lifecycle.admission'), \
                patch('importlib.metadata.version',return_value='11.3.0'), \
                patch('hala.serving.lifecycle.WorkerEngine.start') as start:
            with self.assertRaises((OSError,ValueError)): operator.main()
            start.assert_not_called()

    def test_restart_font_and_raqm_gates_precede_start(self):
        from evaluation.serving_lifecycle import restart_worker
        engine = unittest.mock.Mock()
        with tempfile.TemporaryDirectory() as root:
            font = Path(root)/'font.ttf'
            font.write_bytes(b'unpinned')
            for version,raqm in [('11.3.0',True),('11.3.0',False),('0.0.0',True)]:
                with self.subTest(version=version,raqm=raqm), \
                        patch('importlib.metadata.version',return_value=version), \
                        patch('PIL.features.check_feature',return_value=raqm):
                    with self.assertRaises(ValueError): restart_worker(engine,font)
                    engine.start.assert_not_called()

    def test_source_closure_rejects_unfrozen_dependency_and_dirty_git(self):
        from evaluation import serving_lifecycle as operator
        self.assertIn('evaluation/heldout_fixtures.py',operator.source_files())
        self.assertIn('evaluation/__init__.py',operator.source_files())
        with patch('subprocess.run') as run:
            run.return_value.stdout = 'b'*40+'\n'
            with self.assertRaises(ValueError): operator.verify_git('a'*40)
            run.side_effect = [SimpleNamespace(stdout='a'*40+'\n'),SimpleNamespace(stdout=' M evaluation/__init__.py\n')]
            with self.assertRaises(ValueError): operator.verify_git('a'*40)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dependency = root/'evaluation/__init__.py'
            dependency.parent.mkdir()
            dependency.write_bytes(b'original')
            freeze = root/'experiments/serving-lifecycle/source-freeze.json'
            freeze.parent.mkdir(parents=True)
            freeze.write_text(json.dumps({'files':{'evaluation/__init__.py':hashlib.sha256(b'original').hexdigest()}}))
            expected = hashlib.sha256(freeze.read_bytes()).hexdigest()
            with patch.object(operator,'ROOT',root), patch.object(operator,'source_files',return_value=['evaluation/__init__.py']):
                operator.verify_sources(expected)
                dependency.write_bytes(b'tampered')
                with self.assertRaises(ValueError): operator.verify_sources(expected)

    def test_fault_probe_progress_survives_reload_failure(self):
        from evaluation.serving_lifecycle import probe_lifecycle
        engine = unittest.mock.Mock()
        engine.allocator_probe.return_value = {'exhausted':True}
        saved = []
        with patch('evaluation.serving_lifecycle.probe_fault',return_value={'child_reaped':True}), \
                patch('evaluation.serving_lifecycle.restart_worker',side_effect=RuntimeError('reload failed')):
            with self.assertRaises(RuntimeError): probe_lifecycle(engine,'font',saved.append)
        self.assertTrue(saved[-1]['resource']['exhausted'])
        self.assertIn('disconnect',saved[-1])
    def test_client_rechecks_stale_binding_and_policy_before_synthetic_apply(self):
        from evaluation.serving_lifecycle import Workflow, tasks, apply_proposal
        from hala.serving import Config, Service
        from hala.inference.vision_candidate import CandidateBackend
        from tests.test_serving_v2 import FakeEngine
        with tempfile.TemporaryDirectory() as root:
            workflow = Workflow(tasks()[0],Path(root)/'state')
            payload = workflow.payload(Image.new('RGB',(800,600)))
            engine = FakeEngine()
            engine.output = '{"point":[250,300]}'
            proposal = Service(Config('x'*32,'h23'),candidate_backend=CandidateBackend(engine)).infer(
                'v2_action',payload,RequestContext(time.monotonic()+2))['result']
            proposal['frame_id'] = 'stale'
            with self.assertRaises(ValueError): apply_proposal(workflow,payload,proposal)
            self.assertEqual(workflow.stage,0)
            self.assertFalse(workflow.path.exists())

    def test_unsafe_or_stale_proposals_cannot_persist(self):
        from evaluation.serving_lifecycle import run_tasks
        from tests.test_serving_v2 import FakeEngine
        from hala.serving import Service, Config, create_server
        from hala.inference.vision_candidate import CandidateBackend
        for output in ('{"point":[700,300]}','{"point":[250,300],"done":true}'):
            engine = FakeEngine()
            engine.output = output
            server = create_server(Service(Config('x'*32,'h23'),candidate_backend=CandidateBackend(engine)),port=0)
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as root:
                    records = run_tasks(server.server_address[1],'x'*32,root,renderer=lambda *args: Image.new('RGB',(800,600)))
                    self.assertEqual([r['verified'] for r in records], [False,False])
                    self.assertEqual([r['error'] for r in records], ['invalid_backend_output']*2)
                    self.assertEqual(list(Path(root).iterdir()),[])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_font_bytes_and_raqm_fail_closed_before_render(self):
        from evaluation.serving_lifecycle import Workflow, tasks, render
        with tempfile.TemporaryDirectory() as root:
            font = Path(root)/'font.ttf'
            font.write_bytes(b'unpinned')
            with self.assertRaises(ValueError): render(Workflow(tasks()[0],Path(root)/'state'),font)

    def test_causal_three_turns_and_independent_persisted_outcomes(self):
        from evaluation.serving_lifecycle import Workflow, verify_outcome, tasks
        with tempfile.TemporaryDirectory() as root:
            for task in tasks():
                workflow = Workflow(task, Path(root)/task['id'])
                self.assertFalse(verify_outcome(workflow.path, task))
                with self.assertRaises(ValueError): workflow.apply([400,220])
                for _ in range(3): workflow.apply(workflow.target_center())
                self.assertTrue(verify_outcome(workflow.path, task))
                self.assertFalse(verify_outcome(workflow.path, tasks()[1 if task==tasks()[0] else 0]))
                self.assertEqual(workflow.stage, 3)
    def test_loopback_six_proposals_auth_and_policy_still_authoritative(self):
        from evaluation.serving_lifecycle import Workflow, run_tasks, tasks
        from hala.serving import Service, Config, create_server
        from hala.inference.vision_candidate import CandidateBackend
        from tests.test_serving_v2 import FakeEngine
        engine = FakeEngine()
        engine.output = '{"point":[250,300]}'
        server = create_server(Service(Config('x'*32, 'h23'), candidate_backend=CandidateBackend(engine)), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                records = run_tasks(server.server_address[1], 'x'*32, root, renderer=lambda *args: Image.new('RGB',(800,600)))
                self.assertEqual([r['verified'] for r in records], [True,True])
                self.assertEqual(engine.calls, 6)
                records = run_tasks(server.server_address[1], 'y'*32, Path(root)/'unauthorized', renderer=lambda *args: Image.new('RGB',(800,600)))
                self.assertEqual([r['error'] for r in records], ['unauthorized']*2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


def fake_child():
    return subprocess.Popen([sys.executable, '-u', '-m', 'tests.lifecycle_child'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


class LifecycleTests(unittest.TestCase):
    def test_unreaped_failed_owner_cannot_serve_queued_stale_output(self):
        self.engine.close()
        owned = unittest.mock.Mock()
        owned.pid = 12345
        owned.poll.return_value = None
        owned.wait.side_effect = subprocess.TimeoutExpired('owned-child',2)
        self.engine.process = owned
        self.engine.messages = queue.Queue()
        self.engine.messages.put({'output':'{"point":[500,500]}'})
        with self.assertRaises(subprocess.TimeoutExpired): self.engine.close()
        with self.assertRaises(ServiceError) as caught: self.call()
        self.assertEqual(caught.exception.code,'backend_unqualified')
        self.assertFalse(self.engine.usable)
        self.assertIs(self.engine.process,owned)
        owned.stdin.write.assert_not_called()
        self.assertEqual(owned.wait.call_count,2)  # Rejection does not retry cleanup.
        self.assertEqual(self.engine.messages.qsize(),1)
        owned.wait.side_effect = None
        owned.wait.return_value = -9
        self.engine.close()
        with self.assertRaises(ServiceError): self.call()
        with patch.object(self.engine,'_spawn',side_effect=fake_child): self.engine.start(timeout=3)
        self.assertTrue(self.engine.usable)
        self.assertEqual(self.call(),'{"point":[500,500]}')

    def test_double_reap_timeout_retains_handle_blocks_restart_then_allows_retry(self):
        self.engine.close()
        owned = unittest.mock.Mock()
        owned.pid = 12345
        owned.poll.return_value = None
        owned.wait.side_effect = subprocess.TimeoutExpired('owned-child',2)
        self.engine.process = owned
        with self.assertRaises(subprocess.TimeoutExpired): self.engine.close()
        self.assertIs(self.engine.process,owned)
        self.assertFalse(self.engine.reap_state['confirmed'])
        self.assertEqual(self.engine.reap_state['error_type'],'TimeoutExpired')
        self.assertEqual([call.kwargs['timeout'] for call in owned.wait.call_args_list],[2,2])
        owned.kill.assert_called_once()
        owned.stdin.close.assert_not_called()  # Avoid buffered I/O locks before reap.
        owned.stdout.close.assert_not_called()
        with patch.object(self.engine,'_spawn') as spawn:
            with self.assertRaises(ValueError): self.engine.start()
            spawn.assert_not_called()
        self.assertIs(self.engine.process,owned)
        self.assertEqual(owned.wait.call_count,2)
        owned.wait.side_effect = None
        owned.wait.return_value = -9
        self.engine.close()
        self.assertIsNone(self.engine.process)
        self.assertTrue(self.engine.reap_state['confirmed'])
        self.assertEqual(self.engine.reap_state['returncode'],-9)
        self.assertEqual(self.engine.reap_state['child_pid'],12345)
        owned.stdin.close.assert_called_once()
        owned.stdout.close.assert_called_once()

    def setUp(self):
        self.engine = WorkerEngine()
        with patch.object(self.engine, '_spawn', side_effect=fake_child):
            self.engine.start(timeout=3)
    def tearDown(self):
        self.engine.close()
    def call(self, goal='ok', duration=2):
        return self.engine(Image.new('RGB', (16,16)), {'goal':goal}, RequestContext(time.monotonic()+duration))
    def test_sequential_calls_exact_revision(self):
        for _ in range(3): self.assertEqual(self.call(), '{"point":[500,500]}')
        self.assertEqual(self.engine.provenance, PROVENANCE)
    def test_cancel_after_child_started_fails_closed_and_restart(self):
        self.engine.control, peer = socket.socketpair()
        def acknowledge():
            try:
                raw = peer.recv(4096)
                received = json.loads(raw)['cancel_received_monotonic']
                peer.sendall(json.dumps({'evidence':'cpu_fixture_only','event_pending':True,
                                         'cancel_received_monotonic':received}).encode()+b'\n')
            finally: peer.close()
        threading.Thread(target=acknowledge,daemon=True).start()
        context = RequestContext(time.monotonic()+3)
        errors = []
        def run():
            try: self.engine(Image.new('RGB',(16,16)), {'goal':'wait'}, context)
            except ServiceError as exc: errors.append(exc.code)
        thread = threading.Thread(target=run)
        thread.start()
        until = time.monotonic()+2
        while not self.engine.trace and time.monotonic()<until: time.sleep(.005)
        self.assertEqual(self.engine.trace[-1]['event'], 'cpu_work_started')
        context.cancelled.set()
        thread.join(2)
        self.assertEqual(errors, ['cancelled'])
        self.assertEqual(self.engine.last_fault['cancellation_ack']['event_pending'],True)
        self.assertEqual(self.engine.last_fault['cancellation_ack']['evidence'],'cpu_fixture_only')
        self.assertEqual(self.engine.last_fault['receipt_precision'],'event_set')
        self.assertIsNone(self.engine.process)
        with patch.object(self.engine, '_spawn', side_effect=fake_child): self.engine.start(timeout=3)
        self.assertEqual(self.call(), '{"point":[500,500]}')

    def test_pending_model_event_ack_selects_only_work_enqueued_before_cancel(self):
        from hala.serving.lifecycle_worker import PendingModelWork, control_status
        tracker = PendingModelWork()
        event = unittest.mock.Mock()
        event.query.return_value = False
        with patch('time.monotonic',return_value=10.0): tracker.record(event,'aten.mm.default',lambda value:None)
        newer = unittest.mock.Mock()
        newer.query.return_value = True
        with patch('time.monotonic',return_value=12.0): tracker.record(newer,'aten.bmm.default',lambda value:None)
        left,right = socket.socketpair()
        thread = threading.Thread(target=control_status,args=(right,tracker),daemon=True)
        thread.start()
        self.engine.control = left
        ack = self.engine._capture_cancel(11.0)
        thread.join(1)
        right.close()
        self.assertTrue(ack['event_pending'])
        self.assertEqual(ack['operation'],'aten.mm.default')
        self.assertLessEqual(ack['event_recorded_monotonic'],ack['cancel_received_monotonic'])
        tracker.reset()
        self.assertEqual(tracker.snapshot(20.0)['evidence'],'unavailable')
    def test_deadline_failure_oversize_and_resource_fail_closed(self):
        for goal, code in [('wait','deadline_exceeded'), ('crash','internal_error'),
                           ('oversize','invalid_backend_output'), ('resource','busy')]:
            with self.subTest(goal=goal):
                with self.assertRaises(ServiceError) as caught: self.call(goal, .1 if goal=='wait' else 2)
                self.assertEqual(caught.exception.code, code)
                self.assertIsNone(self.engine.process)
                with patch.object(self.engine, '_spawn', side_effect=fake_child): self.engine.start(timeout=3)
    def test_revision_mismatch_never_ready(self):
        self.engine.close()
        with patch.object(self.engine, '_spawn', side_effect=fake_child), patch('hala.serving.lifecycle.PROVENANCE', {}):
            with self.assertRaises(ServiceError): self.engine.start(timeout=3)
        self.assertIsNone(self.engine.process)
    def test_local_start_rejected_before_spawn(self):
        self.engine.close()
        with patch.object(self.engine, '_spawn', wraps=self.engine._spawn) as spawn:
            with self.assertRaises(ValueError): self.engine.start()
            spawn.assert_called_once()

    def test_resource_probe_is_private_and_cpu_injected(self):
        from hala.inference import RequestContext
        result = self.engine.allocator_probe(RequestContext(time.monotonic()+2))
        self.assertEqual(result, {'kind':'allocator_budget','exhausted':True,'requested_bytes':8388608})
        self.assertEqual(self.call(), '{"point":[500,500]}')
