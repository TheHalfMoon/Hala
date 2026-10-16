"""Private experimental operator process. No public startup or qualification path."""
import base64
import io
import queue
import subprocess
import sys
import threading
import time
import socket

from hala.inference import ServiceError
from hala.inference.remote_qwen import RemoteQwenEngine
from hala.serving import encode, strict_json

PROVENANCE = dict(RemoteQwenEngine.provenance)
CACHE = '/tmp/hala-h23-cache'
MAX_MESSAGE = 1048576


class CancellationReceipt:
    """Timestamp transport's actual Event.set without changing its semantics."""
    def __init__(self,event):
        self.event, self.lock = event, threading.Lock()
        self.received_monotonic = None

    def set(self):
        with self.lock:
            if not self.event.is_set(): self.received_monotonic = time.monotonic()
            self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self,timeout=None):
        return self.event.wait(timeout)


def admission(provider):
    from evaluation.development_runtime import evaluation_guard
    from evaluation.development_precision import CACHE_KEYS, disk_bytes
    from pathlib import Path
    import os
    result = evaluation_guard(provider)
    root = Path(CACHE)
    if root.is_symlink() or root.resolve() != root:
        raise ValueError('Exact ephemeral H23 cache required')
    for key in CACHE_KEYS:
        if os.environ.get(key) and Path(os.environ[key]).resolve() != root:
            raise ValueError('Cache escape forbidden')
    disk_bytes(root)
    return result


class WorkerEngine:
    """Single owned child, no queue; restart is explicit and never automatic retry."""
    provenance = PROVENANCE

    def __init__(self, provider='colab', font=None, freeze=None, source_commit=None):
        self.provider = provider
        self.font, self.freeze, self.source_commit = font, freeze, source_commit
        self.control = None
        self.process = None
        self.usable = False
        self.reap_state = {'confirmed':False,'child_pid':None,'returncode':None,'error_type':None}
        self.lock = threading.Lock()
        self.trace = []
        self.wire_trace = []
        self.last_fault = None
        self.reader = None

    def _spawn(self):
        from evaluation.serving_lifecycle import preflight
        preflight(self.provider, self.font, self.freeze, self.source_commit)
        parent, child = socket.socketpair()
        try:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'hala.serving.lifecycle_worker',
                self.provider, str(self.font), self.freeze, self.source_commit, str(child.fileno())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                pass_fds=(child.fileno(),))
            self.control = parent
            return process
        except Exception:
            parent.close()
            raise
        finally:
            child.close()

    def _capture_cancel(self, received):
        """Bounded private status query before terminating the owned CUDA process."""
        if self.control is None: return {'evidence':'unavailable'}
        try:
            deadline = time.monotonic()+.25
            self.control.settimeout(.25)
            self.control.sendall(encode({'cancel_received_monotonic':received})+b'\n')
            raw = b''
            while not raw.endswith(b'\n') and len(raw)<=4096:
                remaining = deadline-time.monotonic()
                if remaining<=0: raise TimeoutError('Control deadline')
                self.control.settimeout(remaining)
                part = self.control.recv(4097-len(raw))
                if not part: raise OSError('Control EOF')
                raw += part
            if len(raw)>4096: raise ValueError('Control bound')
            return strict_json(raw)
        except Exception as error:
            return {'evidence':'unavailable','capture_error':type(error).__name__}

    def start(self, timeout=300):
        if not 0 < timeout <= 600:
            raise ValueError('Bounded load deadline required')
        if not self.lock.acquire(blocking=False):
            raise ServiceError('busy')
        created = False
        try:
            if self.process is not None:
                raise ValueError('Close before restart')
            self.usable = False
            self.process = self._spawn()
            created = True
            self.reap_state = {'confirmed':False,'child_pid':self.process.pid,'returncode':None,'error_type':None}
            messages = queue.Queue(maxsize=8)
            self.messages = messages
            process = self.process
            def read():
                try:
                    while True:
                        raw = process.stdout.readline(MAX_MESSAGE+1)
                        value = strict_json(raw) if raw and len(raw) <= MAX_MESSAGE else {'pipe_error':'eof' if not raw else 'oversize'}
                        if 'event' in value:
                            if len(self.wire_trace) >= 32: return
                            self.wire_trace.append(dict(value, received_monotonic=time.monotonic()))
                        messages.put_nowait(value)
                        if 'pipe_error' in value: return
                except Exception:
                    try: messages.put_nowait({'pipe_error':True})
                    except queue.Full: pass
            self.reader = threading.Thread(target=read, daemon=True)
            self.reader.start()
            ready = messages.get(timeout=timeout)
            if ready != {'ready': PROVENANCE}:
                raise ServiceError('invalid_backend_output')
            self.usable = True  # Ownership alone never admits inference.
        except Exception:
            if created: self.close()
            raise
        finally:
            self.lock.release()

    def close(self):
        """Terminate only the exact child handle this object created; bounded reap."""
        self.usable = False
        process = self.process
        try:
            if process is not None:
                self.reap_state = {'confirmed':False,'child_pid':process.pid,'returncode':None,'error_type':None}
                if process.poll() is None: process.terminate()
                try: returncode = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=2)
                if type(returncode) is not int: raise RuntimeError('Child exit not confirmed')
                self.reap_state.update(confirmed=True,returncode=returncode)
                self.process = None  # Retain the exact handle until wait confirms reap.
        except Exception as error:
            self.reap_state['error_type'] = type(error).__name__
            raise
        finally:
            # Buffered stream.close may block behind a reader/writer while a
            # child is unreaped. Retain those exact streams for the later retry.
            if process is not None and self.reap_state['confirmed']:
                for stream in (process.stdin, process.stdout):
                    if stream is not None: stream.close()
            if self.reader is not None and self.reader is not threading.current_thread():
                self.reader.join(timeout=1)
            if self.control is not None:
                self.control.close()
                self.control = None

    def __call__(self, image, request, context):
        # The HTTP handler and this worker share the same context. Timestamp its
        # real cancellation signal, rather than the later polling observation.
        context.cancelled = CancellationReceipt(context.cancelled)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        value = {'png':base64.b64encode(stream.getvalue()).decode('ascii'),
                 'request':request, 'timeout':max(.001, context.deadline-time.monotonic())}
        return self._exchange(value, context)

    def allocator_probe(self, context):
        """Private operator operation; never accessible through v2 request fields."""
        return self._exchange({'probe':'allocator_budget'}, context, probe=True)

    def _exchange(self, value, context, probe=False):
        context.check()
        if not self.lock.acquire(blocking=False): raise ServiceError('busy')
        if not self.usable or self.process is None:
            self.lock.release()
            raise ServiceError('backend_unqualified')
        try:
            self.trace, self.wire_trace = [], []
            raw = encode(value)+b'\n'
            if len(raw) > MAX_MESSAGE: raise ServiceError('invalid_request')
            # One bounded request fits the private pipe; a writer must not block cleanup.
            completed = threading.Event()
            failed = []
            process = self.process
            def write():
                try:
                    process.stdin.write(raw)
                    process.stdin.flush()
                except (OSError, ValueError): failed.append(True)
                finally: completed.set()
            writer = threading.Thread(target=write, daemon=True)
            writer.start()
            while True:
                context.check()
                if failed: raise ServiceError('internal_error')
                try: value = self.messages.get(timeout=.005)
                except queue.Empty: continue
                if 'event' in value:
                    if len(self.trace) >= 32: raise ServiceError('invalid_backend_output')
                    self.trace.append(dict(value, observed_monotonic=time.monotonic()))
                    continue
                if probe and set(value) == {'probe_result'}:
                    result = value['probe_result']
                    if (type(result) is not dict or set(result) != {'exhausted','kind','requested_bytes'} or
                            type(result['exhausted']) is not bool or result['kind'] != 'allocator_budget' or
                            result['requested_bytes'] != 8388608):
                        raise ServiceError('invalid_backend_output')
                    context.check()
                    return result
                if set(value) == {'output'} and type(value['output']) is str and len(value['output'].encode()) <= 2048:
                    context.check()
                    return value['output']
                if value == {'error':'busy'}: raise ServiceError('busy')
                if value == {'error':'internal_error'}: raise ServiceError('internal_error')
                if 'pipe_error' in value:
                    raise ServiceError('internal_error' if value['pipe_error'] == 'eof' else 'invalid_backend_output')
                raise ServiceError('invalid_backend_output')
        except Exception as error:
            self.usable = False
            observed = time.monotonic()
            receipt = getattr(context.cancelled,'received_monotonic',None)
            received = receipt if receipt is not None else observed
            ack = self._capture_cancel(received) if isinstance(error,ServiceError) and error.code in ('cancelled','deadline_exceeded') else None
            self.last_fault = {'code':error.code if isinstance(error,ServiceError) else 'internal_error',
                               'observed_monotonic':observed,'cancel_received_monotonic':received,
                               'receipt_precision':'event_set' if receipt is not None else 'engine_observed_only',
                               'cancellation_ack':ack,
                               'child_pid':self.process.pid if self.process else None}
            try:
                self.close()
            finally:
                self.last_fault['reap'] = dict(self.reap_state)
            raise
        finally:
            self.lock.release()
