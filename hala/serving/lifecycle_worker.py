"""Fixed private child entrypoint; model output has no native execution path."""
import base64
import contextlib
import importlib.metadata
import io
import os
import sys
import threading
import time
import socket
from collections import deque

from hala.inference import RequestContext, ServiceError
from hala.inference.remote_qwen import (RemoteQwenEngine, MODEL_ID, MODEL_REVISION,
                                        TORCH_VERSION, TRANSFORMERS_VERSION)
from hala.serving import encode, strict_json
from hala.serving.lifecycle import admission, CACHE, MAX_MESSAGE, PROVENANCE


def allocator_budget_probe(torch):
    """Actual allocator cap exhaustion, not whole-device physical OOM."""
    previous = torch.cuda.get_per_process_memory_fraction(0)
    allocated = torch.cuda.memory_allocated(0)
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        torch.cuda.empty_cache()
        # Allow existing model plus 1 MiB, then request only 8 MiB.
        torch.cuda.set_per_process_memory_fraction(min(1.0, (allocated+1048576)/total), 0)
        try:
            probe = torch.empty(8*1048576, dtype=torch.uint8, device='cuda:0')
            del probe
            return {'exhausted':False, 'kind':'allocator_budget', 'requested_bytes':8388608}
        except torch.cuda.OutOfMemoryError:
            return {'exhausted':True, 'kind':'allocator_budget', 'requested_bytes':8388608}
    finally:
        torch.cuda.set_per_process_memory_fraction(previous, 0)
        torch.cuda.empty_cache()


class PendingModelWork:
    """Snapshot of an event recorded after actual model CUDA matrix work enqueue."""
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=128)
        self.sequence = 0
        self.announced = False
        self.active = False

    def reset(self):
        with self.lock:
            self.history.clear()
            self.announced = False

    def record(self, event, operation, emit):
        with self.lock:
            self.sequence += 1
            self.history.append((event, operation, self.sequence, time.monotonic()))
            pending = not event.query()
            announce = pending and not self.announced
            self.announced |= pending
        if announce:
            emit({'event':'cuda_model_operation_pending','operation':operation,
                  'operation_sequence':self.sequence,'worker_monotonic':time.monotonic()})

    def snapshot(self, received):
        # No synchronize: this query is deliberately concurrent with generation.
        with self.lock:
            receipt = time.monotonic()
            eligible = [item for item in self.history if item[3] <= received]
            if not eligible:
                return {'evidence':'unavailable','worker_receipt_monotonic':receipt}
            event, operation, sequence, recorded = eligible[-1]
            pending = not event.query()
            return {'evidence':'real_model_cuda_event','cancel_received_monotonic':received,
                    'worker_receipt_monotonic':receipt,'event_recorded_monotonic':recorded,
                    'operation':operation,'operation_sequence':sequence,'event_pending':pending}


def control_status(sock, tracker):
    try:
        raw = b''
        while len(raw)<=4096 and not raw.endswith(b'\n'):
            piece = sock.recv(4097-len(raw))
            if not piece: return
            raw += piece
        value = strict_json(raw)
        if set(value) != {'cancel_received_monotonic'}: return
        ack = tracker.snapshot(value['cancel_received_monotonic'])
        sock.sendall(encode(ack)+b'\n')
    except Exception:
        pass  # Parent records unavailable evidence and still terminates its child.


def load(provider, font, freeze, source_commit, emit):
    from evaluation.serving_lifecycle import preflight
    preflight(provider, font, freeze, source_commit)
    for key in ('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME'):
        os.environ[key] = CACHE
    os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    import torch
    import transformers
    if (torch.__version__.split('+')[0] != TORCH_VERSION or transformers.__version__ != TRANSFORMERS_VERSION
            or importlib.metadata.version('Pillow') != '11.3.0' or not torch.cuda.is_available()):
        raise ValueError('Pinned remote CUDA runtime required')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    engine = RemoteQwenEngine()
    engine.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        trust_remote_code=False, cache_dir=CACHE, min_pixels=256*28*28, max_pixels=1024*28*28)
    engine.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
        revision=MODEL_REVISION, trust_remote_code=False, cache_dir=CACHE,
        dtype=torch.float16, attn_implementation='eager').to('cuda:0').eval()
    engine.torch, engine.lock = torch, threading.Lock()
    engine.pending_work = PendingModelWork()
    from torch.utils._python_dispatch import TorchDispatchMode
    class ModelWork(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            # These are real matrix operations from an active model forward,
            # never instrumentation-only CUDA work or metadata/view operations.
            if (engine.pending_work.active and str(func) in ('aten.mm.default','aten.addmm.default','aten.bmm.default') and
                    isinstance(result,torch.Tensor) and result.is_cuda):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(result.device))
                engine.pending_work.record(event,str(func),emit)
            return result
    engine.model_work = ModelWork
    # One synchronized full model forward proves CUDA work, rather than a CPU timer.
    engine.forward_count = 0
    def before(*unused):
        engine.pending_work.active = True
        engine.forward_count += 1
        if engine.forward_count <= 2:
            engine.cuda_start = torch.cuda.Event(enable_timing=True)
            engine.cuda_start.record()
            emit({'event':'cuda_forward_entered', 'forward':engine.forward_count,
                  'worker_monotonic':time.monotonic()})
    def after(*unused):
        engine.pending_work.active = False
        if engine.forward_count <= 2:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            elapsed = engine.cuda_start.elapsed_time(end)
            emit({'event':'cuda_forward_completed', 'cuda_elapsed_ms':elapsed,
                  'forward':engine.forward_count, 'worker_monotonic':time.monotonic()})
    engine.model.register_forward_pre_hook(before)
    engine.model.register_forward_hook(after)
    return engine


def main():
    wire = sys.stdout.buffer
    def emit(value):
        raw = encode(value)+b'\n'
        if len(raw)>MAX_MESSAGE: raise ValueError('Worker output bound')
        wire.write(raw)
        wire.flush()
    # Dependency prints cannot corrupt the protocol or leak to HTTP.
    with contextlib.redirect_stdout(sys.stderr):
        engine = load(*sys.argv[1:5], emit)
    control = socket.socket(fileno=int(sys.argv[5]))
    threading.Thread(target=control_status,args=(control,engine.pending_work),daemon=True).start()
    emit({'ready':PROVENANCE})
    while True:
        raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
        if not raw: return
        if len(raw)>MAX_MESSAGE: return
        value = strict_json(raw)
        if value == {'probe':'allocator_budget'}:
            emit({'probe_result':allocator_budget_probe(engine.torch)})
            continue
        if set(value) != {'png','request','timeout'}: return
        if (type(value['timeout']) not in (int,float) or not 0 < value['timeout'] <= 60 or
                type(value['request']) is not dict or set(value['request']) != {'goal'}): return
        from PIL import Image
        image_bytes = base64.b64decode(value['png'], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as opened:
            if opened.width*opened.height > 1048576: return
            image = opened.convert('RGB')
        engine.forward_count = 0
        engine.pending_work.reset()
        try:
            with contextlib.redirect_stdout(sys.stderr), engine.model_work():
                output = engine(image, value['request'], RequestContext(time.monotonic()+value['timeout']))
            emit({'event':'cuda_generation_returned', 'worker_monotonic':time.monotonic()})
            emit({'output':output})
        except engine.torch.cuda.OutOfMemoryError:
            emit({'error':'busy'})
            return  # failed model context is never reused
        except Exception:
            emit({'error':'internal_error'})
            return


if __name__ == '__main__':
    main()
