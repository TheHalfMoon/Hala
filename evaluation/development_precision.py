"""Prospective H22 profiles over the unchanged H19 closed-loop callback.

One mode per fresh remote process. CPU checks do not qualify model quality.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from evaluation import development_runtime as runtime
from evaluation.quantization import quantization_config, residency

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/development-precision/config.json'
FREEZE = ROOT / 'experiments/development-precision/source-freeze.json'
CACHE = '/tmp/hala-h22-cache'
PINS = {'accelerate': '1.12.0', 'bitsandbytes': '0.50.2'}
CACHE_KEYS = ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
              'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile(mode):
    if mode not in ('fp16', 'int8', 'nf4'):
        raise ValueError('Prespecified fp16/int8/nf4 profile required')
    return {'schema': 'hala.h22.precision.v1', 'mode': mode,
            'quantization': None if mode == 'fp16' else quantization_config(mode),
            'reference_settings': dict(runtime.SETTINGS), 'packages': dict(PINS)}


def check_profile(selected):
    expected = profile(selected['mode'])
    if json.dumps(selected, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError('Prespecified precision profile required')


def verify_sources(expected):
    runtime.check_digest(expected)
    if digest(FREEZE) != expected:
        raise ValueError('External prospective source freeze mismatch')
    frozen = json.loads(FREEZE.read_text())
    for name, expected_hash in frozen['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected_hash:
            raise ValueError('Prospective source changed: ' + name)
    config = json.loads(CONFIG.read_text())
    if json.dumps(config['profiles'], sort_keys=True) != json.dumps([profile(mode) for mode in ('fp16', 'int8', 'nf4')], sort_keys=True):
        raise ValueError('Prospective profiles changed')
    return config


def check_cache(cache, max_entries=100000, max_seconds=5):
    approved = Path(CACHE).resolve()
    if Path(cache).resolve() != approved or Path(CACHE).is_symlink():
        raise ValueError('Exact non-symlink ephemeral H22 cache required')
    for key in CACHE_KEYS:
        if os.environ.get(key) and Path(os.environ[key]).resolve() != approved:
            raise ValueError('Cache escape forbidden: ' + key)
    # The same bounded incremental scan validates every admitted cache entry.
    # Admission must reject rather than silently trust a partially scanned cache.
    disk_bytes(approved, max_entries=max_entries, max_seconds=max_seconds)


def disk_bytes(root, max_entries=100000, max_seconds=5):
    """Count unique physical files; snapshot symlinks/hardlinks cannot double count."""
    seen, total, count = set(), 0, 0
    started = time.perf_counter()
    root = Path(root).resolve()
    pending = [root] if root.exists() else []
    while pending:
        if time.perf_counter() - started >= max_seconds:
            raise TimeoutError('Cache measurement time budget exceeded')
        # Iterate entries rather than materializing an unbounded directory list.
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                count += 1
                if count > max_entries or time.perf_counter() - started >= max_seconds:
                    raise TimeoutError('Cache measurement entry/time budget exceeded')
                path = Path(entry.path)
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Cache escaped during measurement')
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file():
                    stat = entry.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key not in seen:
                        total += stat.st_size
                        seen.add(key)
    return total


def resolve_device_uuid(torch):
    value = str(torch.cuda.get_device_properties(0).uuid)
    if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        value = 'GPU-' + value
    if re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value) is None:
        raise ValueError('Actual logical cuda:0 physical GPU UUID unavailable')
    return value


def process_sample(device_uuid=None):
    row = {'monotonic_seconds': time.perf_counter(), 'cpu_process_seconds': time.process_time(),
           'cpu_rss_bytes': None, 'device': None, 'device_uuid': device_uuid}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                row['cpu_rss_bytes'] = int(line.split()[1]) * 1024
    except Exception as error:
        row['rss_error'] = type(error).__name__
    if device_uuid is None:
        row['device_error'] = 'DeviceIdentityUnavailable'
        return row
    try:
        output = subprocess.run(['nvidia-smi', '--id=' + device_uuid,
            '--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            check=True, timeout=2).stdout.strip().split(',')
        values = [None if value.strip() in ('N/A', '[N/A]', '[Not Supported]') else float(value) for value in output]
        if len(values) != 4 or any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
            raise ValueError('Unexpected device sample')
        row['device'] = dict(zip(('used_mib', 'free_mib', 'utilization_percent', 'power_watts'), values))
        missing = {key: 'unsupported' for key, value in row['device'].items() if value is None}
        if missing:
            row['device_field_errors'] = missing
    except Exception as error:
        row['device_error'] = type(error).__name__
    return row


class CostSampler:
    """Observer outside inference; at most 1802 samples, subprocess timeout two seconds."""
    def __init__(self, sample=None, limit=1802):
        self.device_uuid = None
        self.sample, self.limit = sample or (lambda: process_sample(self.device_uuid)), min(max(limit, 1), 1802)
        self.rows, self.event = [], threading.Event()
        self.thread = None

    def tick(self):
        try:
            self.rows.append(self.sample())
        except Exception as error:
            self.rows.append({'error_type': type(error).__name__})

    def start(self):
        self.tick()
        def observe():
            while len(self.rows) < self.limit and not self.event.wait(2):
                self.tick()
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        return {'samples': list(self.rows), 'minimum_wait_between_samples_seconds': 2,
                'sample_limit': self.limit, 'continuous_peak_qualified': False,
                'scope': 'Current process RSS/CPU; whole GPU includes other processes; best effort.'}


def load_model(torch, model_class, selected, cache):
    kwargs = dict(revision=runtime.REVISION, trust_remote_code=False,
                  dtype=torch.float16, attn_implementation='sdpa', cache_dir=cache)
    if selected['mode'] == 'fp16':
        return model_class.from_pretrained(runtime.MODEL_ID, **kwargs).to('cuda:0').eval()
    from transformers import BitsAndBytesConfig
    quant = dict(selected['quantization'])
    if selected['mode'] == 'nf4':
        quant.update(bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_storage=torch.uint8)
    model = model_class.from_pretrained(runtime.MODEL_ID, **kwargs,
                quantization_config=BitsAndBytesConfig(**quant), device_map={'': 0}).eval()
    expected = 'Linear8bitLt' if selected['mode'] == 'int8' else 'Linear4bit'
    if not any(type(module).__name__ == expected for _, module in model.named_modules()):
        raise ValueError('Actual quantized modules required')
    return model


def identity(result, dataset, selected):
    """Bind task source/protocol, not adaptive response-dependent call sequences."""
    return {key: result[key] for key in ('source_commit', 'manifest_sha256', 'data_file_sha256',
        'frozen_source', 'model_id', 'model_revision', 'split', 'settings', 'evaluation_provider',
        'runtime_source_sha256', 'packages', 'python', 'cuda', 'gpu', 'gpu_capability', 'gpu_uuid')} | {
        'source_freeze_sha256': selected['source_freeze_sha256'],
        'adapter': result.get('adapter'), 'reference_settings': selected['reference_settings'],
        'tasks': [e['id'] for e in dataset['episodes'] if e['split'] == result['split']],
        'mind_tasks': [e['id'] for e in dataset['mind_cases']]}


def paired(baseline, candidate):
    if baseline.get('status') != 'measured' or candidate.get('status') != 'measured':
        raise ValueError('Complete measured runs required')
    required = {'source_commit', 'manifest_sha256', 'data_file_sha256', 'frozen_source',
                'model_id', 'model_revision', 'split', 'settings', 'evaluation_provider',
                'runtime_source_sha256', 'packages', 'python', 'cuda', 'gpu', 'gpu_capability', 'gpu_uuid',
                'source_freeze_sha256', 'adapter', 'reference_settings', 'tasks', 'mind_tasks'}
    if (set(baseline['comparison_identity']) != required or
            json.dumps(baseline['comparison_identity'], sort_keys=True, allow_nan=False) !=
            json.dumps(candidate['comparison_identity'], sort_keys=True, allow_nan=False)):
        raise ValueError('Source/task/prompt/model/adapter identity mismatch')
    for result in (baseline, candidate):
        selected = dict(result['precision_profile'])
        if selected.pop('source_freeze_sha256', None) != result['comparison_identity']['source_freeze_sha256']:
            raise ValueError('Profile source freeze identity mismatch')
        check_profile(selected)
        for key in required - {'tasks', 'mind_tasks', 'source_freeze_sha256', 'reference_settings', 'adapter'}:
            if result.get(key) != result['comparison_identity'][key]:
                raise ValueError('Reported metadata differs from bound comparison identity')
        if result.get('adapter') != result['comparison_identity']['adapter']:
            raise ValueError('Reported adapter differs from bound comparison identity')
        ids = [r['episode'] for r in result['episodes']]
        if not ids or len(ids) != len(set(ids)) or ids != result['comparison_identity']['tasks']:
            raise ValueError('Duplicate/missing/reordered tasks')
        mind_ids = [r['id'] for r in result['mind_cases']]
        if len(mind_ids) != len(set(mind_ids)) or mind_ids != result['comparison_identity']['mind_tasks']:
            raise ValueError('Duplicate/missing/reordered Mind tasks')
        if result['summary'] != runtime.summarize(result['episodes'], result['mind_cases']):
            raise ValueError('Independent summary mismatch')
    if baseline['precision_profile']['mode'] != 'fp16' or candidate['precision_profile']['mode'] == 'fp16':
        raise ValueError('FP16 versus quantized pair required')
    for a, b in zip(baseline['episodes'], candidate['episodes']):
        if any(a[key] != b[key] for key in ('episode', 'family', 'language', 'split')):
            raise ValueError('Task identity mismatch')
    cost_keys = ('load_seconds', 'wall_seconds', 'cpu_process_seconds', 'model_generation_seconds',
                 'memory_before_load', 'memory_loaded', 'memory_final', 'max_sampled_device_used_bytes',
                 'residency_loaded', 'unique_physical_cache_bytes', 'process_cost', 'telemetry_errors')
    def cost(result):
        return {key: result.get(key) for key in cost_keys} | {'call_latency': [
            {key: row.get(key) for key in ('call_index', 'cold_first_call', 'wall_seconds',
                                         'preprocessing_seconds', 'generation_seconds')}
            for row in result.get('calls', [])]}
    return {'baseline_summary': baseline['summary'], 'candidate_summary': candidate['summary'],
            'physical_device_comparison_qualified': baseline['comparison_identity']['gpu_uuid'] is not None,
            'physical_device_comparison_scope': 'Same resolved physical UUID' if baseline['comparison_identity']['gpu_uuid'] is not None
                else 'Physical device identity unknown; matched physical device cost is unqualified',
            'cost': {'fp16': cost(baseline), candidate['precision_profile']['mode']: cost(candidate)},
            'per_episode': [{'episode': a['episode'], 'language': a['language'], 'family': a['family'],
                'baseline_complete': bool(a['metrics']['success']),
                'candidate_complete': bool(b['metrics']['success'])}
                for a, b in zip(baseline['episodes'], candidate['episodes'])],
            'adoption_qualified': False, 'equal_accuracy_qualified': False,
            'scope': 'Descriptive H19 authored bilingual closed-loop comparison; no native/release qualification.'}


def run(mode, source_freeze_sha256, bundle, manifest_sha256, output, font_path, source_commit,
        *, provider='colab', split='dev', adapter=None, adapter_manifest_sha256=None, adapter_identity=None):
    runtime.evaluation_guard(provider)
    selected = profile(mode)
    verify_sources(source_freeze_sha256)
    check_cache(CACHE)
    for name, expected in PINS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Exact precision dependency pins required')
    selected['source_freeze_sha256'] = source_freeze_sha256
    supplied = (adapter, adapter_manifest_sha256, adapter_identity)
    if any(v is not None for v in supplied):
        if any(v is None for v in supplied):
            raise ValueError('Complete external adapter identity required')
        from evaluation.development_adapter import verified_adapter
        with verified_adapter(adapter, adapter_identity, adapter_manifest_sha256) as prepared:
            return runtime._run(bundle, manifest_sha256, output, font_path, source_commit,
                split=split, cache_dir=CACHE, prepared_adapter=prepared, provider=provider, precision=selected)
    return runtime._run(bundle, manifest_sha256, output, font_path, source_commit,
        split=split, cache_dir=CACHE, prepared_adapter=None, provider=provider, precision=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('mode', 'source-freeze-sha256', 'bundle', 'manifest-sha256', 'output', 'font-path', 'source-commit'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--provider', choices=['colab', 'kaggle'], default='colab')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--adapter')
    parser.add_argument('--adapter-manifest-sha256')
    parser.add_argument('--adapter-identity', help='JSON object with full externally pinned retained identity')
    args = vars(parser.parse_args())
    if args['adapter_identity'] is not None:
        args['adapter_identity'] = json.loads(args['adapter_identity'])
    result = run(**args)
    print(json.dumps({'status': result['status'], 'summary': result['summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
