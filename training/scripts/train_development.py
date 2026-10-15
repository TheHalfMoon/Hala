"""One predeclared shared Mind/Type epoch. Remote Colab only; no upload credentials."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from evaluation.development_runtime import (MODEL_ID, REVISION, SETTINGS, VERSIONS,
                                           check_digest, memory_sample, remote_guard, summarize)
from training.adapter_archive import create_bundle, verify_bundle, TARGET_PATTERN
from training.artifacts import strict_json
from training.checkpoints import atomic_json, publish_checkpoint
from training.development_data import encode_row, load_rows
from training.scripts.train_adapter import create_training_scaler

SCHEDULE = {'seed': 42, 'epochs': 1, 'batch_size': 1, 'gradient_accumulation': 1,
            'learning_rate': 5e-5, 'rank': 4, 'alpha': 8, 'dropout': 0,
            'precision': 'fp16', 'initial_scale': 1.0, 'row_order': 'authored',
            'optimizer': 'AdamW', 'max_grad_norm': 1.0}
PACKAGES = dict(VERSIONS, peft='0.20.0')


def record_memory(record, key, torch, sample=None):
    """Best-effort telemetry must not erase numerical training evidence."""
    try:
        record[key] = (sample or memory_sample)(torch)
    except Exception as error:
        record[key] = None
        record.setdefault('telemetry_errors', {})[key] = type(error).__name__[:80]


def record_completed_step(result, metric, torch, path, sample=None):
    result['metrics'].append(metric)
    result['status'] = 'running'
    record_memory(metric, 'memory', torch, sample)
    atomic_json(path, result)


def finalize_training_result(result, path, torch, start, sample=None):
    result['wall_seconds'] = time.perf_counter() - start
    record_memory(result, 'memory_final', torch, sample)
    samples = [result.get(k) for k in ('memory_before_load', 'memory_loaded', 'memory_final')]
    samples += [m.get('memory') for m in result['metrics']]
    used = [m['device_used_bytes'] for m in samples if isinstance(m, dict) and
            type(m.get('device_used_bytes')) is int and m['device_used_bytes'] >= 0]
    result['max_sampled_device_used_bytes'] = max(used) if used else None
    atomic_json(path, result)


def bounded_bytes(path, limit=16 * 1024 * 1024):
    path = Path(path)
    size = path.stat().st_size
    if not path.is_file() or size > limit:
        raise ValueError('Bounded regular input file required')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) != size or len(raw) > limit:
        raise ValueError('Input changed during bounded read')
    return raw


def digest(path):
    return hashlib.sha256(bounded_bytes(path)).hexdigest()


def read_json(path, limit=16 * 1024 * 1024, expected_sha256=None):
    raw = bounded_bytes(path, limit)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('External JSON digest mismatch')
    return strict_json(raw.decode('utf-8'))


def validate_baseline(value, manifest, manifest_sha256):
    from evaluation import development_runtime
    check_digest(value.get('source_commit'), 40)
    expected = {'schema': 'hala.development.baseline.v1', 'status': 'measured',
                'model_id': MODEL_ID, 'model_revision': REVISION, 'settings': SETTINGS,
                'manifest_sha256': manifest_sha256, 'data_file_sha256': manifest['files']['dataset.json'],
                'frozen_source': manifest['source'], 'training_runs': 0, 'adapter_loaded': False,
                'runtime_source_sha256': digest(development_runtime.__file__), 'split': 'dev'}
    for key, wanted in expected.items():
        if json.dumps(value.get(key), sort_keys=True) != json.dumps(wanted, sort_keys=True):
            raise ValueError('Unchanged actual baseline identity mismatch: ' + key)
    if any(value.get('packages', {}).get(p, '').split('+')[0] != v for p, v in VERSIONS.items()):
        raise ValueError('Baseline package mismatch')
    from development.data import build_dataset
    data = build_dataset()
    episodes, mind = value.get('episodes', []), value.get('mind_cases', [])
    expected_ids = {e['id'] for e in data['episodes'] if e['split'] == 'dev'}
    if (len(episodes) != len(expected_ids) or {e['episode'] for e in episodes} != expected_ids or
            any(e['split'] != 'dev' or not e['calls'] for e in episodes) or
            len(mind) != len(data['mind_cases']) or {m['id'] for m in mind} != {m['id'] for m in data['mind_cases']} or
            not value.get('calls') or not all('raw' in c or 'error_type' in c for c in value['calls'])):
        raise ValueError('Complete actual baseline observations required')
    summary = summarize(episodes, mind)
    if value.get('summary') != summary:
        raise ValueError('Baseline summary differs from raw per-episode records')
    # Only development episode deficits trigger training. Held-out Mind test
    # diagnostics never select the schedule or authorize a training intervention.
    all_dev = summary['all']
    if not (all_dev['episodes_complete'] < all_dev['episodes'] or
            all_dev['mind_correct'] < all_dev['mind_calls']):
        raise ValueError('Actual development deficit required before training')
    return summary


def verified_adapter_path(directory, identity, manifest_sha256):
    """Fresh-session seam: verify external anchors BEFORE PeftModel.from_pretrained."""
    manifest = verify_bundle(directory, identity, manifest_sha256)
    if manifest['kind'] != 'trained_adapter':
        raise ValueError('Trained adapter required')
    return Path(directory).resolve()


def preflight(bundle, manifest_sha256, baseline, baseline_sha256, source_commit,
              probe, probe_sha256, probe_commit, repository):
    for h, length in ((manifest_sha256, 64), (baseline_sha256, 64), (source_commit, 40),
                      (probe_sha256, 64), (probe_commit, 40)):
        check_digest(h, length)
    if repository != 'TheHalfMoon/Hala':
        raise ValueError('Authorized artifact repository required')
    baseline_value = read_json(baseline, expected_sha256=baseline_sha256)
    rows, manifest = load_rows(bundle, manifest_sha256)
    validate_baseline(baseline_value, manifest, manifest_sha256)
    identity = {'base_model': MODEL_ID, 'base_revision': REVISION, 'source_commit': source_commit,
                'data_sha256': manifest['files']['dataset.json'], 'evaluation_sha256': baseline_sha256}
    archived = verify_bundle(probe, identity, probe_sha256)
    if archived['kind'] != 'pretraining_roundtrip':
        raise ValueError('Verified harmless pretraining immutable-commit roundtrip required')
    return rows, manifest, identity, baseline_value['source_commit']


def train(bundle, manifest_sha256, baseline, baseline_sha256, source_commit,
          probe, probe_sha256, probe_commit, repository, output):
    remote_guard()
    if os.environ.get('HALA_REMOTE_TRAINING_ALLOWED') != '1':
        raise ValueError('Explicit remote training opt-in required')
    rows, manifest, identity, baseline_commit = preflight(bundle, manifest_sha256, baseline,
        baseline_sha256, source_commit, probe, probe_sha256, probe_commit, repository)
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all', '--',
        'training', 'development', 'evaluation/development_runtime.py'], cwd=root,
        capture_output=True, text=True, check=True).stdout
    if head != source_commit or dirty:
        raise ValueError('Exact published clean training source required')
    for name, version in PACKAGES.items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise ValueError('Exact package pins required: ' + name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = '/tmp/hala-h21-training-cache'
    os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', HF_HOME=cache,
                      HF_HUB_CACHE=cache, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    import peft
    import transformers
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    for name, actual in {'torch': torch.__version__, 'peft': peft.__version__,
                         'transformers': transformers.__version__}.items():
        if actual.split('+')[0] != PACKAGES[name]:
            raise ValueError('Imported package mismatch')
    if not torch.cuda.is_available():
        raise ValueError('Actual remote CUDA required')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'schema': 'hala.development.training.v1', 'status': 'preparing', 'identity': identity,
              'schedule': dict(SCHEDULE, steps=len(rows)), 'baseline_source_commit': baseline_commit,
              'probe_commit': probe_commit, 'probe_manifest_sha256': probe_sha256,
              'repository': repository, 'manifest_sha256': manifest_sha256,
              'training_script_sha256': digest(__file__), 'metrics': [], 'quality_qualified': False,
              'native_qualified': False, 'upload_required_before_reset': True}
    atomic_json(output / 'result.json', result)
    start = time.perf_counter()
    model = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        for row in rows:
            encode_row(row, processor, bundle, Image.open)
        record_memory(result, 'memory_before_load', torch)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION,
            trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda:0')
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=TARGET_PATTERN,
            lora_dropout=0, task_type='CAUSAL_LM', bias='none', revision=REVISION))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if (len(trainable) != 112 or sum(p.numel() for _, p in trainable) != 802816 or
                any('lora_' not in n or 'language_model' not in n or 'visual' in n for n, _ in trainable)):
            raise ValueError('Complete language-only rank4 LoRA with frozen base/vision required')
        params = [p for _, p in trainable]
        for p in params:
            p.data = p.data.float()
        initial = [p.detach().cpu().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=5e-5)
        scaler = create_training_scaler(torch, 'fp16')
        torch.cuda.reset_peak_memory_stats(0)
        record_memory(result, 'memory_loaded', torch)
        for index, row in enumerate(rows, 1):
            before = [p.detach().cpu().clone() for p in params]
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = encode_row(row, processor, bundle, Image.open)
            labels = torch.tensor([labels], dtype=inputs.input_ids.dtype, device='cuda:0')
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**inputs.to('cuda:0'), labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                raise ValueError('Nonfinite adapter gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, 1).item()
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Finite nonzero gradient required')
            scaler.step(optimizer)
            scaler.update()
            if not all(torch.isfinite(p).all() for p in params):
                raise ValueError('Nonfinite updated parameter')
            torch.cuda.synchronize(0)
            changed = sum(not torch.equal(a, p.detach().cpu()) for a, p in zip(before, params))
            if not changed:
                raise ValueError('Actual update required at every step')
            metric = {'step': index, 'row_id': row['id'], 'loss': loss.item(),
                'seconds': time.perf_counter()-step_start, 'gradient_norm': norm,
                'changed_tensors': changed}
            record_completed_step(result, metric, torch, output / 'result.json')
        changed = sum(not torch.equal(a, p.detach().cpu()) for a, p in zip(initial, params))
        checkpoint_identity = {'config_sha256': hashlib.sha256(json.dumps(SCHEDULE, sort_keys=True).encode()).hexdigest(),
            'dataset_sha256': identity['data_sha256'], 'manifest_sha256': manifest_sha256}
        state = dict(checkpoint_identity, step=len(rows), max_steps=len(rows),
            metrics=[{k: m[k] for k in ('step', 'loss', 'seconds')} for m in result['metrics']],
            changed_parameter_tensors=changed, peak_vram_bytes=torch.cuda.max_memory_allocated(0),
            runtime={'packages': PACKAGES, 'schedule': result['schedule']})
        def writer(stage):
            model.save_pretrained(stage / 'adapter', safe_serialization=True)
            # Local trusted checkpoint only. Never included in archival bundle.
            torch.save(dict(checkpoint_identity, step=len(rows), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), stage / 'resume.pt')
        version = publish_checkpoint(output, state, writer)
        staging = output / 'archive-source'
        staging.mkdir()
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            shutil.copyfile(version / 'adapter' / name, staging / name)
        atomic_json(staging / 'training_metadata.json', {'training_run_id': 'h21-shared-pilot',
            'steps': len(rows), 'seed': 42, 'training_script_sha256': digest(__file__)})
        archive_sha = create_bundle(staging, output / 'adapter-bundle', identity)
        verified_adapter_path(output / 'adapter-bundle', identity, archive_sha)
        result.update(status='trained_pending_durable_archive', adapter_manifest_sha256=archive_sha,
                      changed_parameter_tensors=changed, checkpoint=version.name)
    except Exception as error:
        result.update(status='blocked', error_type=type(error).__name__)
        raise
    finally:
        finalize_training_result(result, output / 'result.json', torch, start)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'manifest-sha256', 'baseline', 'baseline-sha256', 'source-commit',
                 'probe', 'probe-sha256', 'probe-commit', 'repository', 'output'):
        parser.add_argument('--' + name, required=True)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
