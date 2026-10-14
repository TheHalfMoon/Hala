"""Bounded remote engineering smoke: continuous two steps versus real partial resume."""
import argparse
import gc
import math
import os
from pathlib import Path
import platform
import time

from training.artifacts import sha256_file
from training.checkpoints import atomic_json, load_checkpoint, validate_version
from training.scripts.train_adapter import load_json, train, QualificationInterrupted

EXPECTED = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct',
            'revision': '89644892e4d85e24eaac8bacfd4f463576704203',
            'precision': 'fp16', 'max_steps': 2, 'batch_size': 1,
            'gradient_accumulation': 1, 'lora_rank': 4, 'learning_rate': 0.00005,
            'platform': 'colab', 'download_remote_weights': True}


def validate_smoke_config(config):
    if config != EXPECTED:
        raise ValueError('Exact pinned two-step rank4 FP16 smoke configuration required')


def compare_values(left, right, *, atol=1e-6, rtol=1e-5):
    """Compare complete nested state; tensor RNG comparisons use zero tolerance."""
    mismatches, near, differences = [], [], []
    def visit(a, b, path):
        if hasattr(a, 'detach') or hasattr(b, 'detach'):
            import torch
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                mismatches.append(path)
                return
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                mismatches.append(path)
                return
            if not torch.equal(a, b):
                delta = (a.double() - b.double()).abs().max().item() if a.numel() else 0
                differences.append(delta)
                if a.is_floating_point() and torch.allclose(a, b, atol=atol, rtol=rtol):
                    near.append(path)
                else:
                    mismatches.append(path)
        elif type(a) is not type(b):
            mismatches.append(path)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                mismatches.append(path + ':keys')
            else:
                for key in a:
                    visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(path + ':length')
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f'{path}[{index}]')
        elif isinstance(a, float):
            if not math.isfinite(a) or not math.isfinite(b):
                mismatches.append(path)
            elif a != b:
                differences.append(abs(a - b))
                (near if math.isclose(a, b, abs_tol=atol, rel_tol=rtol) else mismatches).append(path)
        elif a != b:
            mismatches.append(path)
    visit(left, right, 'state')
    return {'exact': not mismatches and not near, 'within_tolerance': not mismatches,
            'atol': atol, 'rtol': rtol, 'max_absolute_difference': max(differences, default=0),
            'mismatch_paths': mismatches[:100], 'tolerance_paths': near[:100]}


def snapshot_state(output):
    pointer = load_json(Path(output) / 'latest.json')
    version = Path(output) / pointer['version']
    state, _ = validate_version(version)
    return load_checkpoint(output, state)


def checkpoint_evidence(output):
    version, state = snapshot_state(output)
    _, artifacts = validate_version(version, state)
    return {'version': version.name, 'step': state['step'],
            'checkpoint_sha256': sha256_file(version / 'checkpoint.json'),
            'artifacts': artifacts, 'runtime': state['runtime'], 'metrics': state['metrics']}


def compare_checkpoints(continuous, resumed):
    import torch
    from safetensors.torch import load_file
    first, a = snapshot_state(continuous)
    second, b = snapshot_state(resumed)
    for key in ('config_sha256', 'dataset_sha256', 'manifest_sha256', 'max_steps', 'step'):
        if a[key] != b[key] or a['step'] != 2:
            raise ValueError('Comparison checkpoint identity or final step mismatch')
    adapter_a = load_file(str(first / 'adapter/adapter_model.safetensors'))
    adapter_b = load_file(str(second / 'adapter/adapter_model.safetensors'))
    state_a = torch.load(first / 'resume.pt', map_location='cpu', weights_only=True)
    state_b = torch.load(second / 'resume.pt', map_location='cpu', weights_only=True)
    if state_a.keys() != state_b.keys():
        raise ValueError('Resume snapshot field mismatch')
    comparisons = {'adapter': compare_values(adapter_a, adapter_b)}
    comparisons['loss_trajectory'] = compare_values(
        [{'step': row['step'], 'loss': row['loss']} for row in a['metrics']],
        [{'step': row['step'], 'loss': row['loss']} for row in b['metrics']])
    for name in state_a:
        strict = name in ('torch_rng', 'cuda_rng', 'scaler')
        comparisons[name] = compare_values(state_a[name], state_b[name],
                                           atol=0 if strict else 1e-6,
                                           rtol=0 if strict else 1e-5)
    return {'exact': all(row['exact'] for row in comparisons.values()),
            'within_tolerance': all(row['within_tolerance'] for row in comparisons.values()),
            'components': comparisons}


def release_runtime():
    gc.collect()
    # CPU control tests never need model packages.
    import sys
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()


def qualify(config, examples, manifest, baseline, output, *, trainer=None):
    real = trainer is None
    if real and (platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_TRAINING_ALLOWED') != '1'):
        raise ValueError('Explicit remote Linux opt-in required; local weights forbidden')
    validate_smoke_config(load_json(config))
    input_paths = {'config': config, 'examples': examples, 'manifest': manifest, 'baseline': baseline}
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    def guard_inputs():
        if any(sha256_file(path) != input_hashes[name] for name, path in input_paths.items()):
            raise ValueError('Qualification input identity changed between trajectories')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fresh qualification output required')
    output.mkdir(parents=True, exist_ok=True)
    runner = train if real else trainer
    start = time.perf_counter()
    continuous, resumed = output / 'continuous', output / 'resumed'
    runner(config, examples, manifest, baseline, continuous)
    release_runtime()
    guard_inputs()
    interrupted = False
    try:
        runner(config, examples, manifest, baseline, resumed, stop_after_step=1)
    except QualificationInterrupted as stopped:
        if stopped.step != 1:
            raise ValueError('Wrong interruption step') from None
        interrupted = True
    # Exception and its traceback are out of scope before freeing the old model.
    release_runtime()
    if not interrupted:
        raise ValueError('Required deliberate interruption did not occur')
    _, partial = snapshot_state(resumed)
    if partial['step'] != 1 or partial['max_steps'] != 2:
        raise ValueError('Required partial snapshot; completed finalization is not resume')
    first_snapshot = checkpoint_evidence(resumed)
    guard_inputs()
    runner(config, examples, manifest, baseline, resumed, resume=True)
    release_runtime()
    guard_inputs()
    comparison = compare_checkpoints(continuous, resumed)
    continuous_evidence = checkpoint_evidence(continuous)
    resumed_evidence = checkpoint_evidence(resumed)
    reload = resumed_evidence.get('runtime', {}).get('actual_reload') is True and \
        resumed_evidence.get('runtime', {}).get('resumed_from_step') == 1
    report = {'schema': 'hala.lora.runtime-qualification.v1',
              'evidence': 'real_gpu_engineering_smoke' if real else 'cpu_control_test_unqualified',
              'runtime_qualified': bool(real and reload and comparison['within_tolerance']),
              'adapter_quality_qualified': False, 'arabic_remediation_claimed': False,
              'elapsed_seconds': time.perf_counter() - start,
              'inputs': input_hashes,
              'source': {name: sha256_file(Path(__file__).parents[2] / name) for name in
                         ('training/scripts/qualify_lora.py', 'training/scripts/train_adapter.py',
                          'training/checkpoints.py', 'training/artifacts.py', 'training/masking.py')},
              'first_partial_snapshot': first_snapshot, 'continuous': continuous_evidence,
              'resumed': resumed_evidence, 'comparison': comparison,
              'artifact_availability': 'Ephemeral remote notebook storage only; no durable adapter retention promised.',
              'consumed_microbatches_per_trajectory': 2, 'trajectories': 2}
    atomic_json(output / 'qualification-result.json', report)
    if real and not report['runtime_qualified']:
        raise ValueError('Runtime qualification failed; inspect persisted comparison report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'examples', 'manifest', 'baseline', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    qualify(args.config, args.examples, args.manifest, args.baseline, args.output)


if __name__ == '__main__':
    main()
